import pandas as pd
import ast
import re
import networkx as nx
import time
import asyncio
import json
from models import LLM, VLLM
from toolrunner import ToolRunner
from functions import *
import os as system
import multiprocessing as mp
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from multiprocessing import Pool
import yaml

class Eval:
    def __init__(self, args):
        self.args = args
        self.LLM = LLM(args)
        self.VLLM = VLLM(args)
        
        self.toolselection_prompt = yaml.safe_load(open(os.path.join(args.base_path, "prompts/ToolSelection_prompts.yaml"), encoding="utf-8"))        
        self.validation_prompt = yaml.safe_load(open(os.path.join(args.base_path, "prompts/Validation_prompts.yaml"), encoding="utf-8"))
        
        self.constraint_result = []
        self.logs = {}
        self.args = args

    def evaluate(self, scene, instruction, controller):
        self.logs["instruction"] = instruction
        print(f"INSTRUCTION: {instruction}")
        
        #get list of constraints and their type
        if "constraints" in scene.keys():
            constraints = scene["constraints"]
        else:
            raise ValueError("The instruction and the Qwery in the scene Json are different. You must match them.")
                                                    
        #tool sequence planning for each constraint
        constraint_output = [] 
        num_constraint = 0
        num_correct = 0
        fl = [0, 0] # num, corr_num
        ms = [0, 0]
        os = [0, 0]
        op = [0, 0]
        counters = {
            "Floor Layout": fl,
            "Material Selection": ms,
            "Object Selection": os,
            "Object Placement": op
        }
        for i, item in enumerate(constraints):
            curr_constraint_log_out = {}
            curr_constraint_logs = {}

            print(f"    Constraint {i}: ", item)
            curr_constraint_log_out[f"constraint"] = item
            curr_constraint_logs["constraint"] = item

            start_time = time.time()

            tool_sequence, reasoning =  self.get_tools(instruction, item["constraint"], item["constraint_type"], constraint_output, curr_constraint_logs, self.args) #tool_sequence is a list.
            #unnecessary_tools = self.get_unnecessary_tools(instruction, item["constraint"], tool_sequence, reasoning)
            
            validity_log = None
            tool_execution_logs = None
            if not tool_sequence:
                text_information["previous constraint result descriptions"] = constraint_output
                text_information["Information"] = f"Reasoning Behind No Tool Use: {reasoning}"
                image_information = {}
                validity, description, validity_log = self.validate(instruction, item["constraint"], item["constraint_type"], text_information, image_information)
                
            else:
                tool_runner = ToolRunner(tool_sequence, controller, self.args)
                scene_ids = [item['id'] for key in ['objects', 'doors', 'windows', 'walls', 'rooms'] for item in scene.get(key, [])]
                scene_ids += [item['floorMaterial']['name'] for key in ['rooms'] for item in scene.get(key, [])]
                scene_ids += [item['material']['name'] for key in ['walls'] for item in scene.get(key, [])]
                text_information, image_information, tool_execution_logs, text_outputs = tool_runner.run(scene, instruction, item["constraint"], tool_sequence, constraint_output, reasoning, scene_ids, self.args.max_tool_workers)
                
                validity, description, validity_log = self.validate(instruction, item["constraint"], item["constraint_type"], text_information, image_information)
            
            curr_constraint_logs["tool_execution"] = tool_execution_logs
            curr_constraint_logs["validity"] = validity_log

            constraint_type = item["constraint_type"]

            curr_constraint_log_out["validity"] = validity
            curr_constraint_log_out["description"] = description

            if validity == "True":
                print(f"    Constraint {i} is not violated!")
                # print(f"    Result Description: {description}")
                constraint_output.append(description)
                num_constraint += 1
                num_correct += 1
                if constraint_type in counters:
                    counters[constraint_type][0] += 1
                    counters[constraint_type][1] += 1
                self.constraint_result.append(True)
            else:
                print(f"    Constraint {i} is violated!")
                # print(f"    Result Description: {description}")
                constraint_output.append(description)
                num_constraint += 1
                if constraint_type in counters:
                    counters[constraint_type][0] += 1
                self.constraint_result.append(False)
            # else:
            #     self.constraint_result.append(False)
            #     print(f"    Constraint {i} is violated!")
            #     description = "Reliability check failed: A prerequisite constraint that must be satisfied in order for the current constraint to hold has been violated."
            #     print(f"    Result Description: {description}")
            #     num_constraint += 1
            #     if constraint_type in counters:
            #         counters[constraint_type][0] += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(f"\033[93m    Constraint_{i} Execution Time: {elapsed_time:.2f} seconds\033[0m")
            # print(f"    Constraint_{i} Execution Time: {elapsed_time:.2f} seconds")
            curr_constraint_logs["execution_time"] = elapsed_time
            self.logs[f"constraint_{i}"] = curr_constraint_log_out

        return num_constraint, num_correct, fl, ms, os, op

    def get_tools(self, instruction, constraint, constraint_type, constraint_output, curr_constraint_logs, args):
        type_prefixes = {
            "Floor Layout": "FL",
            "Material Selection": "MS",
            "Object Selection": "OS",
            "Object Placement": "OP",
        }

        if constraint_type not in type_prefixes:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        prefix = type_prefixes[constraint_type]

        system_prompt = self.toolselection_prompt[f'{prefix}_system']#[self.toolselection_prompt['type'] == f'{prefix}_system']['prompt'].iloc[0]
        human_prompt = self.toolselection_prompt[f'{prefix}_human']#[self.toolselection_prompt['type'] == f'{prefix}_human']['prompt'].iloc[0]

        human_prompt = human_prompt.replace('$PREVIOUS_CONSTRAINT_OUTPUTS$', str(constraint_output)).replace('$CURRENT_CONSTRAINT$', str(constraint))

        max_retries = 10
        last_err = None
        for attempt in range(1, max_retries + 1):
            output =  self.LLM.generate(system_prompt, human_prompt, my_temp=self.args.temperature_llm+(attempt*0.1))
            try:
                reasoning, tool_names = self.split_reasoning_output(output)
                break
            except ValueError as e:
                last_err = e
                print(f"[Parsing failed] attempt {attempt}/{max_retries}: {e}")
        else:
            # raise RuntimeError(
            #     f"Parsing failed after {max_retries} attempts. Last error: {last_err}"
            # )
            curr_constraint_logs["tool_sequence"] = {
                "system_prompt": system_prompt,
                "human_prompt": human_prompt,
                "sequence": [],
                "reasoning": f"Tool Selection Failed: {output}\nError: {last_err}"
            }
            return [], output

        curr_constraint_logs["tool_sequence"] = {
            "system_prompt": system_prompt,
            "human_prompt": human_prompt,
            "sequence": tool_names,
            "reasoning": reasoning
        }

        return tool_names, reasoning

    
    def validate(self, instruction, constraint, constraint_type, text_information, image_information): #next constraints는 굳이 안넣음(넣어야할수도). 
        type_prefixes = {
            "Floor Layout": "FL",
            "Material Selection": "MS",
            "Object Selection": "OS",
            "Object Placement": "OP",
        }

        if constraint_type not in type_prefixes:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        prefix = type_prefixes[constraint_type]

        system_prompt = self.validation_prompt[f'{prefix}_system']
        human_prompt = self.validation_prompt[f'{prefix}_human']

        human_prompt = human_prompt.replace('$INSTRUCTION$', str(instruction)).replace('$CURRENT_CONSTRAINT$', str(constraint)).replace('$INFORMATION$', str(text_information))
        
        #output will be in the format of <<true/false, description>> we will separate this into answer, description.
        max_retries = 5  # 원하는 최대 재시도 횟수

        for attempt in range(max_retries):
            if len(image_information) == 0:
                output = self.LLM.generate(system_prompt, human_prompt, my_temp = self.args.temperature_llm+(attempt*0.1))
            else:
                output = self.VLLM.generate(system_prompt, human_prompt, image_information, my_temp = self.args.temperature_vllm+(attempt*0.1))

            match = re.search(r"<<(True|False),\s*(.*?)>>", output.strip(), re.DOTALL)
            if match:
                break
            else:
                print(f"[Warning] Wrong outpur for Validation (attempt {attempt + 1}/{max_retries}): {output!r}")
                if attempt == max_retries - 1:
                    raise ValueError("Final answer format is incorrect after retries. Must match <<True/False, reason>>.")
        else:
            # for-else 구조를 쓰고 싶을 때(옵션)
            raise RuntimeError("Did not output the right format for Validation even after 5 attempts.")

        answer = match.group(1)       # "True" 또는 "False"
        description = match.group(2)  # 이유 텍스트
        # if len(image_information) == 0:
        #     output = self.LLM.generate(system_prompt, human_prompt)
        # else:
        #     output = self.VLLM.generate(system_prompt, human_prompt, image_information)

        # match = re.search(r"<<(True|False),\s*(.*?)>>", output.strip(), re.DOTALL)
        # if not match:
        #     raise ValueError("Final answer format is incorrect. Must match <<True/False, reason>>.")
        # answer = match.group(1)
        # description = match.group(2)

        validity = {
            "txt_input": [constraint, text_information],
            "img_input": image_information,
            "system_prompt": system_prompt,
            "human_prompt": human_prompt,
            "output": output
        }

        return answer, description, validity
    
    def split_reasoning_output(self, text):
        """
        Parse the LLM output text into reasoning and tool names list robustly.
        Supports slight formatting variations.
        """
        # Normalize text first (remove excessive spaces around colons)
        text = re.sub(r'\s*:\s*', ': ', text.strip())

        # More flexible regex pattern
        reasoning_pattern = r"Chain-of-Thoughts?: (.*?)\s*Tool Sequence:"
        output_pattern = r"Tool Sequence: (.*)"

        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        output_match = re.search(output_pattern, text, re.DOTALL)

        if reasoning_match and output_match:
            reasoning = reasoning_match.group(1).strip()
            output_raw = output_match.group(1).strip()

            # Try parsing output
            try:
                tool_names = ast.literal_eval(output_raw)
                if not isinstance(tool_names, list):
                    raise ValueError("Parsed output is not a list.")
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Failed to parse tool names. Output content:\n{output_raw}\nError: {e}")
            
            return reasoning, tool_names
        
        else:
            raise ValueError(
                f"Failed to find proper 'Chain-of-Thought' and 'Output' sections.\n"
                f"Received text:\n{text}"
            )


    def get_unnecessary_tools(self, instruction, constraint, tool_sequence, reasoning):
        system_prompt = self.toolfiltering_prompt["system"]
        human_prompt = self.toolfiltering_prompt["human"]

        human_prompt = human_prompt.replace("$INSTRUCTION$", str(instruction)).replace("$CONSTRAINT$", str(constraint)).replace("$TOOL_SEQUENCE$", str(tool_sequence)).replace("$REASONING$", str(reasoning))

        output = self.LLM.generate(system_prompt, human_prompt)
        try:
            unnecessary_tools = ast.literal_eval(output.split("Unnecessary Tools: ")[1].strip())
            unnecessary_tools = list(set(unnecessary_tools))
        except Exception as e:
            print(e)
            print("Invalid Output format for Tool Filtering:")
            print(output)
            return []
                
        return unnecessary_tools


    def remove_unnecessary_text_information(self, text_information, unnecessary_tools):
        for tool in unnecessary_tools:
            if tool in text_information.keys():
                text_information.pop(tool)

        return text_information

