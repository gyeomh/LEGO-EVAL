import networkx as nx
import ast
import pandas as pd
from models import LLM
import functions
from multiprocessing import Pool, Manager
import time
import asyncio
import re
from typing import Dict
import yaml
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"
# OBJAVERSE_ASSET_DIR = '/root/mount/hwangbo/.objathor-assets/2023_09_23/assets'

def execute_tool(task):
    tool_log = dict.fromkeys(("tool_name", "output", "system_prompt", "human_prompt", "reasoning"))
    try:
        tool_name, scene, args, constraint, tool_sequence, text_info, images, past_obj_args, past_mat_args, my_controller, system_args = task
        tool_fn = {
            "get_room_list": functions.get_room_list,
            "get_room_info": functions.get_room_info,
            "get_window_list": functions.get_window_list,
            "get_window_info": functions.get_window_info,
            "get_door_list": functions.get_door_list,
            "get_door_info": functions.get_door_info,
            "get_wall_list": functions.get_wall_list,
            "get_wall_info": functions.get_wall_info,
            "get_object_list": functions.get_object_list,
            "get_object_info": functions.get_object_info,
            "get_topdown_scene": functions.get_topdown_scene,
            "get_topdown_room": functions.get_topdown_room,
            "get_material_image": functions.get_material_image,
            "get_wall_scene": functions.get_wall_scene,
            "get_multiview_rendered_object": functions.get_multiview_rendered_object,
            "get_multiview_scene_object": functions.get_multiview_scene_object,
            "get_spatial_relation": functions.get_spatial_relation,
            "get_topdown_object": functions.get_topdown_object,
            "get_property_verification": functions.get_property_verification,
            "get_object_match": functions.get_object_match,
            "get_frontview_object": functions.get_frontview_object,
            "get_property_description": functions.get_property_description,
            # "call_llm": functions.call_llm,
            # "call_vllm": functions.call_vllm,
            # "get_multiview_scene": functions.get_multiview_scene,
        }.get(tool_name)
        tool_log["tool_name"] = tool_name

        if tool_name in ['get_room_list', 'get_topdown_scene', 'get_multiview_scene']:
            result = tool_fn(scene, system_args, my_controller)
            tool_log["output"] = result

        elif tool_name in ['get_property_verification', 'get_object_match', 'get_property_description']: ### 모델 호출 함수는 tool_log를 직접 전달해서 프롬프트, 입출력 저장
            if (past_mat_args is not None) and (past_obj_args is not None):
                past_args = past_mat_args + past_obj_args
                result = tool_fn(scene, constraint, past_args, images, 'all', tool_log, system_args)
            elif past_obj_args is not None:
                result = tool_fn(scene, constraint, past_obj_args, images, 'obj', tool_log, system_args)
            else:
                result = tool_fn(scene, constraint, past_mat_args, images, 'mat', tool_log, system_args)

        else:
            if tool_name in ['get_topdown_room', 'get_wall_scene', 'get_multiview_rendered_object', 'get_multiview_scene_object', 'get_topdown_object', 'get_spatial_relation', 'get_frontview_object']:
                result = tool_fn(scene, args, system_args, my_controller)
            else:
                result = tool_fn(scene, args, system_args)
            tool_log["output"] = result

    except Exception as e:
        import traceback
        print()
        print(f"error with {tool_name}")
        print(e)
        print()
        return tool_name, {"error": f"{repr(e)}\n{traceback.format_exc()}"}, tool_log
    
    return tool_name, result, tool_log


# def split_reasoning_output(text):
#         """
#         Parse the LLM output text into reasoning and tool names list robustly.
#         Supports slight formatting variations.
#         """
#         # Normalize text first (remove excessive spaces around colons)
#         text = re.sub(r'\s*:\s*', ': ', text.strip())

#         # More flexible regex pattern
#         reasoning_pattern = r"Chain-of-Thought?: (.*?)\s*Arguments:"
#         output_pattern = r"Arguments: (.*)"

#         reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
#         output_match = re.search(output_pattern, text, re.DOTALL)

#         if reasoning_match and output_match:
#             reasoning = reasoning_match.group(1).strip()
#             output_raw = output_match.group(1).strip()

#             # Try parsing output
#             try:
#                 tool_args = ast.literal_eval(output_raw)
#                 if not isinstance(tool_args, list):
#                     raise ValueError("Parsed output is not a list.")
#             except (ValueError, SyntaxError) as e:
#                 raise ValueError(f"Failed to parse tool names. Output content:\n{output_raw}\nError: {e}")
            
#             return tool_args, reasoning
        
#         else:
#             raise ValueError(
#                 f"Failed to find proper 'Chain-of-Thought' and 'Output' sections.\n"
#                 f"Received text:\n{text}"
#             )

def split_reasoning_output(input_data):
    def split_reasoning_output_fallback(text):
        text = re.sub(r'\s*:\s*', ': ', text.strip())
        reasoning_pattern = r"Chain-of-Thought?: (.*?)\s*Arguments:"
        output_pattern = r"Arguments: (.*)"
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        output_match = re.search(output_pattern, text, re.DOTALL)

        if reasoning_match and output_match:
            reasoning = reasoning_match.group(1).strip()
            output_raw = output_match.group(1).strip()
            try:
                tool_args = ast.literal_eval(output_raw)
                if not isinstance(tool_args, list):
                    raise ValueError("Parsed output is not a list.")
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Failed to parse tool names. Output content:\n{output_raw}\nError: {e}")
            return tool_args, reasoning
        else:
            raise ValueError("No matching 'Chain-of-Thought' and 'Arguments' found.")

    if isinstance(input_data, str):
        try:
            return split_reasoning_output_fallback(input_data)
        except ValueError:
            pass

    try:
        if isinstance(input_data, str):
            input_data = input_data.strip("`\n ").removeprefix("json").strip()
            obj = json.loads(input_data)
        else:
            obj = input_data
        reasoning = obj.get("Chain-of-Thought", "").strip()
        tool_args = obj.get("Arguments", [])
        if not isinstance(tool_args, list):
            raise ValueError("Arguments must be a list")
        return tool_args, reasoning
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    raise ValueError("Input format not recognized. Cannot extract reasoning and arguments.")
        
def determine_arguments_task(task, sys_args):
    tool_name, instruction, constraint, tool_sequence, text_outputs, constraint_output, reasoning, argumentselector_prompt, scene_ids = task
    tool_type_mapping = {
        1: ['get_door_list', 'get_window_list', 'get_wall_list', 'get_object_list'],
        2: ['get_room_info', 'get_door_info', 'get_window_info', 'get_wall_info', 'get_object_info'],
        3: ['get_topdown_room', 'get_wall_scene', 'get_material_image'],
        4: ['get_multiview_rendered_object', 'get_multiview_scene_object', 'get_topdown_object', 'get_frontview_object'],
        5: ['get_spatial_relation']
    }

    tool_type = next((t for t, tools in tool_type_mapping.items() if tool_name in tools), None)
    if tool_type is None:
        return None, None, None  #몇몇 tool은 arg필요없

    system_prompt = argumentselector_prompt[f'{tool_type}_system']#[argumentselector_prompt['type'] == f'{tool_type}_system']["prompt"].iloc[0]
    human_prompt = argumentselector_prompt[f'{tool_type}_human']#[argumentselector_prompt['type'] == f'{tool_type}_human']["prompt"].iloc[0]
    human_prompt = (human_prompt
                    .replace('$INSTRUCTION$', str(instruction))
                    .replace('$PREVIOUS_CONSTRAINT_OUTPUTS$', str(constraint_output))
                    .replace('$CURRENT_CONSTRAINT$', str(constraint))
                    .replace('$TOOL_SEQUENCE$', str(tool_sequence))
                    .replace('$REASONING$', str(reasoning))
                    .replace('$PREVIOUS_TOOL_OUTPUTS$', str(text_outputs))
                    .replace('$TOOL_TO_USE$', str(tool_name)))

    max_retries = 20
    for attempt in range(max_retries):
        args_str = worker_LLM.generate(system_prompt, human_prompt, my_temp=sys_args.temperature_llm + 0.01*attempt)
        try:
            arguments, reasoning_arguments = split_reasoning_output(args_str)
            # is_all_in = all(item in total_ids for item in arguments)
            break  # 성공하면 루프 탈출
        except Exception as e:
            print(f"[Warning] split_reasoning_output failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"args_str: {args_str}")
            if attempt == max_retries - 1:
                # raise RuntimeError("split_reasoning_output failed")
                arg_log = {
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt,
                    "output": f"{args_str}\nError: {e}"
                }
                return [], "argument parse failed", arg_log

    arg_log = {
        "system_prompt": system_prompt,
        "human_prompt": human_prompt,
        "output": args_str
    }

    return arguments, reasoning_arguments, arg_log

class ToolRunner:
    def __init__(self, tool_sequence, controller, system_args):
        self.system_args = system_args
        self.build_graph(tool_sequence)
        # self.argumentselector_prompt = yaml.safe_load(open("prompts/ArgumentSelector_prompts.yaml", encoding="utf-8"))
        self.argumentselector_prompt = yaml.safe_load(open("./prompts/ArgumentSelector_prompts.yaml", encoding="utf-8"))
        self.text_outputs = {}
        self.image_outputs = {}
        self.past_obj_args = None
        self.past_material_args = None
        self.tool_execution_logs = []
        self.controller_lock = asyncio.Lock()
        self.graph_lock = asyncio.Lock()
        self.controller = controller
        self.init_worker()
    
    def init_worker(self):
        global worker_LLM
        worker_LLM = LLM(self.system_args)
        

    def build_graph(self, tool_sequence):
        G = nx.DiGraph()
        for edge in tool_sequence:
            G.add_edge(edge['from'], edge['to'])
        self.graph = G
        

    def run(self, scene, instruction, constraint, tool_sequence, constraint_output, reasoning, scene_ids, num_workers=4):
        manager = Manager()
        outputs = manager.dict()
        lock = manager.Lock()

        # calculate in-degree for dependency management
        in_degree = {node: len(list(self.graph.predecessors(node))) for node in self.graph.nodes}

        # starting with tools that have no dependencies (successors of 'START')
        ready_tools = list(self.graph.successors('START'))
        running_tools = set()
        completed_tools = set()

        pending_tasks = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while ready_tools or running_tools:
                # add new tasks for ready tools if there are available workers
                while ready_tools and len(running_tools) < num_workers:
                    tool_name = ready_tools.pop(0)
                    running_tools.add(tool_name)

                    # prepare arguments
                    prev_outputs = dict(self.text_outputs)
                    prev_constraint_output = dict(constraint_output) if isinstance(constraint_output, dict) else constraint_output
                    arg_task = (tool_name, instruction, constraint, tool_sequence, prev_outputs, prev_constraint_output, reasoning, self.argumentselector_prompt, scene_ids)

                    # synchronous argument selection
                    args_result, args_reason, arg_log = determine_arguments_task(arg_task, self.system_args)

                    if tool_name in ['get_multiview_rendered_object', 'get_multiview_scene_object']:
                        self.past_obj_args = args_result
                    if tool_name in ['get_material_image']:
                        self.past_material_args = args_result

                    # prepare tool execution task
                    task = (
                        tool_name, scene, args_result, constraint, tool_sequence,
                        self.text_outputs, self.image_outputs,
                        self.past_obj_args, self.past_material_args, self.controller, self.system_args
                    )

                    # print(f"            Tool Execution: {tool_name}")
                    # print(f"            Tool Arguments: {args_result}")
                    # print(f"            Tool Argument Selection Reasoning: {args_reason}")

                    future = executor.submit(execute_tool, task)
                    pending_tasks[tool_name] = (future, arg_log)

                # check completed futures
                completed_tasks = []
                for tool_name in list(running_tools):
                    future, arg_log = pending_tasks[tool_name]
                    if future.done():
                        try:
                            tool_name_result, result, tool_log = future.result()
                            tool_log["argument_selection"] = arg_log
                            self.tool_execution_logs.append(tool_log)
                            completed_tasks.append((tool_name_result, result))
                        except Exception as e:
                            print(f"[ERROR] Tool '{tool_name}' failed with exception: {e}")
                            running_tools.remove(tool_name)
                            continue

                        del pending_tasks[tool_name]
                        running_tools.remove(tool_name)
                        completed_tools.add(tool_name)

                # process completed tasks and update outputs/dependencies
                for tool_name, result in completed_tasks:
                    for key in list(result.keys()):
                        if key == 'delete':
                            for image in list(self.image_outputs.keys()):
                                if image in result['delete']:
                                    del self.image_outputs[image]

                        elif key == 'empty':
                            if result["empty"] == "all":
                                self.past_obj_args = None
                                self.past_material_args = None
                            elif result["empty"] == 'obj':
                                self.past_obj_args = None
                            elif result["empty"] == 'mat':
                                self.past_material_args = None

                        elif key == 'text_del':
                            for text in result["text_del"]:
                                if text in self.text_outputs:
                                    del self.text_outputs[text]

                        elif 'output__image' in key:
                            self.image_outputs[key] = result[key]

                        else:
                            outputs[tool_name] = result
                            self.text_outputs[key] = result[key]

                    for special_key in ['delete', 'empty', 'text_del']:
                        self.text_outputs.pop(special_key, None)

                    # update dependencies for successors
                    for succ in self.graph.successors(tool_name):
                        with lock:
                            in_degree[succ] -= 1
                            if in_degree[succ] == 0 and succ != 'START' and succ not in completed_tools and succ not in running_tools and succ not in ready_tools:
                                ready_tools.append(succ)

                time.sleep(0.1)  # prevent busy waiting
        
        def remove_keys(obj, keys_to_remove):
            if isinstance(obj, dict):
                return {
                    k: remove_keys(v, keys_to_remove)
                    for k, v in obj.items()
                    if k not in keys_to_remove
                }
            elif isinstance(obj, list):
                return [remove_keys(item, keys_to_remove) for item in obj]
            else:
                return obj
        outputs = remove_keys(outputs, ['delete', 'empty', 'text_del'])

        return self.text_outputs, self.image_outputs, self.tool_execution_logs, outputs
    
if __name__ == "__main__":
    import json
    with open("ms_data/data_5/data_5.json", "r") as f:
        data = json.load(f)
        
    door_id = [door['id'] for door in data['doors']]
    obj_id = [obj['id'] for obj in data['objects']]
    room_id = [room['id'] for room in data['rooms']]
    wall_id = [wall['id'] for wall in data['walls']]
    window_id = [window['id'] for window in data['windows']]
    
    total_id = door_id + obj_id + room_id + wall_id + window_id
    tool_runner = ToolRunner(data['tool_sequence'], None)
    
    tool_runner.total_id = total_id