import argparse
from datetime import datetime
import tqdm
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import json
import asyncio
import yaml
from copy import deepcopy
import re

def extract_constraints(output_text):
    """
    Extracts numbered constraints from the given output string and returns them as a Python list.
    """
    # 정규식으로 "1. ..." 형식의 라인만 추출
    constraint_lines = re.findall(r'\d+\.\s+(.*)', output_text)
    return constraint_lines


def load_data(data_path):

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def load_sys(prompt_path):

    with open(prompt_path, 'r') as f:
        data = yaml.safe_load(f)  # YAML을 딕셔너리로 파싱
    
    return data.get('System')  # System 키만 가져오기

def load_user(prompt_path):

    with open(prompt_path, 'r') as f:
        data = yaml.safe_load(f) 
        
    return  data.get('User')


TOTAL_COST = 0

def all_process(scene_paths):
    decomposed_data = []

    for scene_path in scene_paths:
        name_file = scene_path.split("/")[-1]
        scene_path += f"/{name_file}.json"
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)

        scene_query = scene_data['query']
        decomposed_data.append({'instruction': scene_query})
    
    return decomposed_data

def prepare_model_input(sys_prompt, user_prompt, data):
    turn = dict()
    turn["instruction"] = data["instruction"]
    # turn["constraints"] = data["constraints"]
    # turn["labels"] = data["labels"]
    
    turn["system_input"] = sys_prompt
    turn["model_input"] = user_prompt.format(**{
        "instruction": data["instruction"]
    })

    return turn

def load_and_prepare_data(data, args):
    all_model_inputs = []
    # data = load_data(args.data_path)
    
    sys_prompt = load_sys(os.path.join(args.base_path, 'prompts/ConstraintIdentification_prompts.yaml'))
    user_prompt = load_user(os.path.join(args.base_path, 'prompts/ConstraintIdentification_prompts.yaml'))
    for dialog in data:  # 데이터의 각 샘플에 대해 반복
        all_model_inputs.append(prepare_model_input(sys_prompt, user_prompt, dialog))

    return all_model_inputs

# async def async_generate(llm, model_data, idx):
#     global TOTAL_COST

#     system_message = SystemMessage(content=model_data["system_input"])
#     human_message = HumanMessage(content=model_data["model_input"])
#     constraints_list = []  # Initialize constraints_list with default empty list
    
#     while True:
#         try:
#             # get_openai_callback 
#             #with openai_callback 
        
#             response = await llm.agenerate([[system_message, human_message]])
#             token_used = response.llm_output['token_usage']['total_tokens']
#             TOTAL_COST += token_used / 1000 * 0.002 # gpt-3.5-turbo
#             print(idx, TOTAL_COST)
#             response_text = response.generations[0][0].text.split(":")[-1].strip()  # Extract the text after the colon
#             constraints_list = extract_constraints(response_text)
#             # TOTAL_COST += token_used / 1000 * 0.06  # gpt-4
#             break
#         except Exception as e:
#             print(f"Exception occurred: {e}")
#             # If there's an exception, constraints_list remains as empty list
#             break
    
#     result = deepcopy(model_data)
    
#     result["constraints"] = constraints_list
    
#     result.pop('system_input', None)
#     result.pop('labels', None)
#     result.pop('model_input', None)
#     result["labels"] = model_data["labels"]

#     return result

async def async_generate(client, model_data, idx, args):
    global TOTAL_COST

    system_prompt = model_data["system_input"]
    human_prompt = model_data["model_input"]
    constraints_list = []

    while True:
        try:
            response = await client.chat.completions.create(
                model=args.model_const,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt},
                ],
                temperature=0.5,
                max_tokens=1500,
                timeout=90.0,
            )

            token_used = response.usage.total_tokens
            TOTAL_COST += token_used / 1000 * 0.002  # gpt-3.5-turbo
            print(idx, TOTAL_COST)

            response_text = response.choices[0].message.content
            response_text = response_text.split(":")[-1].strip()

            constraints_list = extract_constraints(response_text)
            break

        except Exception as e:
            print(f"Exception occurred at idx {idx}: {e}")
            break

    result = deepcopy(model_data)
    result["constraints"] = constraints_list

    result.pop("system_input", None)
    result.pop("model_input", None)
    # result.pop("labels", None)
    # result["labels"] = model_data["labels"]

    return result


# async def generate_concurrently(all_model_data, start_idx, args):
#     base_url = args.base_url_const
#     model_name = args.model_const
#     api_key = args.api_key_const
#     client = AsyncOpenAI(api_key=api_key, base_url=base_url)
#     response = client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": human_prompt}
#                 ],
#                 timeout=90.0
#             )

#     # llm = ChatOpenAI(model_name=args.model_const,  # 'gpt-3.5-turbo' or 'gpt-4o-mini'
#                     #   temperature=0.5,max_tokens=1500, max_retries=100, api_key = args.api_key_const)
#     # tasks = [async_generate(llm, model_data, i+start_idx)
#     #          for i, model_data in enumerate(all_model_data)]
#     return await tqdm_asyncio.gather(*tasks)


async def generate_concurrently(all_model_data, start_idx, args):
    client = AsyncOpenAI(
        api_key=args.api_key_const,
        base_url=args.base_url_const,
    )

    tasks = [
        async_generate(client, model_data, start_idx + i, args)
        for i, model_data in enumerate(all_model_data)
    ]

    return await tqdm_asyncio.gather(*tasks)


async def main(scene_paths, args):
    global TOTAL_COST

    data = all_process(scene_paths)

    all_model_data = load_and_prepare_data(data, args)

    if os.path.exists(args.decompose_save_dir):
        print("The save_dir already exists. Adding 0 to the end of the folder name.")
        args.decompose_save_dir = args.decompose_save_dir + "_0"
    
    os.makedirs(args.decompose_save_dir, exist_ok=True)
    
    results = []
    print(len(all_model_data), "model data loaded")
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    total_result_path = f"{args.decompose_save_dir}/const_w_mixedInst_{timestamp}.json"
    k = len(all_model_data)
    # k = 29
    n = 40 #이게 동시에 돌리는 개수인데 추후에 args에 넣을 예정
    start_idx = 0
    for i in range(start_idx, k, n):
        batch = all_model_data[i:i+n]
        results.extend(await generate_concurrently(batch, i, args))
        print("TOTAL COST: ", i, TOTAL_COST)
        with open(total_result_path, "w", encoding='UTF-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False) 
    return total_result_path
    

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))