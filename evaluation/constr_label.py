import argparse
import time
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
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--api_key_path", default='./conf.d/config.yaml')
#     parser.add_argument("--turn")
#     # parser.add_argument('--data_path', type=str, default='full_data.json') #/home/intern/hwangbo/env_gen_bench/logs/holodeck_lego
#     parser.add_argument('--data_path', type=str, default='/home/minseok/hwangbo/hwangbo/env_gen_bench/instconst/mixed_const/const_w_mixedInst_251128_080542.json') #/home/intern/hwangbo/env_gen_bench/logs/holodeck_lego
#     parser.add_argument("--save_path", type=str, default='/home/minseok/hwangbo/hwangbo/env_gen_bench/instconst/mixed_const')
#     parser.add_argument('--prompt_2', type=str, default='re_annotate.yaml')
#     args = parser.parse_args()

#     return args


def prepare_model_input(sys_prompt, user_prompt, data, cons, id, j):
    turn = dict()
    turn["id"] = id
    turn["instruction"] = data["instruction"]
    turn["constraint"] = cons
    turn["system_input"] = sys_prompt
    turn["model_input"] = user_prompt.format(**{
        "constraint": cons, "instruction" : data["instruction"]
    })
    return turn

def load_and_prepare_data(data_path, args):
    all_model_inputs = []
    data = load_data(data_path)
    sys_prompt = load_sys(os.path.join(args.base_path, "prompts/ConstraintLabel_prompts.yaml"))
    user_prompt = load_user(os.path.join(args.base_path, "prompts/ConstraintLabel_prompts.yaml"))
    for id, dialog in enumerate(data):
        for j, cons in enumerate(dialog.get("constraints", [])):
            all_model_inputs.append(prepare_model_input(sys_prompt, user_prompt, dialog, cons, id, j))
    return all_model_inputs

# async def async_generate(llm, model_data, idx):
#     global TOTAL_COST
#     system_message = SystemMessage(content=model_data["system_input"])
#     human_message = HumanMessage(content=model_data["model_input"])
#     response = None
#     while True:
#         try:
#             response_obj = await llm.agenerate([[system_message, human_message]])
#             print(idx)
#             token_used = response_obj.llm_output['token_usage']['total_tokens']
#             TOTAL_COST += token_used / 1000 * 0.002 # gpt-3.5-turbo
#             # print(idx, TOTAL_COST)
#             response = response_obj.generations[0][0].text.strip()
#             break
#         except Exception as e:
#             print(f"Exception occurred: {e}")
#             break
#     return {"id": model_data["id"], "j": idx, "label": response}

# from openai import AsyncOpenAI

async def async_generate(client, model_data, idx, args):
    global TOTAL_COST

    system_prompt = model_data["system_input"]
    human_prompt = model_data["model_input"]
    response_text = None

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

            print(idx)

            token_used = response.usage.total_tokens
            TOTAL_COST += token_used / 1000 * 0.002  # gpt-3.5-turbo

            response_text = response.choices[0].message.content.strip()
            break

        except Exception as e:
            print(f"Exception occurred at idx {idx}: {e}")
            break

    return {
        "id": model_data["id"],
        "j": idx,
        "label": response_text
    }

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


# async def generate_concurrently(all_model_data, start_idx, args):
#     llm = ChatOpenAI(model_name=args.model_const,  # 'gpt-3.5-turbo' or 'gpt-4o-mini'
#                       temperature=0.5,max_tokens=1500, max_retries=100, api_key = args.api_key_conts)
#     tasks = [async_generate(llm, model_data, i+start_idx)
#              for i, model_data in enumerate(all_model_data)]
#     return await tqdm_asyncio.gather(*tasks)

async def main(data_path, args):
    global TOTAL_COST
    # 1. 데이터 로드
    with open(data_path, 'r') as f:
        data = json.load(f)
    # 2. 분류할 constraint별 모델 입력 생성
    all_model_inputs = load_and_prepare_data(data_path, args)
    # os.makedirs(args.decompose, exist_ok=True)
    # if os.path.exists(args.save_path):
    #     print("The save_dir already exists. Please change the save_dir.")
    print(len(all_model_inputs), "model data loaded")
    total_result_path = os.path.join(args.decompose_save_dir, 'full_identified.json')
    # 3. 분류 실행
    k = len(all_model_inputs)
    n = 500
    results = []
    for i in range(0, k, n):
        batch = all_model_inputs[i:i+n]
        results.extend(await generate_concurrently(batch, i, args))
        print("TOTAL COST: ", i, TOTAL_COST)
        await asyncio.sleep(5)
    # 4. 결과를 원본 데이터에 labels_seperate로 병합
    id2labels = {}
    for r in results:
        id2labels.setdefault(r["id"], []).append(r["label"])
    for idx, dialog in enumerate(data):
        dialog["labels"] = id2labels.get(idx, [])

    for inst in data:
        new_labels = []
        for const_id, const_label in enumerate(inst['labels']):
            new_labels.append({'condition_idx': str(const_id), 'condition_type': const_label.title()})
        inst['labels'] = new_labels
    with open(total_result_path, "w", encoding='UTF-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return data

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))