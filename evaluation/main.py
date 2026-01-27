import json
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
from eval import Eval
import yaml
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import argparse
import glob
import asyncio

from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
# THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"
# OBJAVERSE_ASSET_DIR = '/root/mount/.objathor-assets/2023_09_23/assets'

import multiprocessing
import instr_decompose
import constr_label 
# BASE_PATH = "/root/mount/soohyun"
# DATA_PATH = "/root/mount/soohyun/data"

# BATCH_SIZE = 1
# MAX_WORKERS = 1

def get_unique_log_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    
    idx = 1
    while True:
        new_path = f"{base_path}_{idx}"
        if not os.path.exists(new_path):
            return new_path
        idx += 1

def evaluate_sync(scene, instruction, controller, args):
    evaluator = Eval(args)  # 여기서 만들면 각 쓰레드에 독립 인스턴스 → thread-safe
    result = evaluator.evaluate(scene, instruction, controller)
    return result, evaluator.logs

def process_batch(batch, args):
    # 1) 메인 스레드에서 미리 Controller 인스턴스(MAX_WORKERS 개) 생성
    controller_queue = Queue()
    x_org_options = args.xorg_screens

    for num in range(args.max_inst_workers):
        x_org = x_org_options[num % len(x_org_options)]

        c = Controller(
            commit_id=args.thor_id,
            agentMode="default",
            makeAgentsVisible=False,
            visibilityDistance=1.5,
            x_org=x_org,
            scene='Procedural',
            width=1200,
            height=1200,
            fieldOfView=90,
            action_hook_runner=ProceduralAssetHookRunner(
                asset_directory=args.objaverse_dir,
                asset_symlink=True,
                verbose=True,
            ),
        )
        print(f"Generated One Controller - #{num} with x_org={x_org}")
        controller_queue.put(c)

    # 2) worker 함수: 스레드 안에서 컨트롤러를 꺼내 쓰고, 다시 리턴
    def worker(scene):
        controller = controller_queue.get()
        try:
            return evaluate_sync(scene, scene['query'], controller, args)
        finally:
            # 작업이 끝나면 동일한 컨트롤러를 다시 큐에 넣어준다
            controller_queue.put(controller)

    # 3) ThreadPoolExecutor 로 병렬 처리
    with ThreadPoolExecutor(max_workers=args.max_inst_workers) as executor:
        results = list(executor.map(worker, batch))

    # 4) 모든 작업이 끝났으면, 남아 있는 컨트롤러들을 순차적으로 종료
    while not controller_queue.empty():
        c = controller_queue.get()
        try:
            c.stop()
        except:
            pass

    return results

def main_threaded(scenes, args):
    LOGS_PATH = get_unique_log_path(args.save_dir)
    os.makedirs(LOGS_PATH)

    print(args.data_path)
    
    holistic_correct = 0
    partial_correct = 0
    holistic_total = 0
    partial_total = 0
    
    fl = [0, 0]  # total_num, correct_num
    ms = [0, 0]
    os_temp = [0, 0]
    op = [0, 0]
    
    total = len(scenes)
    
    for i in range(0, total, args.batch_size):
        batch = scenes[i:i+args.batch_size]
        print(f"\nProcessing batch {i//args.batch_size+1} ({i}~{min(i+args.batch_size, total)-1})")
        start_time = time.time()
        
        batch_results = process_batch(batch, args)
        
        for idx_in_batch, (scene, (result, logs)) in enumerate(zip(batch, batch_results)):
            idx = i + idx_in_batch
            instruction = scene['query']
            
            print("------------------------")
            print(f"Instruction {idx}: ", instruction)
            
            num_constraint, num_correct, out_fl, out_ms, out_os, out_op = result
            scene_data_name = scene['index']
            log_filename = os.path.join(LOGS_PATH, f"data_{scene_data_name}_log.json")
            
            partial_correct += num_correct
            partial_total += num_constraint
            
            fl[0] += out_fl[0]
            fl[1] += out_fl[1]
            ms[0] += out_ms[0]
            ms[1] += out_ms[1]
            os_temp[0] += out_os[0]
            os_temp[1] += out_os[1]
            op[0] += out_op[0]
            op[1] += out_op[1]
            
            holistic_total += 1
            if num_constraint == num_correct:
                holistic_correct += 1
                holistic = 1
                print(f"Scene_{idx} Correctly Built!")
            else:
                holistic = 0
                print(f"Scene_{idx} Not Correctly Built!")
            
            print(f"Instruction_{idx} Execution Time: {time.time() - start_time:.2f} seconds")
            
            partial = num_correct / num_constraint
            
            print(f"----------------")
            print(f"RESULT OF INSTRUCTION:")
            print(f"Holistic Accuracy: {holistic}% ({holistic}/1)")
            print(f"Partial Accuracy: {partial}% ({num_correct}/{num_constraint})")
            print(f"Floor Layout Accuracy: {out_fl[1]}/{out_fl[0]}")
            print(f"Material Selection Accuracy: {out_ms[1]}/{out_ms[0]}")
            print(f"Object Selection Accuracy: {out_os[1]}/{out_os[0]}")
            print(f"Object Placement Accuracy: {out_op[1]}/{out_op[0]}")
            
            final_output = dict()
            final_output["Holistic_Accuracy"] = f"{holistic}% ({holistic}/1)"
            final_output["Partial_Accuracy"] = f"{partial}% ({num_correct}/{num_constraint})"
            final_output["Floor_Layout_Accuracy"] = f"{out_fl[1]}/{out_fl[0]}"
            final_output["Material_Selection_Accuracy"] = f"{out_ms[1]}/{out_ms[0]}"
            final_output["Object_Selection_Accuracy"] = f"{out_os[1]}/{out_os[0]}"
            final_output["Object_Placement_Accuracy"] = f"{out_op[1]}/{out_op[0]}"
            
            # print(f"TOKEN_USAGE: {TOKEN_USAGE}")

            logs["final_result"] = final_output
            
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"Batch {i//args.batch_size+1} done in {time.time()-start_time:.2f} seconds.")
    
    holistic_accuracy = holistic_correct / holistic_total if holistic_total else 0
    partial_accuracy = partial_correct / partial_total if partial_total else 0
    fl_accuracy = fl[1] / fl[0] if fl[0]!=0 else 0
    ms_accuracy = ms[1] / ms[0] if ms[0]!=0 else 0
    os_accuracy = os_temp[1] / os_temp[0] if os_temp[0]!=0 else 0
    op_accuracy = op[1] / op[0] if op[0]!=0 else 0
    
    print(f"----------------")
    print(f"RESULT OF TOTAL:")
    print(f"Holistic Accuracy: {holistic_accuracy}% ({holistic_correct}/{holistic_total})")
    print(f"Partial Accuracy: {partial_accuracy}% ({partial_correct}/{partial_total})")
    print(f"Floor Layout Accuracy: {fl_accuracy}% ({fl[1]}/{fl[0]})")
    print(f"Material Selection Accuracy: {ms_accuracy}% ({ms[1]}/{ms[0]})")
    print(f"Object Selection Accuracy: {os_accuracy}% ({os_temp[1]}/{os_temp[0]})")
    print(f"Object Placement Accuracy: {op_accuracy}% ({op[1]}/{op[0]})")
    
    return holistic_accuracy, partial_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--objaverse_dir", type=str, required=True)
    parser.add_argument("--material_dir", type=str, required=True)
    parser.add_argument("--obj_img_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--decompose_save_dir", type=str, default="")
    parser.add_argument("--thor_id", type=str, default="3213d486cd09bcbafce33561997355983bdf8d1a", required=True)
    parser.add_argument("--model_llm", type=str, default="", required=True)
    parser.add_argument("--model_vllm", type=str, default="", required=True)
    parser.add_argument("--model_const", type=str, default="", required=True)
    parser.add_argument("--base_url_llm", type=str, default="", required=True)
    parser.add_argument("--base_url_vllm", type=str, default="", required=True)
    parser.add_argument("--base_url_const", type=str, default="", required=True)
    parser.add_argument("--lego_bench", action="store_true", help="Enable LEGO-Bench evaluation")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_inst_workers", type=int, default=1)
    parser.add_argument("--max_tool_workers", type=int, default=1)
    parser.add_argument("--top_p_llm", type=float, default=1.0)
    parser.add_argument("--top_k_llm", type=int, default=1)
    parser.add_argument("--temperature_llm", type=float, default=0.0)
    parser.add_argument("--top_p_vllm", type=float, default=1.0)
    parser.add_argument("--top_k_vllm", type=int, default=1)
    parser.add_argument("--temperature_vllm", type=float, default=0.0)
    parser.add_argument("--api_key_llm", type=str, default="", required=True)
    parser.add_argument("--api_key_vllm", type=str, default="", required=True)
    parser.add_argument("--api_key_const", type=str, default="", required=True)
    parser.add_argument("--xorg_screens", nargs="+", default=[":0"], required=True)



    args = parser.parse_args()

    # load_dotenv()
    whole_run_start_time = time.time()
    
    scene_paths = []
    # json_files = glob.glob(os.path.join(args.data_path, "*.json"))
    # scene_paths.extend(json_files)
    data_dirs = glob.glob(os.path.join(args.data_path, "data_*"))
    scene_paths = [p for p in data_dirs if os.path.isdir(p)]
    print(f"Found {len(scene_paths)} JSON files.")
    print(scene_paths)
    
    
    scenes = []
    for path in scene_paths:
        folder_name = os.path.basename(path)  
        scene_graph_path = os.path.join(path, f"{folder_name}.json")
        with open(scene_graph_path, 'r') as f:
            scene = json.load(f)
        print(path)
            
        name = [n for n in path.split("/")][0]
        scene['data_name'] = name
        # scene["index"] = int(name.split("_")[1])
        scenes.append(scene)
    
    print(args.lego_bench)
    if args.lego_bench == True:
        with open(os.path.join(args.data_path, 'full_data.json'), 'r') as f:
            constraint_data = json.load(f)
    else:
        decompose_save_dir = asyncio.run(instr_decompose.main(scene_paths, args))
        constraint_data = asyncio.run(constr_label.main(decompose_save_dir, args))


    instruction_to_constraints = {
            entry['instruction']: entry for entry in constraint_data
        }
    
    for scene in scenes:
        scene_instruction = scene['query']
        if scene_instruction in instruction_to_constraints:
            entry = instruction_to_constraints[scene_instruction]
            info = {}
            for label in entry['labels']:
                index = label['condition_idx']
                cond_type = label['condition_type']
                info[index] = {"constraint_type": cond_type}
            # print(scene['index'])
            constraints = entry.get('constraints', [])
            constraint_list = [
                {"constraint": constraints[i], "constraint_type": info[str(i)]["constraint_type"]}
                for i in range(len(constraints))
            ]
            scene["constraints"] = constraint_list
    
    
    print("running....")

    main_threaded(scenes, args)

    elapsed = time.time() - whole_run_start_time

    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Whole Process Execution Time: {hours}h {minutes}m {seconds:.2f}s")