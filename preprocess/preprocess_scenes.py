from adj_functions import get_topdown_scene, get_topdown_room, get_wall_scene
from adj_func_utils import *
import os
import json
import multiprocessing
import traceback
import re
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner

import argparse
from pathlib import Path

def init_images(scene, scene_rooms, room_id, x_num, out_dir, curroom_walls, args):

    room_images = get_topdown_room(scene, scene_rooms, room_id, x_num, args)
    print("    Finished Room Images")
    
    #print(scene_walls)
    r_images=  {}
    r_images.update(room_images)
    # print(f"Saved Json for Scene {room_id}")

    for name, image in r_images.items():
        filename = f"{name}.png"
        filepath = os.path.join(out_dir, filename)
        image.save(filepath)
        print(f"    Saved: {filepath}")

def init_images_2(scene,scene_rooms,room_id,  x_num, out_dir, curroom_walls, args):

    rooms = {}
    for room in scene.get("rooms", []):
        rooms[room['id']] = []
    for wall in scene.get("walls", []):  
        if wall["id"] in curroom_walls and scene["rooms"][room_id]["id"] == scene_rooms:
            roomId = wall["roomId"]
            direction = wall["direction"]
            rooms[roomId].append([wall["id"], direction])
            
    new_scene = seperate_room(scene, scene_rooms)
    
    filtered_objs = []
    offset = 0.7
    curr_room = new_scene['rooms'][0]

    max_x= curr_room['floorPolygon'][2]['x']
    max_y= curr_room['floorPolygon'][2]['z']
    min_x= curr_room['floorPolygon'][0]['x']
    min_y= curr_room['floorPolygon'][0]['z']
    
    for obj in new_scene['objects']:
        x, z = obj['position']['x'], obj['position']['z']
        is_near_edge = (
            abs(z - max_y) <= offset or
            abs(z - min_y) <= offset or
            abs(x - min_x) <= offset or
            abs(x - max_x) <= offset
        )
        if is_near_edge:
            filtered_objs.append(obj)
        
    new_scene['objects'] = filtered_objs

    controller = Controller( #for문 밖으로 제외
        commit_id=args.thor_id,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=new_scene,
        x_server=x_num,
        width=1200,
        height=1200,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=args.objaverse_dir,
            asset_symlink=True,
            verbose=True,
        )
    )    
        
    wall_images = get_wall_scene(scene, rooms, new_scene, x_num, controller, args)
    
    controller.stop()
    
    #print(scene_walls)
    r_images=  {}
    r_images.update(wall_images)

    # print(f"Saved Json for Scene {room_id}")

    for name, image in r_images.items():
        filename = f"{name}.png"
        filepath = os.path.join(out_dir, filename)
        image.save(filepath)
        print(f"    Saved: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--thor_id", type=str, default="3213d486cd09bcbafce33561997355983bdf8d1a", required=True)
    parser.add_argument("--objaverse_dir", type=str, default="", required=False)
    parser.add_argument("--lego_bench", action="store_true", help="Enable LEGO-Bench evaluation")
    parser.add_argument("--lego_bench_data_dir", type=str, required=True)
    parser.add_argument("--xorg_screens", nargs="+", default=[":0"], required=True)

    args = parser.parse_args()
    
    scene_paths = sorted(str(p.resolve()) for p in Path(args.base_path).glob("*.json"))

    if args.lego_bench:
        with open(args.lego_bench_data_dir, 'r') as f:
            lego_bench_data = json.load(f)
        full_set_bench = {}
        for inst_id, data in enumerate(lego_bench_data):
            full_set_bench[data['instruction']] = inst_id

   
    for scene_number, scene_path in enumerate(scene_paths):
        try:
            with open(scene_path, 'r') as f:
                scene = json.load(f)
            
            if args.lego_bench:
                scene_id = full_set_bench[scene['query']]
            else:
                scene_id = scene_number

            scene['index'] = scene_id

            print(f"Starting Scene #{scene_id}")

            out_dir = os.path.join(args.save_dir, f'data_{scene_id}/scene_images')
            os.makedirs(out_dir, exist_ok=True)
            print(f'    Created: {out_dir}')

            
            with open(args.save_dir+f"/data_{scene_id}/data_{scene_id}.json", 'w', encoding='UTF-8') as f:
                json.dump(scene, f, indent=4, ensure_ascii=False) 

            x_num = args.xorg_screens[0] #":3" # 0,2,3

            images = {}

            scene_rooms = [item['id'] for item in scene['rooms']]
            scene_walls = [item['id'] for item in scene['walls']]
            result = {}

            for item in scene_walls:
                parts = item.split('|')
                if len(parts) >= 3:
                    room = parts[1]
                    
                    if room not in result:
                        result[room] = []
                    
                    result[room].append(item)
                    
            scene_image, information = get_topdown_scene(scene, x_num, args)
            images.update(scene_image)
            
            if os.path.exists(f'{out_dir}/scene_metadata.json'):
                os.remove(f'{out_dir}/scene_metadata.json')
                print(f"Deleted file for Scene {scene_id}")

            with open(f'{out_dir}/scene_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(information, f, indent=2)
                
            
            for room_id, room in enumerate(scene_rooms):
                # 각 룸에 대해 새로운 프로세스를 생성하고 시작합니다.
                p = multiprocessing.Process(target=init_images, args=(scene, room, room_id, x_num, out_dir, result[room], args))
                p.start()
                p.join() 
                
            for room_id, room in enumerate(scene_rooms):
                # 각 룸에 대해 새로운 프로세스를 생성하고 시작합니다.
                p = multiprocessing.Process(target=init_images_2, args=(scene, room, room_id, x_num, out_dir, result[room], args))
                p.start()
                p.join() 
                
            print("    Finished Room Images")

            print("    Finished Wall Images")     
            
            for name, image in images.items():
                #save image
                filename = f"{name}.png"
                filepath = os.path.join(out_dir, filename)
                image.save(filepath)
                print(f"    Saved: {filepath}")
        
        except Exception as e:
            print(f"[Error] Scene #{scene_path} failed: {e}")
            traceback.print_exc()
    else:
        print(f"already processed data Scene #{scene_path}")