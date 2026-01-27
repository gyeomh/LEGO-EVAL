import json
import os
import pandas as pd
import asyncio
from PIL import Image, ImageEnhance, ImageOps
from dotenv import load_dotenv
import base64
from io import BytesIO
import shutil
import re
import os
import ast
from func_utils import cal_coords, topdown_scene, topdown_room, multiview_scene, sideview_scene, sceneobject_image, objs_relation_image, get_close_image, get_rendered_object, scenewd_image, draw_name, pil_image_to_base64_str, draw_room, get_front_image
from models import LLM, VLLM
import yaml


def pil_image_to_base64_str(img):
    # img = img.resize((512, 512), Image.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_room_list(scene: dict, sys_args, my_controller = None) -> dict:
    """
    Returns a list of room IDs from the given scene.
    """
    output = {}
    output["room_list"] = [room["id"] for room in scene.get("rooms", [])]

    return output


def get_room_info(scene: dict, room_list: list, sys_args) -> dict:
    """
    Returns detailed information about specific rooms in the given scene.
    """
    room_info = {}
    for room in scene.get("rooms", []):
        if room["id"] in room_list:
            name1 = room["id"] + "_coordinates"
            name2 = room["id"] + "_floor_material"
            room_info[name1] = room.get("vertices", [])
            room_info[name2] = room.get("floorMaterial", {}).get("name", "Unknown")
    return room_info

def get_window_list(scene: dict, room_list, sys_args) -> dict:
    """
    Returns a list of window IDs from the given scene.
    """
    output = {}
    for room in room_list:
        output[room] = []
    for window in scene.get("windows", []):
        if window["roomId"] in room_list:
            if 'exterior' not in window["roomId"]:
                output[window["roomId"]].append(window["id"])

    return output

def get_window_info(scene: dict, window_list: list, sys_args) -> dict:
    """
    Returns detailed information about specific windows in the given scene.
    """
    window_info = {}
    for window in scene.get("windows", []):
        if window["id"] in window_list:

            walls = [w for w in scene.get("walls", []) if w["id"] == window["wall0"]]
            wall = walls[0]

            pos = window.get("assetPosition", {})
            pos["x"] = (wall['polygon'][0]['x'] + wall['polygon'][3]['x']) / 2
            pos['z'] = (wall['polygon'][0]['z'] + wall['polygon'][3]['z']) / 2
            
            for axis in ["x", "y", "z"]:
                if isinstance(pos[axis], float):
                    decimal_part = str(pos[axis]).split(".")[1]
                    if len(decimal_part) > 2:
                        pos[axis] = round(pos[axis], 2)


            name0 = window["id"] + "_id"
            name1 = window["id"] + "_assetId"
            name2 = window["id"] + "_rooms"
            name3 = window["id"] + "_walls"
            name5 = window["id"] + "_position"
            window_info[name0] = window["id"]
            window_info[name1] = window.get("assetId", "None")
            window_info[name2] = (window.get("room0", "None"), window.get("room1", "None"))
            window_info[name3] = (window.get("wall0", "None"), window.get("wall1", "None"))
            window_info[name5] = pos
    return window_info

def get_door_list(scene: dict, room_list: list, sys_args) -> dict:
    """
    Returns a list of door IDs from the given scene.
    """
    output = {}
    for room in room_list:
        output[room] = []
    for door in scene.get("doors", []):
        if (door["room0"] in room_list):
            output[door["room0"]].append(door["id"])
        if door["room1"] in room_list:
            output[door["room1"]].append(door["id"])

    return output

def get_door_info(scene: dict, door_list: list, sys_args) -> dict:
    """
    Returns detailed information about specific doors in the given scene.
    """
    door_info = {}
    for door in scene.get("doors", []):
        if door["id"] in door_list:

            walls = [w for w in scene.get("walls", []) if w["id"] == door["wall0"]]
            wall = walls[0]

            pos = door.get("assetPosition", {})
            pos["x"] = (wall['polygon'][0]['x'] + wall['polygon'][3]['x']) / 2
            pos['z'] = (wall['polygon'][0]['z'] + wall['polygon'][3]['z']) / 2

            for axis in ["x", "y", "z"]:
                if isinstance(pos[axis], float):
                    decimal_part = str(pos[axis]).split(".")[1]
                    if len(decimal_part) > 2:
                        pos[axis] = round(pos[axis], 2)

            name0 = door["id"] + "_id"
            name1 = door["id"] + "_assetId"
            name2 = door["id"] + "_rooms"
            name3 = door["id"] + "_walls"
            name5 = door["id"] + "_position"
            name6 = door["id"] + "_openess"
            door_info[name0] = door["id"]
            door_info[name1] = door.get("assetId", "None")
            door_info[name2] = (door.get("room0", "None"), door.get("room1", "None"))
            door_info[name3] = (door.get("wall0", "None"), door.get("wall1", "None"))
            door_info[name5] = pos
            door_info[name6] = door.get("openness", "None")
    return door_info

def get_wall_list(scene: dict, room_list: list, sys_args) -> dict:
    """
    Returns a list of wall IDs from the given scene.
    """
    output = {}
    for room in room_list:
        output[room+'_wall_list'] = []
    for wall in scene.get("walls", []):
        if wall["roomId"] in room_list:
            if 'exterior' not in wall['id']:
                output[wall["roomId"]+'_wall_list'].append(wall["id"])

    return output

def get_wall_info(scene: dict, wall_list: dict, sys_args) -> dict:
    """
    Returns detailed information about specific walls in the given scene.
    """
    wall_info = {}
    for wall in scene.get("walls", []):  
        if wall["id"] in wall_list:
            name0 = wall["id"] + "_id"
            name1 = wall["id"] + "_height"
            name2 = wall["id"] + "_roomId"
            name3 = wall["id"] + "_coordinates"
            name4 = wall["id"] + "_width"
            name5 = wall["id"] + "_material"
            name6 = wall["id"] + "_direction"
            wall_info[name0] = wall["id"]
            wall_info[name2] = wall.get("roomId", "None")
            wall_info[name3] = wall.get("segment", [])
            wall_info[name5] = wall.get("material", {}).get("name", "None")
            wall_info[name4] = wall.get("width", "None")
            wall_info[name1] = wall.get("height", "None")
            wall_info[name6] = wall.get("direction", "None")
    return wall_info

def get_object_list(scene: dict, roomId: list, sys_args) -> dict:
    """
    Returns a list of object IDs for specified rooms in the given scene.
    """
    object_list = {}
    for room in roomId:
        object_list[room] = []
    for obj in scene.get("objects", []):
        if obj["roomId"] in roomId:
            object_list[obj["roomId"]].append(obj["id"])
    return object_list

def get_object_info(scene: dict, object_list: list, sys_args) -> dict:
    """
    Returns detailed information about specific objects in the given scene.
    """
    info = {}
    for obj in scene.get('objects', []):
        obj_id = obj.get('id')
        if obj_id not in object_list:
            continue

        data = {
            "id": obj_id,
            "assetId": obj.get("assetId"),
            "roomId": obj.get("roomId"),
            "position": {},
            "rotation": obj.get("rotation", {}),
            "coordinates": []
        }

        pos = obj.get("position", {})
        for axis in ("x", "y", "z"):
            val = pos.get(axis)
            if isinstance(val, float):
                data["position"][axis] = round(val, 2)
            else:
                data["position"][axis] = val

        coords = cal_coords(obj.get("assetId"), pos, data["rotation"], sys_args)
        for vertex in coords:
            rounded = []
            for coord_val in vertex:
                if isinstance(coord_val, float):
                    rounded.append(round(coord_val, 2))
                else:
                    rounded.append(coord_val)
            data["coordinates"].append(rounded)

        info[obj_id] = data

    return info

def get_topdown_scene(scene: dict, sys_args, my_controller=None) -> dict:
    """
    Returns a top-down view of the entire scene.
    """
    output = {}

    # image = topdown_scene(scene, my_controller)
    # # image = draw_room(scene, image)

    # output["scene*/*topdown_scene*/*image"] = image
    index = scene["index"]
    # path = f'../data/data_{index}/scene_images/scene*|*topdown_scene*|*image.png'
    path = os.path.join(sys_args.data_path, f'data_{index}/scene_images/scene*|*topdown_scene*|*image.png')
    image = Image.open(path)
    image = draw_room(scene, image, sys_args)

    output["scene*/*topdown_scene*/*output__image"] = image

    return output

def get_topdown_room(scene: dict, room_list: list, sys_args, my_controller=None) -> dict:
    """
    Returns a top-down view of a specific room in the scene.
    """
    output = {}
    for roomId in room_list:
        name = f"{roomId}*/*topdown_room*/*output__image"
        index = scene["index"]
        # path = f"../data/data_{index}/scene_images/{roomId}*|*topdown_room*|*image.png"
        path = os.path.join(sys_args.data_path, f"data_{index}/scene_images/{roomId}*|*topdown_room*|*image.png")
        image = Image.open(path)
        image = pil_image_to_base64_str(image)
        # image = topdown_room(scene, roomId)
        output[name] = image

    return output

def get_multiview_scene(scene: dict, sys_args, my_controller) -> dict:
    """
    Returns multiple views of the entire scene from different angles.
    """
    # images = {}
    # for degree in [0, 90, 180, 270]:
    #     name = f"multiview_scene({degree} degree)_image"
    #     images[name] = await multiview_scene(scene, degree)
    # images["multiview_scene(topdown)_image"] = await topdown_scene(scene)
    
    images = multiview_scene(scene, my_controller, sys_args)

    return images

def get_material_image(scene:dict, materials: list, sys_args) -> dict:
    """
    Returns images for the specified materials.
    """
    material_images = {}
    for material in materials:
        image_path = f"{sys_args.material_dir}/{material}.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.convert("RGBA")
            name = f"{material}*/*material*/*output__image"
            material_images[name] = pil_image_to_base64_str(image)

    return material_images

def get_wall_scene(scene: dict, wall_list: list, sys_args, my_controller=None) -> dict:
    """
    Returns a wall scene for a specific room and direction in the scene.
    """
    images = {}
    rooms = {}
    for room in scene.get("rooms", []):
        rooms[room['id']] = []
    for wall in scene.get("walls", []):  
        if wall["id"] in wall_list:
            roomId = wall["roomId"]
            direction = wall["direction"]
            rooms[roomId].append([wall["id"], direction])
            id = wall["id"]
            name = f'{id}*/*wall in scene*/*output__image'
            index = scene['index']

            # path = f'../data/data_{index}/scene_images/{id}*|*wall in scene*|*image.png'
            path = os.path.join(sys_args.data_path, f'data_{index}/scene_images/{id}*|*wall in scene*|*image.png')
            image = Image.open(path)
            images[name] = pil_image_to_base64_str(image)

    # images = sideview_scene(scene, rooms)

    return images

def get_multiview_rendered_object(scene: dict, objects: list, sys_args, my_controller=None):
    """
    Returns rendered images for the specified objects.
    """
    objcts = []
    for object in scene['objects']:
        if object['id'] in objects:
            objcts.append(object)

    for window in scene['windows']:
        if window['id'] in objects:
            objcts.append(window)
    
    for door in scene['doors']:
        if door['id'] in objects:
            objcts.append(door)

    object_images = {}
    for obj in objcts:
        real_name = obj["id"]
        asset = obj["assetId"]
        for direction in ['front', 'right', 'left', 'top', 'back']:
            # path = f'../../object_images/{asset}_{direction}.png'
            path = os.path.join(sys_args.obj_img_dir, f'{asset}_{direction}.png')
            image = Image.open(path)

            img_name = f"{real_name}*/*{direction}_rendered*/*output__image"
            if direction == 'front':
                image = draw_name(image, real_name, sys_args)
            object_images[img_name] = pil_image_to_base64_str(image)

    # object_images = get_rendered_object(objcts)

    return object_images

def get_multiview_scene_object(scene: dict, objects: list, sys_args, my_controller=None):
    """
    Returns multiple views of the specified objects from different angles.
    """
    
    objs = []
    for obj in scene.get("objects", []):
        if obj["id"] in objects:
            objs.append(obj)

    images = sceneobject_image(scene, objs, my_controller, sys_args)

    wds = []
    for obj in scene.get("windows", []):
        if obj["id"] in objects:
            wds.append(obj)
    for obj in scene.get("doors", []):
        if obj["id"] in objects:
            wds.append(obj)
    if len(wds) >= 1:
        new_images = scenewd_image(scene, wds, my_controller)
        images.update(new_images)

    return images

def get_spatial_relation(scene: dict, object_pairs: list, sys_args, my_controller=None):

    """
    Returns image of the scene where only object pair exist in the scene.
    """
    relations = {}
    for pair in object_pairs:
        rel_img = objs_relation_image(scene, pair, sys_args, my_controller)
        first_name = "_".join(list(pair))
        name = f"{first_name}*/*spatial_relation*/*output__image"
        relations[name] = rel_img

    return relations

def get_topdown_object(scene: dict, objects: list, sys_args, my_controller=None):
    """
    Returns object-centric images for the specified objects in the scene.
    """
    positions = []
    for obj in scene.get("objects", []):
        if obj["id"] in objects:
            positions.append([obj, 'obj'])
    for window in scene.get("windows", []):
        if window['id'] in objects:
            positions.append([window, 'wd'])
    for door in scene.get("doors", []):
        if door['id'] in objects:
            positions.append([window, 'wd'])
    images = get_close_image(scene, positions, sys_args, my_controller)
    return images

def get_frontview_object(scene: dict, objects: list, sys_args, my_controller=None):
    """
    Returns object-centric images for the specified objects in the scene.
    """
    positions = []
    for obj in scene.get("objects", []):
        if obj["id"] in objects:
            positions.append([obj, 'obj'])
    for window in scene.get("windows", []):
        if window['id'] in objects:
            positions.append([window, 'wd'])
    for door in scene.get("doors", []):
        if door['id'] in objects:
            positions.append([window, 'wd'])
    images = get_front_image(scene, positions, sys_args, my_controller)
    return images


def get_property_verification(scene, current_constraint, things:list, images: dict, type, tool_log, sys_args):
    """
    Given a scene, an instruction, and related images, this function uses an LLM and a VLM to verify object properties and returns a summarized visual description based on the instruction.
    """

    llm = LLM(sys_args)
    vlm = VLLM(sys_args)
    
    # tool_prompts = yaml.safe_load(open(os.path.join(BASE_PATH, "prompts/ToolUsage_prompts.yaml"), encoding="utf-8"))
    tool_prompts = yaml.safe_load(open(os.path.join(sys_args.base_path, "prompts/ToolUsage_prompts.yaml"), encoding="utf-8"))

    llm_system_prompt = tool_prompts["get_property_verification_llm_system"]
    llm_human_prompt = tool_prompts["get_property_verification_llm_human"]
    llm_human_prompt = llm_human_prompt.replace("$INSTRUCTION$", current_constraint)

    llm_output = llm.generate(llm_system_prompt, llm_human_prompt)
    instruction = llm_output.split("Features:")[1].strip()

    with open(os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json'), 'r') as f:
        annotations = json.load(f)
    
    vlm_system_prompt = tool_prompts["get_property_verification_vlm_system"]
    vlm_human_prompt = tool_prompts["get_property_verification_vlm_human"]

    outputs = {'delete':[]}
    
    sets = {}
    
    for name, image in images.items():
        arg, func, _ = name.split('*/*')
        if ('rendered' in func) or ('view in scene' in func) or ('material' in func):
            if arg in things:
                if arg not in sets:
                    sets[arg] = {name:image}
                else:
                    sets[arg][name] = image
                if 'front' not in name:
                    outputs['delete'].append(name)

    assets = {}


    for arg, images in sets.items():
        image_names = []
        image_list = {}

        asset = None

        obj_description = None 

        for category in ["objects", "windows", "doors"]:
            for obj in scene.get(category, []):
                if obj.get("id") == arg:
                    asset = obj.get("assetId")
                    annotation = annotations.get(asset, {})
                    if annotation.get("description"):
                        obj_description = annotation["description"]
                    elif annotation.get("description_auto"):
                        obj_description = annotation["description_auto"]
                    break
            if obj_description: 
                break

        if asset is None:
            obj_description = annotations[arg]["description"]
            asset = arg
            
        if asset not in assets:
            for name, image in images.items():
                image_names.append(name)
                image_list[name] = image
            
            vlm_human_prompt = vlm_human_prompt.replace('$INSTRUCTION$', instruction)
            vlm_human_prompt = vlm_human_prompt.replace('$OBJECT_DESCRIPTION$', str(obj_description))

            output = vlm.generate(vlm_system_prompt, vlm_human_prompt, base64_images=image_list)
            
            _, description, reasoning = split_reasoning_output(output)

            assets[asset] = {}
            assets[asset]['description'] = description
            assets[asset]['reasoning'] = reasoning

        else:
            description = assets[asset]['description']
            reasoning = assets[asset]['reasoning']

        outputs[f'summarized visual information of {arg}'] = description

        tool_log["input"] = image_list
        # tool_log["system_prompt"] = vlm_human_prompt
        # tool_log["human_prompt"] = vlm_human_prompt
        tool_log["system_prompt"] = "LLM \n" + llm_system_prompt + "\n\n" + "VLM \n" + vlm_system_prompt
        tool_log["human_prompt"] = "LLM \n" + llm_human_prompt + "\n\n" + "VLM \n" + vlm_human_prompt
        tool_log["reasoning"] = reasoning
        tool_log["output"] = "LLM \n" + instruction + "\n\n" + "VLM \n" + description

    outputs["empty"] = type
    
    return outputs

def get_property_description(scene, current_constraint, things:list, images: dict, type, tool_log, sys_args):
    """
    Returns a description indicating whether the objects or materials satisfy the required property.
    """
    model = VLLM(sys_args)
    with open(os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json'), 'r') as f:
        annotations = json.load(f)
    
    # tool_vllm_prompt = yaml.safe_load(open(os.path.join(BASE_PATH, "prompts/ToolUsage_prompts.yaml"), encoding="utf-8"))
    tool_vllm_prompt = yaml.safe_load(open(os.path.join(sys_args.base_path, "prompts/ToolUsage_prompts.yaml"), encoding="utf-8"))
    system_prompt = tool_vllm_prompt['get_property_description_system']#[tool_vllm_prompt['type'] == 'get_property_verification_system']['prompt'].iloc[0]
    
    human_prompt = tool_vllm_prompt['get_property_description_human']#[tool_vllm_prompt['type'] == 'get_property_verification_human']['prompt'].iloc[0]

    outputs = {'delete':[]}
    # outputs = {}
    
    sets = {}

    
    for name, image in images.items(): #desk-0*/*render_front*/*image, desk-0*/*render_left*/*image
        arg, func, _ = name.split('*/*') #name is in the format as arg*/*func*/*image
        if ('rendered' in func) or ('view in scene' in func) or ('material' in func):
            if arg in things:
                if arg not in sets:
                    sets[arg] = {name:image}
                else:
                    sets[arg][name] = image

                if 'front' not in name:
                    outputs['delete'].append(name)

    assets = {}
    
    for arg, images in sets.items():
        
        image_names = []
        image_list = {}
        asset = None
        
        obj_description = None 

        for category in ["objects", "windows", "doors"]:
            for obj in scene.get(category, []):
                if obj.get("id") == arg:
                    asset = obj.get("assetId")
                    annotation = annotations.get(asset, {})
                    if annotation.get("description"):
                        obj_description = annotation["description"]
                    elif annotation.get("description_auto"):
                        obj_description = annotation["description_auto"]
                    break
            if obj_description: 
                break

        #material인 경우
        if asset is None:
            obj_description = annotations[arg]["description"]
            asset = arg

        if asset not in assets:
            for name, image in images.items():
                image_names.append(name)
                image_list[name] = image
            
            current_human_prompt = human_prompt.replace('$OBJECT_DESCRIPTION$', str(obj_description))
            # print("obj_description : ", obj_description)
            # print()
            # output = model.generate(system_prompt, human_prompt, base64_images=image_list)
            
            # _, description, reasoning = split_reasoning_output(output)
            max_retries = 10  # 원하는 최대 재시도 횟수

            for attempt in range(max_retries):
                output = model.generate(system_prompt, current_human_prompt, base64_images=image_list, my_temp=attempt*0.05)
                
                try:
                    _, description, reasoning = split_reasoning_output(output)
                    break  
                except Exception as e:
                    print(f"[Warning] split_reasoning_output Failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise RuntimeError("split_reasoning_output failed due to invalid output format")

            else:
                raise RuntimeError("split_reasoning_output failed due to invalid output format")

            assets[asset] = {}
            assets[asset]['description'] = description
            assets[asset]['reasoning'] = reasoning

        else:
            description = assets[asset]['description']
            reasoning = assets[asset]['reasoning']

        outputs[f'summarized visual information of {arg}'] = description

        tool_log["input"] = image_list
        tool_log["system_prompt"] = system_prompt
        tool_log["human_prompt"] = current_human_prompt
        tool_log["reasoning"] = reasoning
        tool_log["output"] = description

    outputs["empty"] = type
    
    return outputs

def get_object_match(scene, current_constraint, things, images, type, tool_log, sys_args):
    """
    Returns a description that explains what type of object each ID refers to.
    """
    model = VLLM(sys_args)
    llm_model = LLM(sys_args)
    with open(os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json'), 'r') as f:
        annotations = json.load(f)
    # tool_vllm_prompt = yaml.safe_load(open(os.path.join(BASE_PATH, "prompts/ToolUsage_prompts.yaml"), encoding="utf-8"))
    tool_vllm_prompt = yaml.safe_load(open(os.path.join(sys_args.base_path, "prompts/ToolUsage_prompts.yaml")), encoding="utf-8")

    llm_system_prompt = tool_vllm_prompt['get_category_list_system']
    llm_human_prompt = tool_vllm_prompt['get_category_list_human']
    llm_human_prompt = llm_human_prompt.replace('$CONSTRAINT$', str(current_constraint))

    system_prompt = tool_vllm_prompt['get_object_match_system']#[tool_vllm_prompt['type'] == f'get_object_match_system']['prompt'].iloc[0]
    human_prompt = tool_vllm_prompt['get_object_match_human']#[tool_vllm_prompt['type'] == f'get_object_match_human']['prompt'].iloc[0]

    outputs = {'delete':[]}
    # outputs = {}
    sets = {}
    for name, image in images.items():
        arg, func, _ = name.split('*/*') #name is in the format as arg*/*func*/*image
        if ('rendered' in func) or ('view in scene' in func):
            if arg in things:
                if arg not in sets:
                    sets[arg] = {name:image}
                else:
                    sets[arg][name] = image
                if 'front' not in name:
                    outputs['delete'].append(name)
        
    for arg, images in sets.items():
        image_names = []
        image_list = {}

        obj_description = None 

        for category in ["objects", "windows", "doors"]:
            for obj in scene.get(category, []):
                if obj.get("id") == arg:
                    asset = obj.get("assetId")
                    annotation = annotations.get(asset, {})
                    if annotation.get("description"):
                        obj_description = annotation["description"]
                    elif annotation.get("description_auto"):
                        obj_description = annotation["description_auto"]
                    break
            if obj_description: 
                break

        for name, image in images.items():
            image_names.append(name)
            image_list[name] = image
        
        # llm_output = llm_model.generate(llm_system_prompt, llm_human_prompt)
        # categories = split_cat(llm_output)
        max_retries = 10

        for attempt in range(max_retries):
            llm_output = llm_model.generate(llm_system_prompt, llm_human_prompt, my_temp=0.05*attempt)

            try:
                categories = split_cat(llm_output)
                break 
            except Exception as e:
                print(f"[Warning] split_cat Failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("split_cat Failed")
        else:
            raise RuntimeError("Output for get_object_match is not correct.")
        
        # categories.append('Object')
        
        human_prompt = human_prompt.replace('$OBJECT_DESCRIPTION$', str(obj_description)).replace('$CATEGORIES$', str(categories))

        # output = model.generate(system_prompt, human_prompt, base64_images=image_list)

        # obj_type, _, reasoning = split_reasoning_output(output)
        max_retries = 10  # 최대 재시도 횟수

        for attempt in range(max_retries):
            output = model.generate(system_prompt, human_prompt, base64_images=image_list, my_temp=0.05*attempt)

            try:
                obj_type, _, reasoning = split_reasoning_output(output)
                break  # 성공하면 루프 탈출
            except Exception as e:
                print(f"[Warning] split_reasoning_output 실패 (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("split_reasoning_output did not succeed due to different output shape.")

        else:
            raise RuntimeError("split_reasoning_output did not succeed due to different output shape.")

        outputs[f'Information of {arg}'] = {'object type': obj_type}

        tool_log["input"] = image_list
        tool_log["system_prompt"] = system_prompt
        tool_log["human_prompt"] = human_prompt
        tool_log["reasoning"] = reasoning
        tool_log["output"] = f"Object Type: {obj_type}"#\n{description}

    return outputs

def split_cat(text):
    match = re.search(r"Categories\s*:\s*(\[[^\]]*\])", text)
    if match:
        list_str = match.group(1)
        categories = ast.literal_eval(list_str)  
    else:
        print("No match found")
    
    return categories

def split_reasoning_output(text):
    """
    Parses LLM output into: object_type (or None), description/info_summary (or None), reasoning (required)
    """
    text = re.sub(r'\s*:\s*', ': ', text.strip())

    reasoning_pattern = r"(?i)\*{0,2}Chain[- ]?of[- ]?Thought\*{0,2}\s*:\s*(.*?)(?=\s*(\*{0,2}Object Type\*{0,2}\s*:|\*{0,2}Information Summary\*{0,2}\s*:|\*{0,2}Description\*{0,2}\s*:|$))"
    obj_pattern = r"(?i)\*{0,2}Object Type\*{0,2}\s*:\s*(.*?)(?=\s*(\*{0,2}Information Summary\*{0,2}\s*:|\*{0,2}Description\*{0,2}\s*:|$))"
    output_pattern = r"(?i)\*{0,2}(Information Summary|Description)\*{0,2}\s*:\s*(.*)"

    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    obj_match = re.search(obj_pattern, text, re.DOTALL)
    output_match = re.search(output_pattern, text, re.DOTALL)

    if not reasoning_match:
        raise ValueError(
            f"Failed to find 'Chain-of-Thought' section.\nReceived text:\n{text}"
        )

    reasoning = reasoning_match.group(1).strip()

    # Parse Object Type
    obj_type = None
    if obj_match:
        obj_raw = obj_match.group(1).strip()
        try:
            parsed = ast.literal_eval(obj_raw)
            obj_type = parsed if isinstance(parsed, (list, str)) else obj_raw
        except (ValueError, SyntaxError):
            obj_type = obj_raw

    # Parse Description or Information Summary
    info_summ = None
    if output_match:
        output_raw = output_match.group(2).strip()
        try:
            parsed = ast.literal_eval(output_raw)
            info_summ = parsed if isinstance(parsed, (list, dict)) else output_raw
        except (ValueError, SyntaxError):
            info_summ = output_raw

    return obj_type, info_summ, reasoning


def main():
    load_dotenv()
    scene = "../data/data_2/data_2.json"
    # scene = "data/raw_scenes/data_20/data_20.json"
    with open(scene, "r") as f:
        scene = json.load(f)

    #data 9, data 12

    # print(get_object_info(scene, ['bed-0 (bedroom 1)', 'dresser-0 (bedroom 1)', 'bookshelf-0 (bedroom 1)', 'bedside_table-0 (bedroom 1)', 'bedside_table-1 (bedroom 1)', 'reading_nook-0 (bedroom 1)', 'reading_chair-0 (bedroom 1)', 'laundry_basket-0 (bedroom 1)', 'floating_shelf-0 (bedroom 1)', 'floating_shelf-1 (bedroom 1)', 'painting-0 (bedroom 1)', 'painting-1 (bedroom 1)', 'wall_mirror-0 (bedroom 1)', 'bedspread-0|laundry_basket-0 (bedroom 1)', 'pillow-2|bookshelf-0 (bedroom 1)', 'pillow-3|reading_nook-0 (bedroom 1)', 'magazine-3|reading_nook-0 (bedroom 1)', 'bedspread-0|bedside_table-1 (bedroom 1)', 'pillow-3|dresser-0 (bedroom 1)', 'pillow-1|reading_chair-0 (bedroom 1)', 'pillow-0|reading_chair-0 (bedroom 1)', 'small cushion-0|reading_chair-0 (bedroom 1)', 'bedspread-0|reading_chair-0 (bedroom 1)', 'pillow-1|bedside_table-0 (bedroom 1)', 'bedspread-0|bedside_table-0 (bedroom 1)', 'bedspread-0|wall_mirror-0 (bedroom 1)', 'pillow-2|wall_mirror-0 (bedroom 1)', 'throw blanket-0|wall_mirror-0 (bedroom 1)', 'pillow-2|floating_shelf-1 (bedroom 1)', 'pillow-1|floating_shelf-1 (bedroom 1)', 'pillow-1|floating_shelf-0 (bedroom 1)', 'pillow-0|floating_shelf-0 (bedroom 1)']))
    # image = get_spatial_relation(scene, [("window|wall|meeting room|north|1|0|0", 'wall|meeting room|west|0', "door|0|exterior|meeting room", "storage_cabinet-0 (meeting room)")]) #"cabinet-0 (main room)"
    # image = await get_multiview_scene_object(scene, ["side_table-0 (smaller living room)", "sectional_sofa-0 (larger living room)", "media_console-0 (larger living room)", "armchair-0 (larger living room)",
                                                    #  "coffee_table-0 (larger living room)", "floor_lamp-1 (larger living room)", "plant_stand-0 (larger living room)", "loveseat-0 (smaller living room)"])
    # image = await get_multiview_scene_object(scene, [obj['id'] for obj in scene.get("windows", [])])
    # image = get_topdown_scene(scene)
    # image_path = '0.png'
    # with open(image_path, "rb") as image_file:
    #     base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    # text = get_property_verification("The floor of the bedroom is blue.", ["window|wall|living room|east|3|0|0"], {'window|wall|living room|east|3|0|0*/*top_rendered*/*image':base64_str}, 'all')
    # print('TEXT:', text)
    # image = get_topdown_scene(scene)
    # image = get_multiview_scene(scene)
    # image = get_topdown_room(scene, ['study room'])
    # image = get_topdown_object(scene, ['coffee_table-0 (living room)'])
    # print(get_door_info(scene, ['door|1|music room|library']))
    # image = get_multiview_rendered_object(scene, ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'])
    # text = get_property_verification(scene, 'abced', ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'], image, None, {})
    # text = get_object_match(scene, 'abed', ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'], image, None,{})
    
    # image = get_topdown_object(scene, ['cello-1|shelving_unit-1 (instrument storage room)'])
    # image = get_topdown_scene(scene=scene)
    # image = get_spatial_relation(scene, [("dining_table-4 (restaurant)", "door|0|exterior|restaurant"), ("wall|restaurant|north|1|exterior", "dining_table-3 (restaurant)")])
    image = get_frontview_object(scene, ['soap dispenser-0|vanity_unit-0 (bathroom)'])
    for i, (name, value) in enumerate(image.items()):
        
        # enhancer = ImageEnhance.Color(value)
        # value = enhancer.enhance(0.4)
        # i+=3
        value.save(f"{i}.png")
    # info = get_object_info(scene, ['chair-0 (waiting room)', 'chair-1 (waiting room)'])
    # print(info)

if __name__ == "__main__":
    main()
