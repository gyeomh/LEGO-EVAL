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
from adj_func_utils import *
import yaml
import time
# LLM = LLM()

def pil_image_to_base64_str(img):
    # img = img.resize((512, 512), Image.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_room_list(scene: dict) -> dict:

    output = {}
    output["room_list"] = [room["id"] for room in scene.get("rooms", [])]

    return output


def get_room_info(scene: dict, room_list: list) -> dict:

    room_info = {}
    for room in scene.get("rooms", []):
        if room["id"] in room_list:
            name1 = room["id"] + "_coordinates"
            name2 = room["id"] + "_floor_material"
            room_info[name1] = room.get("vertices", [])
            room_info[name2] = room.get("floorMaterial", {}).get("name", "Unknown")
    return room_info

def get_window_list(scene: dict, room_list) -> dict:

    output = {}
    for room in room_list:
        output[room] = []
    for window in scene.get("windows", []):
        if window["roomId"] in room_list:
            if 'exterior' not in window["roomId"]:
                output[window["roomId"]].append(window["id"])

    return output

def get_window_info(scene: dict, window_list: list) -> dict:

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
            window_info[name2] = (window.get("room1", "None"), window.get("room2", "None"))
            window_info[name3] = (window.get("wall0", "None"), window.get("wall1", "None"))
            window_info[name5] = pos
    return window_info

def get_door_list(scene: dict, room_list: list) -> dict:


    output = {}
    for room in room_list:
        output[room] = []
    for door in scene.get("doors", []):
        if (door["room0"] in room_list):
            output[door["room0"]].append(door["id"])
        if door["room1"] in room_list:
            output[door["room1"]].append(door["id"])

    return output

def get_door_info(scene: dict, door_list: list) -> dict:

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
            door_info[name2] = (door.get("room1", "None"), door.get("room2", "None"))
            door_info[name3] = (door.get("wall0", "None"), door.get("wall1", "None"))
            door_info[name5] = pos
            door_info[name6] = door.get("openess", "None")
    return door_info

def get_wall_list(scene: dict, room_list: list) -> dict:

    output = {}
    for room in room_list:
        output[room+'_wall_list'] = []
    for wall in scene.get("walls", []):
        if wall["roomId"] in room_list:
            if 'exterior' not in wall['id']:
                output[wall["roomId"]+'_wall_list'].append(wall["id"])

    return output

def get_wall_info(scene: dict, wall_list: dict) -> dict:

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

def get_object_list(scene: dict, roomId: list) -> dict:
  
    object_list = {}
    for room in roomId:
        object_list[room] = []
    for obj in scene.get("objects", []):
        if obj["roomId"] in roomId:
            object_list[obj["roomId"]].append(obj["id"])
    return object_list

def get_object_info(scene: dict, object_list: list) -> dict:
 
    object_info = {}
    for object in scene['objects']:
        if object['id'] in object_list:
            name0 = object['id'] + "_id"
            name1 = object['id'] + "_assetId"
            name2 = object['id'] + "_roomId"
            name3 = object['id'] + "_position"
            name4 = object['id'] + "_rotation"
            name5 = object['id'] + "_coordinates"
            object_info[name0] = object['id']
            object_info[name1] = object.get("assetId", "None")
            object_info[name2] = object.get("roomId", "None")

            pos = object.get("position", {})
            for axis in ["x", "y", "z"]:
                if isinstance(pos[axis], float):
                    decimal_part = str(pos[axis]).split(".")[1]
                    if len(decimal_part) > 2:
                        pos[axis] = round(pos[axis], 2)

            object_info[name3] = pos
            object_info[name4] = object.get("rotation", {})
            coords = cal_coords(object.get("assetId", None), object.get("position", {}), object.get("rotation", {}))
            for coord in [0, 1, 2, 3]:
                for i in [0, 1]:
                    if isinstance(coords[coord][i], float):
                        decimal_part = str(coords[coord][i]).split(".")[1]
                        if len(decimal_part) > 2:
                            coords[coord][i] = round(coords[coord][i], 2)
                        object_info[name5] = coords
            
    return object_info

def get_topdown_scene(scene: dict, x_num, args) -> dict:
 
    output = {}
    image, information = topdown_scene(scene, x_num, args)

    output["scene*|*topdown_scene*|*image"] = image

    return output, information

def get_topdown_room(scene, room_list,room_id, x_num, args) -> dict:
 
    output = {}

    name = f"{room_list}*|*topdown_room*|*image"
    image = topdown_room(scene, room_list, x_num, args)
    output[name] = image

    return output

def get_multiview_scene(scene: dict) -> dict:
 
    # images = {}
    # for degree in [0, 90, 180, 270]:
    #     name = f"multiview_scene({degree} degree)_image"
    #     images[name] = await multiview_scene(scene, degree)
    # images["multiview_scene(topdown)_image"] = await topdown_scene(scene)
    
    images = multiview_scene(scene)

    return images

# def get_material_image(scene:dict, materials: list) -> dict:
#     """
#     Returns images for the specified materials.

#     Input:
#         - materials (list): A list of material names (strings) for which images are required.

#     Output:
#         - material_images (dict): A dictionary where keys are formatted as "material(<material_name>)_image" and values are image arrays (RGBA) for the corresponding materials. Only includes materials found at the specified path.
#     """
#     material_images = {}
#     for material in materials:
#         image_path = f"{PATH_TO_MATERIALS}/{material}.png"
#         if os.path.exists(image_path):
#             image = Image.open(image_path)
#             image = image.convert("RGBA")
#             name = f"{material}*/*material*/*image"
#             material_images[name] = pil_image_to_base64_str(image)

#     return material_images

def get_wall_scene(scene: dict, rooms,new_scene, x_num, controller, args) -> dict:
 

    images = sideview_scene_3(scene, rooms,new_scene, x_num, controller, args)

    return images

def get_multiview_rendered_object(scene: dict, objects: list):
  
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
    object_images = get_rendered_object(objcts)

    return object_images

def get_multiview_scene_object(scene: dict, objects: list):
 
    
    objs = []
    for obj in scene.get("objects", []):
        if obj["id"] in objects:
            objs.append(obj)

    images = sceneobject_image(scene, objs)

    wds = []
    for obj in scene.get("windows", []):
        if obj["id"] in objects:
            wds.append(obj)
    for obj in scene.get("doors", []):
        if obj["id"] in objects:
            wds.append(obj)
    if len(wds) >= 1:
        new_images = scenewd_image(scene, wds)
        images.update(new_images)

    return images

def get_spatial_relation(scene: dict, object_pairs: list):

    relations = {}
    for pair in object_pairs:
        rel_img = objs_relation_image(scene, pair)
        first_name = "_".join(list(pair))
        name = f"{first_name}*/*spatial_relation*/*image"
        relations[name] = rel_img

    return relations

def get_topdown_object(scene: dict, objects: list):
 
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
    images = get_close_image(scene, positions)
    return images



def main():
    load_dotenv()
    
    scene = "data/data_0/data_0.json"
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
    image = get_wall_scene(scene)
    # image = get_multiview_scene(scene)
    # image = get_topdown_room(scene, ['study room'])
    # image = get_topdown_object(scene, ['coffee_table-0 (living room)'])
    #.image = get_multiview_rendered_object(scene, ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'])
    
    #text = get_property_verification(scene, 'abced', ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'], image, None, {})
    # text = get_object_match(scene, 'abed', ['wall_shelf_1-0 (study room)', 'wall_shelf_2-0 (study room)'], image, None,{})
    
    for i, (name, value) in enumerate(image.items()):
        
        # enhancer = ImageEnhance.Color(value)
        # value = enhancer.enhance(0.4)
        value.save(f"test/test_{i}.png")
    #print(text)

if __name__ == "__main__":
    main()
