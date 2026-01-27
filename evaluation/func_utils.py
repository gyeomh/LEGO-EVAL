import math
import copy
import numpy as np
from tqdm import tqdm
import os
import json
import cv2
import math
import colorsys
from shapely.geometry import box
import yaml
from scipy.spatial.transform import Rotation as R

import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import asyncio
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner

# from ai2holodeck.constants import HOLODECK_BASE_DATA_DIR, THOR_COMMIT_ID
# THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"
# OBJAVERSE_ASSET_DIR = '/root/mount/.objathor-assets/2023_09_23/assets'

# DATA_PATH = "/root/mount/hwangbo/env_gen_bench/data"
# BASE_PATH = "/root/mount/soohyun/LEGO/LEGO_qwen"

def pil_image_to_base64_str(img):
    # img = img.resize((512, 512), Image.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_bbox_dims(boundingcoord):
    bbox_info = boundingcoord
    if "x" in bbox_info:
        return bbox_info
    if "size" in bbox_info:
        return bbox_info["size"]
    mins = bbox_info["min"]
    maxs = bbox_info["max"]
    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}

def cal_coords(assetId: str, position: dict, rotation: dict, sys_args) -> list:
    # Load annotations
    with open(os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json'), 'r') as file:
        anno = json.load(file)
    obj_data = anno[assetId]
    if "assetMetadata" in obj_data:
        dim = obj_data['assetMetadata']['boundingBox']
    elif "thor_metadata" in obj_data:
        dim = obj_data["thor_metadata"]["assetMetadata"]['boundingBox']
    
    point = (position["x"] * 100, position["z"] * 100)
    rotation = rotation["y"]  # Rotation in degrees
    bbx = get_bbox_dims(dim)

    # Dimension assignment
    object_dim = (bbx["x"] * 100, bbx["z"] * 100)
    obj_length, obj_width = object_dim
    obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

    # Center of the object
    center_x, center_y = point

    # Convert rotation to radians (counterclockwise)
    rad = math.radians(rotation)

    # Define unrotated corners
    if rotation == 0:
        corners = [
            (obj_half_length, -obj_half_width),   # Lower-right
            (obj_half_length, obj_half_width),    # Upper-right
            (-obj_half_length, obj_half_width),   # Upper-left
            (-obj_half_length, -obj_half_width),  # Lower-left
        ]
    else:
        corners = [
            (obj_half_length, -obj_half_width),   # Lower-right
            (-obj_half_length, -obj_half_width),  # Lower-left
            (-obj_half_length, obj_half_width),   # Upper-left
            (obj_half_length, obj_half_width),    # Upper-right
        ]

    # Rotate each corner around the center
    rotated_corners = []
    for x, y in corners:
        rot_x = x * math.cos(rad) - y * math.sin(rad)
        rot_y = x * math.sin(rad) + y * math.cos(rad)
        world_x = center_x + rot_x
        world_y = center_y + rot_y
        rotated_corners.append([world_x, world_y])

    # Close the polygon by appending the first point
    rotated_corners.append(rotated_corners[0])

    if rotation not in [0, 90, 180, 270]:
        new_x = position["x"] * 100
        rotated_corners = [[2 * new_x - x, y] for x, y in rotated_corners]

    return rotated_corners

def get_rendered_object(obcts: list, my_controller, sys_args):
    controller = my_controller
    controller.reset(renderInstanceSegmentation=True, renderDepthImage=True)
    items = {}
    for obj in obcts:
        real_name = obj["id"]
        name = obj["assetId"]


        controller.reset() 
        controller.step(
            action="RandomizeLighting",
            randomizeColor=False,
            brightness=(1.25, 1.25),
        )
        # Spawn the asset
        event = controller.step(
            action="SpawnAsset", assetId=name, generatedId="asset_0"
        )
        
        asset_iterator = (obj for obj in event.metadata["objects"] if obj["objectId"] == "asset_0")
        asset = next(asset_iterator, None)
        if asset is None:
            print(f"Asset 'asset_0' not found in metadata for {name}. Skipping.")
            continue

        bb = asset["axisAlignedBoundingBox"]["size"]
        
        controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=0, y=0, z=0),
            rotation=dict(x=0, y=0, z=0),
            orthographic=True,
            skyboxColor="white",
            orthographicSize=max(bb["x"], bb["z"], bb["y"]) * 0.75,
            renderInstanceSegmentation=True,
        )

        camera_sides = {
            "top": dict(x=90, y=180, z=0),
            "front": dict(x=0, y=180, z=0),
            "back": dict(x=0, y=0, z=0),
            "right": dict(x=0, y=270, z=0),
            "left": dict(x=0, y=90, z=0),
        }
        for side, rotation in camera_sides.items():
            controller.step(
                action="TeleportObject",
                objectId="asset_0",
                position=asset["position"],
                rotation=rotation,
                forceAction=True,
            )
            asset_iterator = (obj for obj in controller.last_event.metadata["objects"] if obj["objectId"] == "asset_0")
            asset = next(asset_iterator, None)
            if asset is None:
                print(f"Asset 'asset_0' disappeared after teleport for {name} ({side}). Skipping.")
                continue

            bb = asset["axisAlignedBoundingBox"]
            
            controller.step(
                action="UpdateThirdPartyCamera",
                position=dict(x=bb["center"]["x"], y=bb["center"]["y"], z=-20),
            )

            event = controller.step(action="Pass")
            
            if not all([event.third_party_camera_frames, 
                    event.third_party_instance_segmentation_frames, 
                    event.third_party_depth_frames]):
                print(f"Missing frames for {name} ({side}): RGB={bool(event.third_party_camera_frames)}, "
                    f"Seg={bool(event.third_party_instance_segmentation_frames)}, "
                    f"Depth={bool(event.third_party_depth_frames)}. Skipping.")
                continue

            rgb_frame = event.third_party_camera_frames[0]
            r, g, b = np.rollaxis(rgb_frame, axis=-1)
            seg_frame = event.third_party_instance_segmentation_frames[0]
            a = np.zeros_like(r)
            for key in event.object_id_to_color:
                if key.startswith("asset_0"):
                    a += (np.all(seg_frame == event.object_id_to_color[key], axis=-1).astype(np.uint8) * 255)
            depth_frame = event.third_party_depth_frames[0]
            a += (depth_frame == 0).astype(np.uint8) * 255
            mask = a > 0
            b = b.copy()
            r = r.copy()
            b[mask] = (b[mask] * 0.9).astype(np.uint8)
            r[mask] = (r[mask] * 1.1).astype(np.uint8)
            
            # Save with side-specific filenames
            image = Image.fromarray(np.stack([r, g, b, a], axis=2), "RGBA")
            # image = image.convert("RGB")
            img_name = f"{real_name}*/*{side}_rendered*/*output__image"
            if side == 'front':
                image = draw_name(image, real_name, sys_args)
            items[img_name] = pil_image_to_base64_str(image)
    return items

def draw_name(img, name, sys_args):
    # Convert the image to RGB mode
    img = img.convert("RGBA")

    # Create a new image with a transparent background
    new_img = Image.new("RGBA", img.size, (255, 255, 255, 0))

    # Draw the original image on the new image
    new_img.paste(img, (0, 0), img)

    # Create a draw object
    draw = ImageDraw.Draw(new_img)

    # Define font size and color
    font_color = (255, 0, 0)  # Red color

    # Load a default font
    # font = ImageFont.truetype("../utils/roboto.ttf", size=20)
    font = ImageFont.truetype(os.path.join(sys_args.base_path, "evaluation/utils/roboto.ttf"), size=20)

    new_name = "Image of: " + name

    # Draw the text on the image
    draw.text((20, 20), new_name, fill=font_color, font=font)

    return new_img


def seperate_room(scene: dict, room_name: str) -> dict:
    new_scene = {}

    ### rooms
    for room in scene["rooms"]:
        if room["id"] == room_name:
            curr_room = room
    diff_x, diff_y = curr_room["vertices"][0]

    for i in range(0, 4):
        curr_room["vertices"][i][0] -= diff_x
        curr_room["vertices"][i][1] -= diff_y
        curr_room["floorPolygon"][i]["x"] -= diff_x
        curr_room["floorPolygon"][i]["z"] -= diff_y
        curr_room["full_vertices"][i][0] -= diff_x
        curr_room["full_vertices"][i][1] -= diff_y
    
    new_scene["rooms"] = [curr_room]

    ### doors 
    curr_doors = []

    for door in scene["doors"]:
        if (door["room0"] == room_name) and (door["room1"] == "exterior"):
            curr_doors.append(door)
        elif (door["room1"] == room_name) and (door["room0"] == "exterior"):
            curr_doors.append(door)
        elif (door["room0"] == room_name) and (door["room1"] != "exterior"):
            door["room1"] = "exterior"
            door["wall1"] = door["wall0"]+"|exterior"
            curr_doors.append(door)
        elif (door["room1"] == room_name) and (door["room0"] != "exterior"):
            door["room0"] = "exterior"
            door["wall0"] = door["wall1"]+"|exterior"
            curr_doors.append(door)
    
    new_scene["doors"] = curr_doors

    ### metadata 
    new_scene["metadata"] = scene["metadata"]
    new_scene["metadata"]["roomSpecId"] = room_name

    ### objects
    curr_objects = []

    max_x, max_y = curr_room["vertices"][2]
    min_x, min_y = curr_room['vertices'][0]

    for obj in scene["objects"]:
        if (obj["roomId"] == room_name):
            obj["position"]["x"] -= diff_x
            obj["position"]["z"] -= diff_y

            if 0 <= obj["position"]["x"] <= max_x and 0 <= obj["position"]["z"] <= max_y:
                curr_objects.append(obj)
    new_scene["objects"] = curr_objects

    ### walls 
    curr_walls = []

    for wall in scene["walls"]:
        if "exterior" in wall["id"]:
            continue
        wall_seg0 = wall['segment'][0]
        wall_seg1 = wall['segment'][1]
        # inside = all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in (wall_seg0, wall_seg1))

        if wall["roomId"] == room_name:
            wall["connect_exterior"] = wall["id"]+"|exterior"
            wall["polygon"] = [
                {"x": point["x"]-diff_x, "y": point["y"], "z": point["z"]-diff_y}
                for point in wall["polygon"]
            ]
            wall["segment"] = [[seg[0]-diff_x, seg[1]-diff_y] for seg in wall["segment"]]
            curr_walls.append(wall)
        new_walls = []
        for wall in curr_walls:
            new_wall = copy.deepcopy(wall)
            new_wall["polygon"] = [new_wall['polygon'][3], new_wall['polygon'][2], new_wall['polygon'][1], new_wall['polygon'][0]]
            new_wall.pop("connect_exterior", None)
            new_wall['segment'] = [new_wall['segment'][1], new_wall['segment'][0]]
            new_wall["id"] = new_wall["id"]+"|exterior"
            new_wall["material"]["name"] = "Walldrywall4Tiled"
            new_walls.append(new_wall)
    
    new_scene["walls"] = curr_walls+new_walls

    ## proceduralParams
    new_scene['proceduralParameters'] = scene['proceduralParameters']
    curr_lights = []
    for light in scene["proceduralParameters"]["lights"]:
        if light["id"] == "DirectionalLight":
            curr_lights.append(light)
        elif light["roomId"] == room_name:
            light['position']['x'] -= diff_x
            light['position']['z'] -= diff_y
            light['rgb'] = {"r": 0.95, "g": 0.95, "b": 0.95}
            curr_lights.append(light)
    
    new_scene["proceduralParameters"]["lights"] = curr_lights
    
    ### windows 
    curr_windows = []
    for window in scene["windows"]:
        if (window["room0"] == room_name) or (window["room1"] == room_name):
            curr_windows.append(window)
        elif (window["room0"] == room_name) or (window["room1"] != room_name):
            window["room1"] = room_name
            window["wall1"] = window["wall0"]+"|exterior"
        elif (window["room1"] == room_name) or (window["room0"] != room_name):
            window["room0"] = room_name
            window["wall0"] = window["wall1"]+"|exterior"
    
    new_scene["windows"] = curr_windows

    return new_scene
            

def all_edges_white(img):
    # Define a white pixel
    white = [255, 255, 255]

    # Check top edge
    if not np.all(np.all(img[0, :] == white, axis=-1)):
        return False
    # Check bottom edge
    if not np.all(np.all(img[-1, :] == white, axis=-1)):
        return False
    # Check left edge
    if not np.all(np.all(img[:, 0] == white, axis=-1)):
        return False
    # Check right edge
    if not np.all(np.all(img[:, -1] == white, axis=-1)):
        return False

    # If all the conditions met
    return True

def topdown_scene(scene, my_controller, width=1200, height=1200, without=False, rel=False):
    controller = my_controller
    controller.reset(scene=scene)
    # Setup the top-down camera
    
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])
    bounds = event.metadata["sceneBounds"]["size"]

    pose["fieldOfView"] = 60
    pose["position"]["y"] = bounds["y"]
    del pose["orthographicSize"]

    try:
        wall_height = wall_height = max(
            [point["y"] for point in scene["walls"][0]["polygon"]]
        )
    except:
        wall_height = 2.5

    for i in range(20):
        pose["orthographic"] = False

        pose["farClippingPlane"] = pose["position"]["y"] + 10
        pose["nearClippingPlane"] = pose["position"]["y"] - wall_height

        # add the camera to the scene
        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]

        # check if the edge of the frame is white
        if all_edges_white(top_down_frame):
            break

        pose["position"]["y"] += 0.75
    p_pose = pose["position"]
    image = Image.fromarray(top_down_frame)
    if without:
        return image
    if rel:
        bboxes = []

        for obj_name in rel:
            bbox = None
            for item in event.metadata["objects"]:
                if item["name"] == obj_name:
                    bbox = item["axisAlignedBoundingBox"]
                    break
            if bbox:
                bboxes.append(bbox)
            else:
                raise ValueError(f"Bounding box not found for object: {obj_name}")
        return image, pose, bboxes
    # return image
    return pil_image_to_base64_str(image)

def topdown_room(scene, room_name=None, my_controller = None, width=1200, height=1200): 
    ## 하나의 방에 대한.

    room = seperate_room(scene, room_name)
    image = topdown_scene(room, my_controller)

    return image

def multiview_scene(scene, my_controller, sys_args, width=1200, height=1200): 
    controller = my_controller
    controller.reset(scene=scene)
    
    # Setup the top-down camera
    images = {}
    for degree in [0, 180, 90, 270, "topdown"]:
        event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]

        pose["fieldOfView"] = 60
        pose["position"]["y"] = bounds["y"]
        if degree != "topdown":
            pose["rotation"]["y"] = degree
            pose["rotation"]["x"] = 70
        if degree == 0:
            pose["position"]["z"] = 0
        if degree == 180:
            pose["position"]["z"] *= 2
        if degree == 90:
            pose["position"]["x"] = 0
        if degree == 270:
            pose["position"]["x"] *= 2
        
        del pose["orthographicSize"]

        try:
            wall_height = wall_height = max(
                [point["y"] for point in scene["walls"][0]["polygon"]]
            )
        except:
            wall_height = 2.5

        for i in range(20):
            pose["orthographic"] = False

            pose["farClippingPlane"] = pose["position"]["y"] + 10
            pose["nearClippingPlane"] = pose["position"]["y"] - wall_height

            # add the camera to the scene
            event = controller.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )
            top_down_frame = event.third_party_camera_frames[-1] #원래 -1

            # check if the edge of the frame is white
            if all_edges_white(top_down_frame):
                break

            pose["position"]["y"] += 0.75

        image = Image.fromarray(top_down_frame)
        image = pil_image_to_base64_str(image)
        images[f"scene*/*multiview_scene({degree})*/*output__image"] = image
    return image
    #return images

def sideview_scene(scene, rooms, my_controller, width=1200, height=1200): 

    images = {}
    new_scene = scene
    for room, wall_info in rooms.items():
        for r in scene["rooms"]:
            if r["id"] == room:
                curr_room = r
        
        max_x= curr_room['floorPolygon'][2]['x']
        max_y= curr_room['floorPolygon'][2]['z']
        min_x= curr_room['floorPolygon'][0]['x']
        min_y= curr_room['floorPolygon'][0]['z']
        
        controller = my_controller
        for wall_id, direction in wall_info:

            if direction == "north" or direction == 0:
                new_pos = dict(x=max_x/2, y=new_scene["metadata"]["agent"]["position"]["y"], z=1)
                new_rot = dict(x=0, y=0, z=0)
                new_scene['objects'] = [obj for obj in new_scene["objects"] if abs(obj["position"]["z"] - max_y) <= 1]
            elif direction == "south" or direction == 180:
                new_pos = dict(x=max_x/2, y=new_scene["metadata"]["agent"]["position"]["y"], z=max_y-1)
                new_rot = dict(x=0, y=180, z=0)
                new_scene['objects'] = [obj for obj in new_scene["objects"] if abs(obj["position"]["z"] - min_y) <= 1]
            elif direction == "west" or direction == 270:
                new_pos = dict(x=max_x-1, y=new_scene["metadata"]["agent"]["position"]["y"], z=max_y/2)
                new_rot = dict(x=0, y=270, z=0)
                new_scene['objects'] = [obj for obj in new_scene["objects"] if abs(obj["position"]["x"] - min_x) <= 1]
            elif direction == "east" or direction == 90:
                new_pos = dict(x=1, y=new_scene["metadata"]["agent"]["position"]["y"], z=max_y/2)
                new_rot = dict(x=0, y=90, z=0)
                new_scene['objects'] = [obj for obj in new_scene["objects"] if abs(obj["position"]["x"] - max_x) <= 1]

            # controller.reset(scene=new_scene, commit_id=THOR_COMMIT_ID, width=1200, height=1200)
            
            controller.reset(scene =new_scene)
            
            event = controller.step(
            action="AddThirdPartyCamera",
            position=new_pos,
            rotation=new_rot,
            fieldOfView=90
            )

            frame = event.third_party_camera_frames[-1]
            image = Image.fromarray(frame)
            images[f"{wall_id}*/*wall in scene*/*output__image"] = pil_image_to_base64_str(image)
            
    return images

def sceneobject_image(scene, objs, my_controller, sys_args, width=1200, height=1200): 

    #make a new scene.
    images = {}
    controller = my_controller
    for obj in objs:
        id = obj["id"]
        assetId = obj["assetId"]
        position = obj["position"]
        rotation = obj["rotation"]
        roomId = obj["roomId"]
        new_scene = seperate_room(scene, roomId)

        new_room_vertices = new_scene["rooms"][0]["vertices"]
        position_x = (new_room_vertices[2][0] + new_room_vertices[0][0]) / 2
        position_y = (new_room_vertices[2][1] + new_room_vertices[0][1]) / 2
        obj["position"]["x"] = position_x
        obj["position"]["z"] = position_y
        position["x"] = position_x 
        position["z"] = position_y
        
        new_scene["objects"] = [obj]

        controller.reset(scene = new_scene)
        ## 특정 object에 대한 이미지를 특정 각도에서 제공. 
        vertices = cal_coords(assetId, position, rotation, sys_args)
        min_x = min([point[0] for point in vertices]) / 100
        min_y = min([point[1] for point in vertices]) / 100
        max_x = max([point[0] for point in vertices]) / 100
        max_y = max([point[1] for point in vertices]) / 100
        
        for direction in [0, 90, 180, 270]:
            if direction == 0:
                x = (min_x+max_x) / 2
                z = min_y - 1
                deg = 0
            elif direction == 90:
                x = (min_x) - 1
                z = (min_y+max_y) / 2
                deg = 90
            elif direction == 180:
                x = (min_x+max_x) / 2
                z = max_y + 1
                deg = 180
            elif direction == 270:
                x = (max_x) + 1
                z = (min_y+max_y) / 2
                deg = 270

            agent_y = scene["metadata"]["agent"]["position"]["y"]
            new_pos = dict(x=x, y=agent_y, z=z)
            new_rot = dict(x=0, y=deg, z=0)

            event = controller.step(
                action="AddThirdPartyCamera",
                position=new_pos,
                rotation=new_rot,
                fieldOfView=90
            )

            frame = event.third_party_camera_frames[-1]
            image = Image.fromarray(frame)
            images[f"{id}*/*{direction} view in scene*/*output__image"] = pil_image_to_base64_str(image) 
            
    return images

def scenewd_image(scene, objs, my_controller, width=1200, height=1200): 

    #make a new scene.
    images = {}
    scene["objects"] = []
    controller = my_controller
    controller.reset(scene=scene)
    
    for obj in objs:
        id = obj["id"]
        walls = get_wall_direction(scene, obj)

        for i, wall in enumerate(walls):
            obj_segment = wall["object_segment"]
            direction = wall["direction"]
            position_x = (obj_segment[0][0] + obj_segment[1][0]) / 2
            position_z = (obj_segment[0][1] + obj_segment[1][1]) / 2
            position_y = obj["assetPosition"]['y']
        
            if direction == "north":
                deg = 0
                position_z -= 2
                # position_x += 0.5
            elif direction == "south":
                deg = 180
                position_z += 2
                position_x += 0.2
            elif direction == "east":
                deg = 90
                position_x -= 2
                position_z -= 0.2
            elif direction == "west":
                deg = 270
                position_x += 2
                position_z -= 0.2

            agent_y = scene["metadata"]["agent"]["position"]["y"]
            new_pos = dict(x=position_x, y=position_y, z=position_z)
            new_rot = dict(x=0, y=deg, z=0)

            event = controller.step(
                action="AddThirdPartyCamera",
                position=new_pos,
                rotation=new_rot,
                fieldOfView=90
            )

            frame = event.third_party_camera_frames[-1]
            image = Image.fromarray(frame)
            images[f"{id}*/*{direction} view in scene*/*output__image"] = pil_image_to_base64_str(image) 
            
    return images

def is_same_line(seg1, seg2, tol=1e-4):
    # seg2 의 x, y(여기서는 z) 최소/최대값 계산
    x0, y0 = seg2[0]
    x1, y1 = seg2[1]
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)

    # seg1 의 각 점이 seg2 bbox 안에 있는지 확인
    for px, py in seg1:
        if not (min_x - tol <= px <= max_x + tol and
                min_y - tol <= py <= max_y + tol):
            return False
    return True

def get_wall_direction(scene, wd):

    walls = [w for w in scene.get("walls", []) if w["id"] in [wd["wall0"], wd["wall1"]]]

    output = []
    for wall in walls:
        wall_direction = None

        roomId = wall["roomId"]
        room = next((r for r in scene["rooms"] if r["id"] == roomId), None)
        if not room:
            continue

        # 벽의 선분
        segment = [[wall['polygon'][0]['x'], wall['polygon'][0]['z']],
                   [wall['polygon'][3]['x'], wall['polygon'][3]['z']]]

        # 방의 각 방향에 해당하는 선분들
        fp = room["floorPolygon"]
        min_x = min(fp[0]["x"], fp[1]["x"], fp[2]["x"], fp[3]["x"])
        max_x = max(fp[0]["x"], fp[1]["x"], fp[2]["x"], fp[3]["x"])
        min_z = min(fp[0]["z"], fp[1]["z"], fp[2]["z"], fp[3]["z"])
        max_z = max(fp[0]["z"], fp[1]["z"], fp[2]["z"], fp[3]["z"])
        room_edges = {
            "west":  [ [min_x, min_z], [min_x, max_z] ],
            "north": [ [min_x, max_z], [max_x, max_z] ],
            "east":  [ [max_x, max_z], [max_x, min_z] ],
            "south": [ [max_x, min_z], [min_x, min_z] ],
        }

        for direction, edge in room_edges.items():
            if is_same_line(segment, edge):
                wall_direction = direction
                break

        output.append({
            "wall_id": wall["id"],
            "object_segment": segment,
            "direction": wall_direction
        })

    return output


def get_view_matrix(pos, rot_deg):
    
    rot = R.from_euler('xyz', rot_deg, degrees=True)
    forward = rot.apply(np.array([0, 0, -1]))  
    up = rot.apply(np.array([0, 1, 0]))        

    target = pos + forward

    # LookAt 행렬 구성
    f = (target - pos)
    f = f / np.linalg.norm(f)
    r = np.cross(up, f)
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)

    view = np.eye(4)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ pos

    return view

def get_projection_matrix(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return proj

def world_to_screen(point, view, proj, img_size):
    p = np.append(point, 1.0)
    clip_space = proj @ view @ p
    ndc = clip_space[:3] / clip_space[3]
    
    x = (ndc[0] + 1) * 0.5 * img_size
    y = (ndc[1] + 1) * 0.5 * img_size
    return (x, y)

def draw_bbox(image, screen_points, color, thickness=2):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    connections = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in connections:
        pt1 = tuple(map(int, screen_points[i]))
        pt2 = tuple(map(int, screen_points[j]))
        cv2.line(image, pt1, pt2, color, thickness)
    
    return image

def generate_distinct_colors(n):
    colors = []
    skip_ranges = [(0.5, 0.7), (0.25, 0.35)]  # 피할 hue 구간 (남색, 짙은 녹색)
    valid_hues = []
    step = 0.001  # 아주 작은 간격으로 hue 생성
    h = 0.0
    while h < 1.0:
        if not any(start <= h <= end for start, end in skip_ranges):
            valid_hues.append(h)
        h += step
    selected_hues = [valid_hues[int(i * len(valid_hues) / n)] for i in range(n)]
    for hue in selected_hues:
        saturation = 1.0
        value = 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_255)
    return colors

def rotate_point(point, center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    x, y = point
    cx, cy = center

    # Flip Y to math coordinate system
    y = 2 * cy - y

    # Translate to origin
    x -= cx
    y -= cy

    # Rotate
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translate back
    x_rot += cx
    y_rot += cy

    # Flip Y back to image coordinate system
    y_rot = 2 * cy - y_rot

    return x_rot, y_rot

def compute_corner_points(center, size):
    """
    center: dict with keys 'x', 'y', 'z'
    size: dict with keys 'x', 'y', 'z'
    
    Returns:
        List of 8 corner points, each a list [x, y, z]
    """
    dx = size['x'] / 2
    dy = size['y'] / 2
    dz = size['z'] / 2

    cx, cy, cz = center['x'], center['y'], center['z']

    corner_points = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                x = cx + sx * dx
                y = cy + sy * dy
                z = cz + sz * dz
                corner_points.append([x, y, z])

    return corner_points

def objs_relation_image(scene, obj_ids, sys_args, my_controller=None, width=1200, height=1200):
    objects = []
    wd_flags = []
    wall_bboxes = []
    wall_bbox_names = []
    
    # Search in objects
    for obj_id in obj_ids:
        found = False
        for obj in scene["objects"]:
            if obj["id"] == obj_id:
                objects.append(obj)
                wd_flags.append(False)
                found = True
                break
        if found:
            continue

        # Search in windows
        window_bbox_names = []
        for obj in scene["windows"]:
            if obj["id"] == obj_id:
                objects.append(obj)
                wd_flags.append(True)
                window_bbox_names.append(obj["id"])
                found = True
                break
        if found:
            continue

        # Search in walls
        wall_bboxes = []
        wall_bbox_names = []
        wall_dir = []
        for obj in scene["walls"]:
            if obj["id"] == obj_id:
                objects.append(obj)   
                wd_flags.append(False)
                wall_rot = obj["direction"]
                wall_seg = obj["segment"]
                x_min = min(wall_seg[0][0], wall_seg[1][0])
                x_max = max(wall_seg[0][0], wall_seg[1][0])
                y_min = min(wall_seg[0][1], wall_seg[1][1])
                y_max = max(wall_seg[0][1], wall_seg[1][1])
                if (wall_rot == "north") or (wall_rot == "south"):
                    y_min -= 0.1
                    y_max += 0.1
                elif (wall_rot == "west") or (wall_rot == "east"):
                    x_min -= 0.1
                    x_max += 0.1

                center = {"x": (x_max + x_min) / 2, "y": obj["height"] / 2, "z": (y_max + y_min) / 2}
                size = {"x": x_max - x_min, "y": obj['height'], "z": y_max - y_min}
                corner_points = compute_corner_points(center, size)
                wall_bboxes.append({"center": center, "cornerPoints": corner_points, "size": size})
                wall_bbox_names.append(obj["id"])
                wall_dir.append(obj["direction"])
                # obj_ids.remove(obj["id"])

                found = True
                break
        if found:
            continue

        # Search in doors
        for obj in scene["doors"]:
            if obj["id"] == obj_id:
                objects.append(obj)
                wd_flags.append(True)
                found = True
                break

        if not found:
            raise ValueError(f"Could not find object: {obj_id}")

    # Handle receptacle relationships
    additional_ids = []
    for obj, is_wd in zip(objects, wd_flags):
        if '|' in obj['id'] and not is_wd:
            target_id = obj['id'].split('|')[1]
            additional_obj = next((o for o in scene["objects"] if o["id"] == target_id), None)
            if additional_obj:
                additional_ids.append(additional_obj['id'])

    # Collect all unique object IDs
    object_ids = [obj['id'] for obj, is_wd in zip(objects, wd_flags) if is_wd == False]
    object_ids += additional_ids
    object_ids = list(set(object_ids))  # remove duplicates

    # Find anchor object -> north = no change / south = 180 change / east = counter clockwise 90 / west = clockwise 90
    anchor_obj, wd_check = next(((obj, is_wd) for obj, is_wd in zip(objects, wd_flags) if obj["id"] == obj_ids[0]), None)
    
    dir_convert = {"north": 0, "south": 180, "east": 90, "west": 270} 

    if anchor_obj['id'] in wall_bbox_names:
        #find the index of the anchor in wall_bbox_names
        index = wall_bbox_names.index(anchor_obj["id"])
        anchor_rot = dir_convert[wall_dir[index]]
    elif wd_check:
        anchor_room1 = anchor_obj["room0"]
        anchor_room2 = anchor_obj["room1"]
        room_score = [0, 0]
        anchor_wall1 = anchor_obj["wall0"]
        anchor_wall2 = anchor_obj["wall1"]
        #check other objects and find the matching room

        if anchor_room1 == "exterior":
            for wall in scene["walls"]:
                if wall["id"] == anchor_wall2:
                    anchor_obj["direction"] = wall["direction"]
                    break
            anchor_obj["roomId"] = anchor_room2
        elif anchor_room2 == "exterior":
            for wall in scene["walls"]:
                if wall["id"] == anchor_wall1:
                    anchor_obj["direction"] = wall["direction"]
                    break
            anchor_obj["roomId"] = anchor_room1
        else:
            new_objects = [obj for obj in objects if "roomId" in list(obj.keys())]
            for obj in new_objects:
                if obj["roomId"] == anchor_room1 and obj["id"] != anchor_obj["id"]:
                    room_score[0] += 1
                if obj["roomId"] == anchor_room2 and obj["id"] != anchor_obj["id"]:
                    room_score[1] += 1

            if room_score[0] >= room_score[1]:
                anchor_obj["roomId"] = anchor_room1
                #get the wall info from the scene
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall1:
                        anchor_obj["direction"] = wall["direction"]
                        break
            else:
                anchor_obj["roomId"] = anchor_room2
                #get the wall info from the scene
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall2:
                        anchor_obj["direction"] = wall["direction"]
                        break

        anchor_rot = dir_convert[anchor_obj["direction"]]
    else:
        anchor_rot = anchor_obj["rotation"]["y"]
    rot_conv = {0: 180, 90:90, 180:0, 270:270}
    anchor_rot = rot_conv[anchor_rot]

    # Filter scene
    new_scene = scene.copy()
    new_scene["objects"] = [obj for obj in scene["objects"] if obj['id'] in object_ids]

    new_obj_ids = [obj for obj in obj_ids if obj not in wall_bbox_names]

    # Render topdown
    # image, cam_set, bboxes = topdown_scene(new_scene, my_controller, width=width, height=height, rel=new_obj_ids)

    index = scene["index"]
    # image = Image.open(f'../data/data_{index}/scene_images/scene*|*topdown_scene*|*image.png')
    # with open(f'../data/data_{index}/scene_images/scene_metadata.json', 'r') as f:
    #     obj_metadata = json.load(f)
    image = Image.open(os.path.join(sys_args.data_path, f'data_{index}/scene_images/scene*|*topdown_scene*|*image.png'))
    with open(os.path.join(sys_args.data_path, f'data_{index}/scene_images/scene_metadata.json'), 'r') as f:
        obj_metadata = json.load(f)

    cam_set = obj_metadata[0]

    bboxes = []

    for obj_name in new_obj_ids:
        bbox = None
        for item in obj_metadata[1:]:
            if item["name"] == obj_name:
                bbox = item["axisAlignedBoundingBox"]
                break
        if bbox:
            bboxes.append(bbox)
        else:
            raise ValueError(f"Bounding box not found for object: {obj_name}")
    
    obj_ids = new_obj_ids + wall_bbox_names

    bboxes += wall_bboxes

    cam_pos = np.array([cam_set["position"]["x"], cam_set["position"]["y"], cam_set["position"]["z"]])
    cam_rot = np.array([cam_set["rotation"]["x"], cam_set["rotation"]["y"], cam_set["rotation"]["z"]])
    fov = 60
    aspect_ratio = 1.0  # since 1200 x 1200
    near, far = cam_set["nearClippingPlane"], cam_set["farClippingPlane"]
    image_size = 1200

    view = get_view_matrix(cam_pos, cam_rot)
    proj = get_projection_matrix(fov, aspect_ratio, near, far)

    colors = generate_distinct_colors(len(bboxes))

    try:
        # font = ImageFont.truetype("../utils/roboto.ttf", 20)
        font = ImageFont.truetype(os.path.join(sys_args.base_path, "evaluation/utils/roboto.ttf"), 20)
    except IOError:
        font = ImageFont.load_default()

    placed_text_boxes = []

    for i, bbox in enumerate(bboxes):
        screen_points = [world_to_screen(p, view, proj, image_size) for p in bbox["cornerPoints"]]
        color = colors[i]
        image = draw_bbox(image, screen_points, color)

        image = Image.fromarray(image)
        image = image.rotate(anchor_rot, expand=True)

        #rotate screen_points accordingly

        w, h = image.size
        center = (w // 2, h // 2)
        screen_points = [rotate_point(p, center, anchor_rot) for p in screen_points]

        draw = ImageDraw.Draw(image)

        xs, ys = zip(*screen_points)
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        obj_id = obj_ids[i]

        bbox_coords = draw.textbbox((0, 0), obj_id, font=font)
        text_width = bbox_coords[2] - bbox_coords[0]
        text_height = bbox_coords[3] - bbox_coords[1]
        padding = 5

        image_width, image_height = image.size

        max_attempts = 10
        offset_step_x = text_width + 2 * padding
        offset_step_y = text_height + 2 * padding

        directions = [
            (0, -1),  # 위
            (0, 1),   # 아래
            (-1, 0),  # 왼쪽
            (1, 0),   # 오른쪽
        ]

        final_position = None

        for attempt in range(1, max_attempts + 1):
            for dx, dy in directions:
                text_x = (min_x + max_x) // 2 - text_width // 2 + dx * attempt * offset_step_x
                text_y = (min_y + max_y) // 2 - text_height // 2 + dy * attempt * offset_step_y

                if text_x < padding or text_x + text_width + padding > image_width:
                    continue
                if text_y < padding or text_y + text_height + padding > image_height:
                    continue

                text_box = (text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding)
                bbox_box = (min_x, min_y, max_x, max_y)
                if (text_box[0] < bbox_box[2] and text_box[2] > bbox_box[0] and
                    text_box[1] < bbox_box[3] and text_box[3] > bbox_box[1]):
                    continue

                overlap = False
                for placed in placed_text_boxes:
                    if (text_box[0] < placed[2] and text_box[2] > placed[0] and
                        text_box[1] < placed[3] and text_box[3] > placed[1]):
                        overlap = True
                        break
                if overlap:
                    continue

                final_position = (text_x, text_y, text_box)
                break  # 방향 루프 종료
            if final_position is not None:
                break  # 시도 루프 종료

        if final_position is None:
            continue  # fallback 처리 필요하면 여기에 추가

        text_x, text_y, text_box = final_position

        draw.rectangle(text_box, fill=(255, 255, 255), outline=color, width=5)
        draw.text((text_x, text_y), obj_id, fill=(0, 0, 0), font=font)


        placed_text_boxes.append(text_box)
        if i != len(bboxes) - 1:
            image = image.rotate(-anchor_rot, expand=True)

    if isinstance(image, np.ndarray):
        result_image = Image.fromarray(image)
    else:
        result_image = image
    result_image = pil_image_to_base64_str(result_image)

    return result_image

    
def get_close_image(scene, items, sys_args, my_controller=None, width=1200, height=1200): 
    #open jsonfile
    to_load = os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json')
    with open(to_load, 'r') as f:
        annotations = json.load(f)  
    

    # ## 특정 object에 대해서 가까운 topdown 이미지를 제공.
    # my_controller = Controller(
    #         commit_id=THOR_COMMIT_ID,
    #         agentMode="default",
    #         makeAgentsVisible=False,
    #         visibilityDistance=1.5,
    #         scene='Procedural',
    #         width=1200,
    #         height=1200,
    #         fieldOfView=90,
    #         action_hook_runner=ProceduralAssetHookRunner(
    #             asset_directory=OBJAVERSE_ASSET_DIR,
    #             asset_symlink=True,
    #             verbose=True,
    #         ),
    #     )
    controller = my_controller
    controller.reset(scene=scene)
    positions = {}
    dir_convert = {"north": 0, "south": 180, "east": 90, "west": 270} 
    for item in items:
        if item[1] == 'obj':
            positions[item[0]['id']] = (item[0]['position'], item[0]['rotation'])
            anchor_height = annotations[item[0]['assetId']]['size'][1]
        elif item[1] == 'wd':
            anchor_obj = item[0]
            anchor_room1 = anchor_obj["room0"]
            anchor_room2 = anchor_obj["room1"]
            room_score = [0, 0]
            anchor_wall1 = anchor_obj["wall0"]
            anchor_wall2 = anchor_obj["wall1"]
            #check other objects and find the matching room
            anchor_rot = None
            anchor_rot1 = None
            anchor_rot2 = None

            if anchor_room1 == "exterior":
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall2:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_obj["roomId"] = anchor_room2
                anchor_rot = dir_convert[anchor_obj["direction"]]
            elif anchor_room2 == "exterior":
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall1:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_obj["roomId"] = anchor_room1
                anchor_rot = dir_convert[anchor_obj["direction"]]
            else:
                anchor_obj["roomId"] = anchor_room1
                    #get the wall info from the scene
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall1:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_rot1 = dir_convert[anchor_obj["direction"]]
                
                anchor_obj["roomId"] = anchor_room2
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall2:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_rot2 = dir_convert[anchor_obj["direction"]]
            anchor_obj_seg = anchor_obj["segment"]
            position_x = (anchor_obj_seg[0][0] + anchor_obj_seg[1][0]) / 2
            position_z = (anchor_obj_seg[0][1] + anchor_obj_seg[1][1]) / 2
            position_y = anchor_obj["height"] / 2
            position = dict(x=position_x, y=position_y, z=position_z)
            
            if anchor_rot:
                positions[item[0]['id']] = (position, anchor_rot)
            if anchor_rot1:
                positions[item[0]['id'] + '_1'] = (position, anchor_rot1)
            if anchor_rot2:
                positions[item[0]['id'] + '_2'] = (position, anchor_rot2)
            anchor_height = anchor_obj["height"]
    images = {}
    for id, i_set in positions.items():
        position, rotation = i_set

        if anchor_height >= 100:
            my_degree = anchor_height * 0.05
        else:
            my_degree = anchor_height * 0.035
        for degree in [my_degree]:
            new_pos = dict(x=position["x"], y=position["y"]+degree, z=position["z"])
            new_rot = dict(x=90, y=0, z=0)

            event = controller.step(
                action="AddThirdPartyCamera",
                position=new_pos,
                rotation=new_rot,
                fieldOfView=90
            )
            frame = event.third_party_camera_frames[-1]
            image = Image.fromarray(frame)

            rot_conv = (180 - rotation['y']) % 360
            image = image.rotate(rot_conv, expand=True)

            images[f"{id}*/*topdown({degree}m)*/*output__image"] = pil_image_to_base64_str(image)
    return images


def get_front_image(scene, items, sys_args, my_controller=None, width=1200, height=1200): 
    #open jsonfile
    # to_load = 'obj_annotations.json'
    to_load = os.path.join(sys_args.base_path, 'evaluation/utils/obj_annotations.json')
    with open(to_load, 'r') as f:
        annotations = json.load(f)  

    # ## 특정 object에 대해서 가까운 frontview 이미지를 제공.
    # my_controller = Controller(
    #         commit_id=THOR_COMMIT_ID,
    #         agentMode="default",
    #         makeAgentsVisible=False,
    #         visibilityDistance=1.5,
    #         scene='Procedural',
    #         width=1200,
    #         height=1200,
    #         fieldOfView=90,
    #         action_hook_runner=ProceduralAssetHookRunner(
    #             asset_directory=OBJAVERSE_ASSET_DIR,
    #             asset_symlink=True,
    #             verbose=True,
    #         ),
    #     )
    controller = my_controller
    positions = {}
    dir_convert = {"north": 0, "south": 180, "east": 90, "west": 270} 
    for item in items:

        if item[1] == 'obj':
            anchor_height = max(annotations[item[0]['assetId']]['size'][0], annotations[item[0]['assetId']]['size'][1], annotations[item[0]['assetId']]['size'][2]) #z. 어차피 앞에서 볼거니까. 
            positions[item[0]['id']] = (item[0]['position'], item[0]['rotation'], item[0]['roomId'], anchor_height)
        elif item[1] == 'wd':
            anchor_obj = item[0]
            anchor_room1 = anchor_obj["room0"]
            anchor_room2 = anchor_obj["room1"]
            room_score = [0, 0]
            anchor_wall1 = anchor_obj["wall0"]
            anchor_wall2 = anchor_obj["wall1"]
            #check other objects and find the matching room
            anchor_rot = None
            anchor_rot1 = None
            anchor_rot2 = None

            if anchor_room1 == "exterior":
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall2:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_obj["roomId"] = anchor_room2
                anchor_rot = dir_convert[anchor_obj["direction"]]
            elif anchor_room2 == "exterior":
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall1:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_obj["roomId"] = anchor_room1
                anchor_rot = dir_convert[anchor_obj["direction"]]
            else:
                anchor_obj["roomId"] = anchor_room1
                    #get the wall info from the scene
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall1:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_rot1 = dir_convert[anchor_obj["direction"]]
                
                anchor_obj["roomId"] = anchor_room2
                for wall in scene["walls"]:
                    if wall["id"] == anchor_wall2:
                        anchor_obj["direction"] = wall["direction"]
                        break
                anchor_rot2 = dir_convert[anchor_obj["direction"]]
            anchor_obj_seg = anchor_obj["segment"]
            position_x = (anchor_obj_seg[0][0] + anchor_obj_seg[1][0]) / 2
            position_z = (anchor_obj_seg[0][1] + anchor_obj_seg[1][1]) / 2
            position_y = anchor_obj["height"] / 2
            position = dict(x=position_x, y=position_y, z=position_z)

            anchor_height = anchor_obj["height"]
            room = anchor_obj['roomId']
            
            if anchor_rot:
                positions[item[0]['id']] = (position, anchor_rot, room, anchor_height)
            if anchor_rot1:
                positions[item[0]['id'] + '_1'] = (position, anchor_rot1, room, anchor_height)
            if anchor_rot2:
                positions[item[0]['id'] + '_2'] = (position, anchor_rot2, room, anchor_height)

    images = {}
    for id, i_set in positions.items():
        position, rotation, room, anchor_height = i_set
        rotation = rotation['y']
        new_scene = scene.copy()

        computed = {}
        if anchor_height > 25:
            offset = 0.01
            position['y'] += 0.2
        elif anchor_height > 10:
            offset = 0.04
            position['y'] += 0.1
        else:
            offset = 0.1
            position['y'] += 0.05

        dx = anchor_height * offset * math.sin(math.radians(rotation))
        dz = anchor_height * offset * math.cos(math.radians(rotation))

        priority_pos = {'position':{'x':position['x'] + dx, 'y':position['y'], 'z':position['z'] + dz}, 'rotation': (rotation + 180) % 360}
        second_pos = {'position':{'x':position['x'] - dx, 'y':position['y'], 'z':position['z'] - dz}, 'rotation': rotation}

        if math.fabs(math.cos(math.radians(rotation))) >= math.fabs(math.sin(math.radians(rotation))):
            third_pos = {'position':{'x':position['x'] + anchor_height*offset, 'y':position['y'], 'z':position['z']}, 'rotation':90}
            fourth_pos = {'position':{'x':position['x'] - anchor_height*offset, 'y':position['y'], 'z':position['z']}, 'rotation':270}
        else:
            third_pos = {'position':{'x':position['x'], 'y':position['y'], 'z':position['z'] + anchor_height*offset}, 'rotation':0}
            fourth_pos = {'position':{'x':position['x'], 'y':position['y'], 'z':position['z'] - anchor_height*offset}, 'rotation':180}

        
        for idx, pos in enumerate([priority_pos, second_pos, third_pos, fourth_pos]):
            name = ['priority_pos', 'second_pos', 'third_pos', 'fourth_pos'][idx]
            computed[name] = pos
        

        room_data = next(r for r in scene['rooms'] if r['id'] == room)
        floor = room_data['floorPolygon']
        xs = [p['x'] for p in floor]
        zs = [p['z'] for p in floor]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        for key in ('priority_pos', 'second_pos', 'third_pos', 'fourth_pos'):
            pos = computed[key]['position']
            if min_x < pos['x'] < max_x and min_z < pos['z'] < max_z:
                cam_pos = pos
                cam_rot = {'x':0, 'y':computed[key]['rotation'], 'z':0}
                break
        
        start_x = min(position['x'], cam_pos['x'])
        end_x = max(position["x"], cam_pos["x"])
        start_z = min(position['z'], cam_pos['z'])
        end_z = max(position['z'], cam_pos['z'])

        new_scene['objects'] = [
            obj for obj in new_scene['objects']
            if not ( start_x < obj['position']['x'] < end_x
                    or start_z < obj['position']['z'] < end_z )
        ]

        controller.reset(scene=new_scene)

        event = controller.step(
                action="AddThirdPartyCamera",
                position=cam_pos,
                rotation=cam_rot,
                fieldOfView=90
            )
        frame = event.third_party_camera_frames[-1]
        image = Image.fromarray(frame)

        images[f"{id}*/*frontview*/*output__image"] = pil_image_to_base64_str(image)
        
    return images

def load_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def generate_distinct_color(index, total):
    """Generate visually distinct RGB color using HSV."""
    hue = index / total
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return tuple(int(c * 255) for c in rgb)

# def draw_room(scene, base64_img, image_width=1200, image_height=1200):
#     img = load_base64_image(base64_img)
#     draw = ImageDraw.Draw(img)

#     # Compute world bounds
#     all_x = [x for room in scene["rooms"] for x, _ in room["vertices"]]
#     all_z = [z for room in scene["rooms"] for _, z in room["vertices"]]
#     world_x_min, world_x_max = min(all_x), max(all_x)
#     world_z_min, world_z_max = min(all_z), max(all_z)

#     try:
#         font = ImageFont.truetype("utils/roboto.ttf", 16)
#     except IOError:
#         font = ImageFont.load_default()

#     total_rooms = len(scene["rooms"])
#     room_draw_info = []

#     # Step 1: Draw all polygons and collect metadata
#     for idx, room in enumerate(scene["rooms"]):
#         room_id = room["id"]
#         room_coords = room["vertices"]
#         color = generate_distinct_color(idx, total_rooms)

#         # Convert vertices to image coords
#         img_coords = [unity_to_image_coords(x, z, world_x_min, world_x_max, world_z_min, world_z_max,
#                                             image_width, image_height) for x, z in room_coords]

#         # Draw room outline
#         draw.polygon(img_coords, outline=color, width=7)

#         # Compute bounding box of polygon
#         xs = [pt[0] for pt in img_coords]
#         ys = [pt[1] for pt in img_coords]
#         bbox = (min(xs), min(ys), max(xs), max(ys))

#         room_draw_info.append({
#             "id": str(room_id),
#             "bbox": bbox,
#             "color": color
#         })

#     # Step 2: Draw text labels just outside bounding boxes
#     for info in room_draw_info:
#         text = info["id"]
#         bbox = draw.textbbox((0, 0), text, font=font)
#         text_w = bbox[2] - bbox[0]
#         text_h = bbox[3] - bbox[1]

#         # Place the label just above the top-right corner of the room box
#         box_x1, box_y1, box_x2, box_y2 = info["bbox"]
#         label_x = box_x2 + 5  # right outside
#         label_y = box_y1 - text_h - 5  # slightly above

#         # Keep label inside the image bounds
#         label_x = min(label_x, image_width - text_w - 2)
#         label_y = max(label_y, 2)

#         label_box = [label_x, label_y, label_x + text_w, label_y + text_h]
#         draw.rectangle(label_box, fill="white", outline=info["color"])
#         draw.text((label_box[0], label_box[1]), text, fill="black", font=font)

#     return img

def draw_room(scene, base64_img, sys_args, image_width=1200, image_height=1200):
    img = base64_img
    # img = load_base64_image(base64_img)
    draw = ImageDraw.Draw(img)

    # Compute world bounds
    all_x = [x for room in scene["rooms"] for x, _ in room["vertices"]]
    all_z = [z for room in scene["rooms"] for _, z in room["vertices"]]
    world_x_min, world_x_max = min(all_x), max(all_x)
    world_z_min, world_z_max = min(all_z), max(all_z)

    try:
        # font = ImageFont.truetype("../utils/roboto.ttf", 30)
        font = ImageFont.truetype(os.path.join(sys_args.base_path, "evaluation/utils/roboto.ttf"), 30)
    except IOError:
        font = ImageFont.load_default()

    total_rooms = len(scene["rooms"])
    room_draw_info = []

    # Step 1: Draw all room polygons and store metadata
    for idx, room in enumerate(scene["rooms"]):
        room_id = room["id"]
        room_coords = room["vertices"]
        color = generate_distinct_color(idx, total_rooms)

        img_coords = [unity_to_image_coords(x, z, world_x_min, world_x_max, world_z_min, world_z_max,
                                            image_width, image_height) for x, z in room_coords]

        draw.polygon(img_coords, outline=color, width=7)

        xs = [pt[0] for pt in img_coords]
        ys = [pt[1] for pt in img_coords]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        room_draw_info.append({
            "id": str(room_id),
            "bbox": bbox,
            "color": color
        })

    room_boxes = [box(*info["bbox"]) for info in room_draw_info]

    # Step 2: Draw labels outside each room's box
    for i, info in enumerate(room_draw_info):
        text = info["id"]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        room_box = box(*info["bbox"])
        label_placed = False

        x1, y1, x2, y2 = info["bbox"]
        offset_positions = [
            (x2 + 5, y1 - text_h - 5),     # right-top
            (x2 + 5, y2 + 5),              # right-bottom
            (x1, y2 + 5),                  # bottom-left
            (x1 - text_w - 5, y1 - text_h) # top-left
        ]

        for label_x, label_y in offset_positions:
            # Clamp within image
            label_x = min(max(2, label_x), image_width - text_w - 2)
            label_y = min(max(2, label_y), image_height - text_h - 2)
            label_box = [label_x, label_y, label_x + text_w, label_y + text_h]
            label_shape = box(*label_box)

            if not any(label_shape.intersects(rb) and rb != room_box for rb in room_boxes):
                border_width = 10
                for i in range(border_width):
                    offset_box = [
                        label_box[0] - i,
                        label_box[1] - i,
                        label_box[2] + i,
                        label_box[3] + i
                    ]
                    draw.rectangle(offset_box, outline=info["color"])
                draw.rectangle(label_box, fill="white")  # fill inner area
                draw.text((label_box[0], label_box[1]), text, fill="black", font=font)
                label_placed = True
                break

        if not label_placed:
            # Fallback position (still clamped)
            label_x = min(max(2, x1), image_width - text_w - 2)
            label_y = min(max(2, y1 - text_h - 2), image_height - text_h - 2)
            label_box = [label_x, label_y, label_x + text_w, label_y + text_h]
            border_width = 10
            for i in range(border_width):
                offset_box = [
                    label_box[0] - i,
                    label_box[1] - i,
                    label_box[2] + i,
                    label_box[3] + i
                ]
                draw.rectangle(offset_box, outline=info["color"])
            draw.rectangle(label_box, fill="white")  # fill inner area
            draw.text((label_box[0], label_box[1]), text, fill="black", font=font)

    return pil_image_to_base64_str(img)

def unity_to_image_coords(x, z, world_x_min, world_x_max, world_z_min, world_z_max, image_width=1200, image_height=1200):
    world_width = world_x_max - world_x_min
    world_height = world_z_max - world_z_min

    # Use the larger dimension for consistent scaling
    scale = max(world_width, world_height)

    # Center the coordinates in the image
    offset_x = (scale - world_width) / 2
    offset_z = (scale - world_height) / 2

    norm_x = (x - world_x_min + offset_x) / scale
    norm_z = (z - world_z_min + offset_z) / scale

    col = int(norm_x * image_width)
    row = int((1 - norm_z) * image_height)  # flip z because image y-axis goes down

    return col, row


if __name__ =="__main__":
    scene = "../0_query_annotool/anno_scenes/data_13/data_13.json"
    with open(scene, "r") as f:
        scene = json.load(f)
    room = "single-room"
    # image = topdown_room(scene, room)
    # image.show()