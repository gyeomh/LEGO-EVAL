```json
{
  "query": <Instruction for generating the scene>, # (str)
  "doors": [
    {
      "assetId": <asset ID of the door>, # ID of the asset. Must be identical to the glb file (str)
      "id": <Door name>, # Unique name of the door. Any name possible. (str)
      "openable": <openable>, # Is the door openable. (bool)
      "openness": <open degree>, # How open is the door in degrees. (float)
      "room0": <room id>, # ID of the room that the door connects to. (str)
      "room1": <room id>, # ID of the room that the door connects to. (str)
      "wall0": <wall id>, # ID of the wall that the door is on. (str)
      "wall1": <wall id>, # ID of the wall that the door is on. (str)
      "holePolygon": [ # the placement of the door on the wall. 전체 공간에서의 위치가 아니라, 벽에서의 위치임.
        { 
          "x": <x coordinate>, # x-coordinate of the left-bottom of the hole. x-axis denotes the left to the right of the wall. (float)
          "y": <y coordinate>, # y-coordinate of the left-bottom of the hole. y-axis denotes the bottom to top of the wall. (float)
          "z": <z coordinate>>, # z-coordinate of the left-bottom of the hole. z-axis denotes the depth of the wall. 0 as default (float)
        },
        {
          "x": <x coordinate>, # x-coordinate of the right-top of the hole. x-axis denotes the left to the right of the wall. (float)
          "y": <y coordinate>, # y-coordinate of the right-top of the hole. y-axis denotes the bottom to top of the wall. (float)
          "z": <z coordinate>>, # z-coordinate of the right-top of the hole. z-axis denotes the depth of the wall. 0 as default (float)
        }
      ],
      "assetPosition": { # The center coordinate of the door. Must be the center of holePolygon
        "x": <x coordinate>, # (float)
        "y": <y coordinate>, # (float)
        "z": <z coordinate>, # (float)
      }
    },
    ...
  ],
  "metadata": { # This is the default setting. You do not have to adjust anything. 
    "agent": { 
      "horizon": 30,
      "position": {
        "x": 0,
        "y": 0,
        "z": 0
      },
      "rotation": {
        "x": 0,
        "y": 0,
        "z": 0
      },
      "standing": True
    },
    "roomSpecId": "",
    "schema": "1.0.0",
    "warnings": {},
    "agentPoses": {
      "arm": {
        "horizon": 30,
        "position": {
         "x": 0,
         "y": 0,
         "z": 0
      },
        "rotation": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "standing": true
      },
      "default": {
        "horizon": 30,
        "position": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "rotation": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "standing": true
      },
      "locobot": {
        "horizon": 30,
        "position": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "rotation": {
          "x": 0,
          "y": 0,
          "z": 0
        }
      },
      "stretch": {
        "horizon": 30,
        "position": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "rotation": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "standing": true
      }
    }
  },
  "objects": [
    {
      "assetId": <asset ID of the object>, # ID of the asset. Must be identical to the glb file (str)
      "id": <object name>, # Unique name of the object. Any name possible. (str)
      "object_name": <Object name>, # Identical as the ID. 
      "kinematic": true,
      "position": { # Position of the object within the whole scene.
        "x": <x coordinate>, # (float)
        "y": <y coordiante>, # (float)
        "z": <z coordiante> # (float)
      },
      "rotation": {
        "x": <x-axis rotation>, # Set as 0 for default. (float)
        "y": <y-axis rotation>, # Set as 0 for default. (float)
        "z": <z-axis rotation> # Set as 0 for default. (float)
      },
      "roomId": <room ID>, # ID of the room that the object is in. (str)
      "material": null,
      "layer": "Procedural0"
    },
    ...
  ],
  "proceduralParameters": {
    "ceilingColor": { # adjust as needed. 
      "b": 0.3,
      "g": 0.3,
      "r": 0.4
    },
    "ceilingMaterial": {
      "name": <material name of the ceiling> # Set as "OrangeDrywall 1" for default. (str)
    },
    "floorColliderThickness": 1.0,
    "lights": [ # point lights are required for each room. Make sure you have one for all rooms. 
      {
        "id": "DirectionalLight", # Global lighting. Default setting. 
        "position": {
          "x": 0.84,
          "y": 0.1855,
          "z": -1.09
        },
        "rotation": {
          "x": 66,
          "y": 75,
          "z": 0
        },
        "shadow": {
          "type": "Soft",
          "strength": 1,
          "normalBias": 0,
          "bias": 0,
          "nearPlane": 0.2,
          "resolution": "FromQualitySettings"
        },
        "type": "directional",
        "intensity": 1,
        "indirectMultiplier": 1.0,
        "rgb": {
          "r": 1.0,
          "g": 1.0,
          "b": 1.0
        }
      },
      {
        "id": <ID of the light for room>, # Unique name of light for room (str)
        "type": "point",
        "position": {
          "x": <x coordiante>, # Should be center x coordinate of the room. 
          "y": <y coordinate>, # Should be somewhat similar or identical to wall height. 
          "z": <z coordinate> # Should be center z coordinate of the room. 
        },
        "intensity": <intensity of the light>, # default to 0.75 (float)
        "range": <range of the light>, # default to 15 (float)
        "rgb": { # <color of the light>. default settings below. 
          "r": 1.0,
          "g": 0.85,
          "b": 0.70
        },
        "shadow": { # <shadow of the light>. default settings below. 
          "type": "Soft",
          "strength": 1,
          "normalBias": 0,
          "bias": 0.05,
          "nearPlane": 0.2,
          "resolution": "FromQualitySettings"
        },
        "roomId": <room ID>, # ID of the room that the light is in. (str)
        "layer": "Procedural0",
        "cullingMaskOff": [
          "Procedural1",
          "Procedural2",
          "Procedural3"
        ]
      },
      ...
    ],
    "receptacleHeight": 0.7,
    "reflections": [],
    "skyboxId": "SkySFDowntown"
  },
  "rooms": [
    {
      "id": <ID of the room>, # Unique name of the room. (str)
      "floorMaterial": {
        "name": <name of the material> # Set as "DarkWoodFloors" for default. (str)
      },
      "wallMaterial": {
        "name": <name of the material> # Set as "OrangeDrywall 1" for default. (str) 
      },
      "vertices": [ # Four coordinates of the room (x coordinate, z coordinate). (float) 
        [ # bottom-left coordinate
          <BL x coordinate>,
          <BL z coordinate>
        ],
        [ # top-left coordinate
          <TL x coordinate>,
          <TL z coordinate>
        ],
        [ # top-right coordinate
          <TR x coordinate>,
          <TR z coordinate>
        ],
        [ # bottom-right coordinate
          <BR x coordinate>,
          <BR z coordinate>
        ]
      ],
      "floorPolygon": [ # Identical as the vertices, but with y axis added. Set y axis as 0. 
        {
          "x": <BL x coordinate>,
          "y": 0,
          "z": <BL z coordinate>
        },
        {
          "x": <TL x coordinate>,
          "y": 0,
          "z": <TL z coordinate>
        },
        {
          "x": <TR x coordinate>,
          "y": 0,
          "z": <TR z coordinate>
        },
        {
          "x": <BR x coordinate>,
          "y": 0,
          "z": <BR z coordinate>,
        }
      ],
      "children": [], # leave as empty 
      "ceilings": [], # leave as empty
      "layer": "Procedural0"
    },
    ...
  ],
  "walls": [ # There should be information for both side of the wall. The walls that are outside must also exist. 
    {
      "id": <ID of the wall>, # Unique ID of the wall. If the wall is outside, then 'exterior' must be included in the ID.
      "roomId": <ID of the room>, # ID of the room. If the wall is outside, then set it as 'exterior'.
      "material": {
        "name": <name of the material> # Set as "OrangeDrywall 1" for default. (str) 
      },
      "polygon": [ # Coordinates of the wall.
        { # bottom-left coordinate
          "x": <x coordinate of the wall>,
          "y": <y coordinate of the wall>,
          "z": <z coordinate of the wall>
        },
        { # top-left coordinate
          "x": <x coordinate of the wall>,
          "y": <y coordinate of the wall>,
          "z": <z coordinate of the wall>,
        },
        { # top-right coordinate
          "x": <x coordinate of the wall>,
          "y": <y coordinate of the wall>,
          "z": <z coordinate of the wall>,
        },
        { # top-left coordinate
          "x": <x coordinate of the wall>,
          "y": <y coordinate of the wall>,
          "z": <z coordinate of the wall>,
        }
      ],
          "roomId": <ID of the room>, # identical to the 'roomId' above. 
          "wallId": <ID of the wall> #identical to the 'id' above. 
        }
        ],
      "width": <width of the wall>, # (float)
      "height": <height of the wall>, # (float)
      "direction": <direction of the wall>, # North, South, East or West. (str)
      "segment": [ #bottom-left coordinate of the wall, and top-right coordinate of the wall. x coordinate and z coordinate. 
        [
          <x coordinate of bottom-left>,
          <z coordinate of bottom-left>
        ],
        [
          <x coordinate of top-right>,
          <z coordinate of top-right>
        ]
      ],
      "connect_exterior": <id of the wall that is on the exterior>, # delete if the other side of the wall is not on the exterior. 
      "layer": "Procedural0"
    },
    ...
  ],
  "windows": [
    {
      "assetId": <asset ID of the window>, # ID of the asset. Must be identical to the glb file (str)
      "id": <Name of the window>, # Unique name of the window
      "room0": <room ID of the other room that the window connects to>, # (str)
      "room1": <room ID of the other room that the window connects to>, # If the other side if exterior, set it as the same room as room0. (str) 
      "wall0": <ID of the wall that the window is on>, # (str)
      "wall1": <ID of the other wall that the window is on>, # (str)
      "holePolygon": [ # the placement of the window on the wall. 전체 공간에서의 위치가 아니라, 벽에서의 위치임. wall0 기준. 
        { 
          "x": <x coordinate>, # x-coordinate of the left-bottom of the hole. x-axis denotes the left to the right of the wall. (float)
          "y": <y coordinate>, # y-coordinate of the left-bottom of the hole. y-axis denotes the bottom to top of the wall. (float)
          "z": <z coordinate>>, # z-coordinate of the left-bottom of the hole. z-axis denotes the depth of the wall. 0 as default (float)
        },
        {
          "x": <x coordinate>, # x-coordinate of the right-top of the hole. x-axis denotes the left to the right of the wall. (float)
          "y": <y coordinate>, # y-coordinate of the right-top of the hole. y-axis denotes the bottom to top of the wall. (float)
          "z": <z coordinate>>, # z-coordinate of the right-top of the hole. z-axis denotes the depth of the wall. 0 as default (float)
        }
      ],
      "assetPosition": { # The center coordinate of the door. Must be the center of holePolygon
        "x": <x coordinate>, # (float)
        "y": <y coordinate>, # (float)
        "z": <z coordinate>, # (float)
      },
      "roomId": <ID of the room that the window is in>, # ID of room 0 or room 1. (str)
      "layer": "Procedural0" #default
    },
  ]
}
  ```