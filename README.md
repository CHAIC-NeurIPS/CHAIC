# TDW Multi-Agent Transport

## Codebase

```
|__ tdw-gym/ 					main code
|       |__ challenge.py         main evaluation code
|       |__ tdw_gym.py           main env code
|       |__ h_agent.py           HP Agent
|       |__ lm_agent.py          Cooperative LLM Agent
|
|__ scenes/ 			code for generating dataset
|
|__ dataset/ 					dataset configuration & dataset storage
|
|__ transport_challenge_multi_agent/ low level controller
|
|__ scripts/ 					scripts for running experiments
```

## Setup

**Step 1:** Run the following commands step by step to setup the environments:

```bash
conda create -n CHAIC python=3.9
conda activate CHAIC
pip3 install -e .
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

If you are running in a remote server without a screen, please refer to [running TDW in server](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md).

After that, you can run the demo scene to verify your setup:

```bash
python demo/demo_scene.py
```

**Step 2:** Install and download perception models:

```bash
pip install -U openmim
mim install mmengine
mim install mmdet
pip install mmaction2
bash detection_pipeline/download_ckpt.sh
```

After that, you can run the perception demos to verify them:

```bash
python tdw-gym/detection.py
python tdw-gym/behavior.py
```

**Notice:** There maybe exists some internal bugs in the `mmaction` package, and you can refer to the [Github issue](https://github.com/open-mmlab/mmaction2/issues/2714) to fix it when you meet trouble.

## Run Experiments

We prepare the example scripts to run experiments with HP baseline and our Cooperative LLM Agent under the folder `scripts`. For example, to run experiments with two LLM Agents, run the following command:


```
./scripts/test_LMs.sh
```

## More details on the environment

### Multi-agent Asynchronized Setting

Agents may take different number of frames to finish (or fail) one action, one env step is finished until any agent's action is not ongoing, and the current obs is returned to all agents.
All agents are asked for a new action, and agents having ongoing actions will directly switch to the new action if actions changed. 

### Gym Scenes

The dataset is modular in its design, consisting of several physical floor plan geometries with a wall and floor texture 
variations (e.g. parquet flooring, ceramic tile, stucco, carpet etc.) and various furniture and prop layouts (tables, 
chairs, cabinets etc.), for a total of 6 separate environments. Every scene has 6 to 8 rooms, 10 objects, and 4 containers.

### Gym Observations
```
"rgb": Box(0, 256, (3, 256, 256))
"depth": Box(0, 256, (256, 256))
"seg_mask": (0, 256, (256, 256, 3))
"agent": Box(-30, 30, (6,)) # position
"status": ActionStatus[ongoing, success,fail,...]
"camera_matrix": Box(-30, 30, (4, 4)),
"valid": bool[True / False] # Whether the last action is valid
"held_objects": [{'id': id/id/None, 'type': 0/1/None, 'name': name/name/None',  contained': [None] * 3 / [..] / [None] * 3}, {'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', 'contained': [None] * 3 / [..] / [None] * 3}],
"oppo_held_objects": [{'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', contained': [None] * 3 / [..] / [None] * 3}, {'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', 'contained': [None] * 3 / [..] / [None] * 3}], # we can see it when opponent is visible
"current_frames": int
'visible_objects': [{'id': ..., 'type': ..., 'seg_color': ..., 'name': ...}]
"containment": {"agent_id": {"visible": True / False, "contained": [id, ...]}}
"messages": (Text(max_length=1000, charset=string.printable), Text(max_length=1000, charset=string.printable))
```

### Api the agent can access
```
belongs_to_which_room((x, y, z)) -> room_name: given a location, return the room name;
center_of_room(room_name) -> [x, y, z]: given a room name, return the center of the room;
check_pos_in_room(x, z) -> bool: given a location, return whether it is inside the floorplan.
```

### Gym resetting
reset gym with {scene, layout, task}, and will return:
```
"goal_description": a dict {'target_name_i': num_i}
"room_type": a list [room_name_0, ...]
```

### Gym Actions
* move forward at 0.5
```
dict {"type": 0} 
```
* turn left 15 degrees
```
dict {"type": 1} 
```
* turn right 15 degrees
```
dict {"type": 2} 
```
* grasp the object with arm
```
dict {"type": 3, "object": object_id, "arm": "left" or "right"} 
```
* put the holding object into the holding container
```
dict {"type": 4} 
```
* drop objects
```
dict {"type": 5, "arm": "left" or "right"}
```
* send messages
```
dict {"type": 6, "message": Text}
```
