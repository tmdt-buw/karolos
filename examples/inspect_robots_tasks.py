# todo derive suitable tests
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data as pd

sys.path.append(str(Path(__file__).parents[1].resolve()))

from karolos.environments import get_env

robots = [
    ("panda", [0, 0, 0, 0, 0, 0, .5, 0, 0]),
    ("ur5", [0.25, 0, 0.25, 0.25, -0.25, 0, 0, 0]),
    ("iiwa", [0, 0, 0, -0.7, 0, 0.5, 0, 0, 0])
]

tasks = [("reach", [.5, -.5, .2]), ("pick_place", [.5, -.5, .2, .5, .5, .4]), ("throw", [-.5, 0, 0])]

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=70,
                             cameraPitch=-27,
                             cameraTargetPosition=(0, 0, 0)
                             )
p.stepSimulation()

p.setRealTimeSimulation(0)
p.setGravity(0, 0, -9.81)

for robot, desired_state_robot in robots:
    for task, desired_state_task in tasks:
        print(f"Display: {robot} & {task}")

        env = get_env({
            "environment": "karolos",
            "bullet_client": p,

            "task_config": {
                "name": task,
                "max_steps": 50
            },
            "robot_config": {
                "name": robot,
            }
        })

        env.reset({"robot": desired_state_robot, "task": desired_state_task})

        input("Press enter to continue")

        del env

p.disconnect()

plt.show()
