# todo derive suitable tests
import time
import numpy as np

from karolos.environments.environments_robot_task.tasks import get_task
import pybullet as p
import pybullet_data as pd

tasks = ["reach", "pick_place"]

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=70,
                             cameraPitch=-27,
                             cameraTargetPosition=(0, 0, 0)
                             )

p.setRealTimeSimulation(0)
p.setGravity(0, 0, -9.81)

for task_name in tasks:
    print(task_name)

    task = get_task({
        "name": task_name,
        "max_steps": 10,
    }, bullet_client=p)

    for _ in range(5):
        state, goal_info = task.reset()

        done = False

        while not done:
            state, goal_info, done = task.step()
            time.sleep(.1)

    del task

p.disconnect()