# todo derive suitable tests
import time
import numpy as np

from karolos.environments.environments_robot_task.robots import get_robot
import pybullet as p
import pybullet_data as pd

robots = ["panda", "ur5", "iiwa"]

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=70,
                             cameraPitch=-27,
                             cameraTargetPosition=(0, 0, 0)
                             )

p.setRealTimeSimulation(0)
p.setGravity(0, 0, -9.81)

for robot_name in robots:
    print(f"Display robot: {robot_name}")

    robot = get_robot({
        "name": robot_name,
        "scale": .1,
        "sim_time": .1
    }, bullet_client=p)

    state = robot.reset()

    action = {k: np.ones_like(v) for k,v in robot.action_space.sample().items()}

    print("Perform max actions")
    for _ in range(25):
        state = robot.step(action)

    print("Perform min actions")
    for _ in range(25):
        state = robot.step(-action)

    state = robot.reset()

    print("Perform random actions")
    for _ in range(25):
        robot.step(robot.action_space.sample())

    del robot

p.disconnect()