from pathlib import Path
import sys

import numpy as np
import pytest
from gym import spaces
import pybullet as p
import pybullet_data as pd

sys.path.append(str(Path(__file__).parents[1].resolve()))

from karolos.environments import get_env
from itertools import product

# robots = ["ur5", "panda"]
# tasks = ["reach", "pick_place"]
robots = ["panda"]
tasks = ["pick_place"]

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=70,
                             cameraPitch=-27,
                             cameraTargetPosition=(0, 0, 0)
                             )

p.setRealTimeSimulation(0)
p.setGravity(0, 0, -9.81)

for robot, task in product(robots, tasks):
    print(robot, task)

    env_config = {
        "environment": "karolos",
        "bullet_client": p,
        "task_config": {
            "name": task,
            "max_steps": 100
        },
        "robot_config": {
            "name": robot,
            "scale": .1,
            "sim_time": .5
        }
    }

    env = get_env(env_config)

    results = []

    trials = 10

    for _ in range(trials):
        try:
            state, goal_info = env.reset()

            done = False

            while not done:
                try:
                    action = goal_info["expert_action"]
                    state, goal_info, done = env.step(action)
                except ValueError:
                    done = True

                done |= env.success_criterion(goal_info)
                print(goal_info["steps"])

            results.append(env.success_criterion(goal_info))
        except:
            pass

    assert len(results) > .5 * trials, f"Only completed {len(results)} / {trials} runs"
    assert np.mean(results) > .5, f"Only solved {np.mean(results) * 100}%"
