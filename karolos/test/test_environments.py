import pytest
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1].resolve()))

from ..environments import get_env

robots = ["panda", "ur5"]
tasks = ["reach", "pick_place"]

gym_environment_names = ["Pendulum-v0"]

@pytest.mark.parametrize("robot", robots)
@pytest.mark.parametrize("task", tasks)
def test_environment_robot_task(robot, task):
    max_steps = 50

    env = get_env({
        "environment": "karolos",

        "task_config": {
            "name": task,
            "max_steps": 50
        },
        "robot_config": {
            "name": robot,
        }
    })

    env.reset()

    step = 0

    done = False

    while not done:
        state, goal, done, info = env.step(env.action_space.sample())

        step += 1

    assert step <= max_steps

@pytest.mark.parametrize("gym_environment_name", gym_environment_names)
def test_environment_gym_wrapper(gym_environment_name):
    max_steps = 100

    env = get_env({
        "environment": "gym",
        "name": gym_environment_name,
        "max_steps": max_steps
    })

    state, goal_info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # * .0
        next_state, goal_info, done = env.step(action)

        done |= env.success_criterion(goal_info)

    assert goal_info["achieved"]["step"] <= max_steps
