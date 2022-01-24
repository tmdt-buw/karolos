import pathlib
import sys

import numpy as np
import pytest
from gym import spaces

sys.path.append(str(pathlib.Path(__file__).parents[1].resolve()))

from utils import unwind_space_shapes
from ..environments import get_env

robots = ["ur5", "panda"]
tasks = ["reach", "pick_place"]

action_space_discrete = spaces.Discrete(6)
action_space_continuous = spaces.Box(-1, 1, (6,))

state_space = spaces.Dict({
    'state': spaces.Box(-1, 1, shape=(100,)),
})

state_spaces = unwind_space_shapes(state_space)


@pytest.mark.parametrize("robot", robots)
@pytest.mark.parametrize('task', tasks)
def test_expert(robot, task):
    env_config = {
        "environment": "karolos",
        "task_config": {
            "name": task,
            "max_steps": 150
        },
        "robot_config": {
            "name": robot,
            "scale": .1,
            "sim_time": .1
        }
    }

    env = get_env(env_config)

    results = []

    for _ in range(10):
        state, goal_info = env.reset()

        done = False

        while not done:
            try:
                action = goal_info["expert_action"]
                state, goal_info, done = env.step(action)
            except ValueError:
                done = True

            done |= env.success_criterion(goal_info)

        results.append(env.success_criterion(goal_info))

    assert np.mean(results) > .8, f"Only solved {np.mean(results) * 100}%"
