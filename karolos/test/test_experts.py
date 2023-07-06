from pathlib import Path
import sys

import numpy as np
import pytest
from gym import spaces

sys.path.append(str(Path(__file__).parents[1].resolve()))

from ..utils import unwind_space_shapes
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
        "name": "robot-task",
        "task_config": {
            "name": task,
            "max_steps": 150
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

            results.append(env.success_criterion(goal_info))
        except:
            pass

    assert len(results) > .5 * trials, f"Only completed {len(results)} / {trials} runs"
    assert np.mean(results) > .5, f"Only solved {np.mean(results) * 100}%"
