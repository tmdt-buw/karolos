import pytest

import pathlib
import sys

from gym import spaces

sys.path.append(str(pathlib.Path(__file__).parents[1].resolve()))

from utils import unwind_space_shapes, unwind_dict_values
from ..agents import get_agent

discrete_algorithms = ["dqn"]
continuous_algorithms = ["sac", "ddpg", "ppo"]

algorithms = discrete_algorithms + continuous_algorithms

action_space_discrete = spaces.Discrete(6)
action_space_continuous = spaces.Box(-1,1,(6,))

observation_space = spaces.Dict({
    'state': spaces.Box(-1, 1, shape=(100,)),
})

state_spaces = unwind_space_shapes(observation_space)


def dummy_state(observation_space):
    state = {}
    goal_info = {}

    for space_name, space in observation_space.spaces.items():
        state[space_name] = space.sample()

    return state, goal_info


def dummy_action(action_space):
    return action_space.sample()

def get_dummy_trajectory(observation_space, action_space):

    trajectory = []

    for _ in range(50):
        trajectory.append(dummy_state(observation_space))
        trajectory.append(dummy_action(action_space))

    trajectory.append(dummy_state(observation_space))

    return trajectory

@pytest.mark.parametrize("algorithm", algorithms)
def test_algorithm(algorithm):

    if algorithm in discrete_algorithms:
        agent = get_agent({"algorithm": algorithm}, observation_space, action_space_discrete)
    elif algorithm in continuous_algorithms:
        agent = get_agent({"algorithm": algorithm}, observation_space, action_space_continuous)
    else:
        raise ValueError("Unknown action space type", algorithm)

    trajectory = get_dummy_trajectory(observation_space, agent.action_space)

    agent.add_experience_trajectory(trajectory)

    agent.learn()

    states = [unwind_dict_values(dummy_state(observation_space)[0]) for _ in range(10)]

    actions = agent.predict(states, deterministic=False)

    assert len(states) == len(actions)
