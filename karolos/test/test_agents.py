import pytest

import pathlib
import sys

from gym import spaces

sys.path.append(str(pathlib.Path(__file__).parents[1].resolve()))

from ..agents.utils import unwind_space_shapes, unwind_dict_values
from ..agents import get_agent

discrete_agents = ["dqn"]
continuous_agents = ["sac", "ddpg", "ppo"]

agents = discrete_agents + continuous_agents

action_space_discrete = spaces.Discrete(6)
action_space_continuous = spaces.Box(-1,1,(6,))

state_space = spaces.Dict({
    'state': spaces.Box(-1, 1, shape=(100,)),
})

state_spaces = unwind_space_shapes(state_space)


def dummy_state(state_space, action_space):
    state = {}
    goal_info = {
        "expert_action": dummy_action(action_space)
    }

    for space_name, space in state_space.spaces.items():
        state[space_name] = space.sample()

    return state, goal_info


def dummy_action(action_space):
    return action_space.sample()

def get_dummy_trajectory(state_space, action_space):

    trajectory = []

    for _ in range(50):
        trajectory.append(dummy_state(state_space, action_space))
        trajectory.append(dummy_action(action_space))

    trajectory.append(dummy_state(state_space, action_space))

    return trajectory

@pytest.mark.parametrize("agent", agents)
def test_agent(agent):

    if agent in discrete_agents:
        agent = get_agent({"name": agent}, state_space, action_space_discrete)
    elif agent in continuous_agents:
        agent = get_agent({"name": agent}, state_space, action_space_continuous)
    else:
        raise ValueError("Unknown action space type", agent)

    trajectory = get_dummy_trajectory(state_space, agent.action_space)

    agent.add_experience_trajectory(trajectory)

    agent.learn()

    states = [unwind_dict_values(dummy_state(state_space, agent.action_space)[0]) for _ in range(10)]

    actions = agent.predict(states, deterministic=False)

    assert len(states) == len(actions)
