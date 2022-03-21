import sys
from pathlib import Path

import pytest
from gym import spaces

sys.path.append(str(Path(__file__).parents[1].resolve()))

from ..agents.utils import unwind_space_shapes, unwind_dict_values
from ..agents import get_agent

discrete_agents = ["dqn"]
continuous_agents = ["sac", "ddpg", "ppo"]

agents = discrete_agents + continuous_agents

action_space_discrete = spaces.Discrete(6)
action_space_continuous = spaces.Box(-1, 1, (6,))

state_space = spaces.Dict({
    'state': spaces.Box(-1, 1, shape=(100,)),
})

goal_space = spaces.Dict({
    'achieved': spaces.Box(-1, 1, shape=(10,)),
    'desired': spaces.Box(-1, 1, shape=(10,)),
})

state_spaces = unwind_space_shapes(state_space)


def dummy_state(state_space, action_space):
    state = {}
    goal = {}

    for space_name, space in state_space.spaces.items():
        state[space_name] = space.sample()

    for space_name, space in goal_space.spaces.items():
        goal[space_name] = space.sample()

    info = {
        "expert_action": dummy_action(action_space)
    }

    return state, goal, info


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
        agent = get_agent({"name": agent}, state_space, goal_space, action_space_discrete)
    elif agent in continuous_agents:
        agent = get_agent({"name": agent}, state_space, goal_space, action_space_continuous)
    else:
        raise ValueError("Unknown action space type", agent)

    trajectory = get_dummy_trajectory(state_space, agent.action_space)

    agent.add_experience_trajectory(trajectory)

    agent.learn()

    states, goals = [], []

    for _ in range(10):
        state, goal, info = dummy_state(state_space, agent.action_space)
        states.append(unwind_dict_values(state))
        goals.append(unwind_dict_values(goal))

    actions = agent.predict(states, goals, deterministic=False)

    assert len(states) == len(actions)
