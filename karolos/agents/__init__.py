import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from replay_buffers import get_replay_buffer


def get_agent(agent_config, observation_space, action_space,
              reward_function=None, experiment_dir=None):
    algorithm = agent_config.pop("algorithm")

    if algorithm == "sac":
        from .sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space, reward_function, experiment_dir)
    elif algorithm == "ddpg":
        from .ddpg import AgentDDPG
        agent = AgentDDPG(agent_config, observation_space, action_space, reward_function, experiment_dir)
    elif algorithm == "dqn":
        from .dqn import AgentDQN
        agent = AgentDQN(agent_config, observation_space, action_space, reward_function, experiment_dir)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent
