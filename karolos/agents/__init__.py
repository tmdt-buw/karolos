def get_agent(agent_config, observation_space, action_space,
              reward_function=None, experiment_dir=None):
    agent_name = agent_config.pop("name")

    if agent_name == "sac":
        from .agent_sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space, reward_function, experiment_dir)
    elif agent_name == "ddpg":
        from .agent_ddpg import AgentDDPG
        agent = AgentDDPG(agent_config, observation_space, action_space, reward_function, experiment_dir)
    elif agent_name == "dqn":
        from .agent_dqn import AgentDQN
        agent = AgentDQN(agent_config, observation_space, action_space, reward_function, experiment_dir)
    elif agent_name == "ppo":
        from .agent_ppo_WIP import AgentPPO
        agent = AgentPPO(agent_config, observation_space, action_space, reward_function, experiment_dir)
    else:
        raise NotImplementedError(f"Unknown agent {agent_name}")

    return agent
