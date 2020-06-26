def get_agent(algorithm, agent_config, observation_space, action_space,
              experiment_dir):
    if algorithm == "sac":
        from agents.sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space,
                         experiment_dir)
    elif algorithm == "ddpg":
        from agents.ddpg import AgentDDPG
        agent = AgentDDPG(agent_config, observation_space, action_space,
                         experiment_dir)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent
