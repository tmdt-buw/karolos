def get_agent(agent_config, observation_space, action_space):
    algorithm = agent_config.pop("algorithm")

    if algorithm == "sac":
        from agents.sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent