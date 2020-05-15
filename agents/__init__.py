def get_agent(algorithm, agent_config, observation_space, action_space):

    if algorithm == "sac":
        from agents.sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent
