def get_agent(algorithm, agent_config, observation_space, action_space, exp_dir):

    if algorithm == "sac":
        from agents.sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space, exp_dir)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent
