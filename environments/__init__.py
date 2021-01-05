def get_env(env_config):

    environment = env_config.pop("environment")

    if environment == "karolos":
        from environments.environment_robot_task import Environment
        env = Environment(**env_config)
    else:
        import gym
        from environments.gym_wrapper import GymWrapper
        env = GymWrapper(gym.make(environment), **env_config)

    return env
