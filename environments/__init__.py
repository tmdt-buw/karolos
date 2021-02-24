def get_env(env_config):

    environment = env_config.pop("environment")

    if environment == "karolos":
        from environments.environment_robot_task import Environment
        env = Environment(**env_config)
    elif environment == "gym":
        from environments.gym_environments import get_gym_env
        env = get_gym_env(env_config)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    return env
