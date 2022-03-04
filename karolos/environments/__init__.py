def get_env(env_config):

    environment = env_config.pop("environment")

    if environment == "karolos":
        from karolos.environments.environment_robot_task import EnvironmentRobotTask
        env = EnvironmentRobotTask(**env_config)
    elif environment == "gym":
        from karolos.environments.environment_gym_wrapper import GymWrapper
        env = GymWrapper(**env_config)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    return env
