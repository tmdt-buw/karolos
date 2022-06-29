def get_env(env_config):

    environment_name = env_config.pop("name", "robot-task")

    if environment_name == "robot-task":
        from karolos.environments.environment_robot_task import EnvironmentRobotTask
        env = EnvironmentRobotTask(**env_config)
    elif environment_name == "gym":
        from karolos.environments.environment_gym_wrapper import GymWrapper
        env = GymWrapper(**env_config)
    else:
        raise ValueError(f"Unknown environment: {environment_name}")

    env.name = environment_name

    return env
