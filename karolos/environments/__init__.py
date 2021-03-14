class Environment:
    def reset(self, desired_state=None):
        """
        Returns:
            state, goal
        """
        raise NotImplementedError()

    def step(self, action):
        """
        Returns:
            state, goal, done
        """
        raise NotImplementedError()


def get_env(env_config):

    environment = env_config.pop("environment")

    if environment == "karolos":
        from environments.robot_task_environments.environment_robot_task import RobotTaskEnvironment
        env = RobotTaskEnvironment(**env_config)
    elif environment == "gym":
        from environments.environment_wrappers.gym_wrapper import GymWrapper
        env = GymWrapper(**env_config)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    return env
