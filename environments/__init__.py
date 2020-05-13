from environments.environment_robot_task import Environment


def get_env(env_config):
    def env_init():
        env = Environment(**env_config)
        return env

    return env_init
