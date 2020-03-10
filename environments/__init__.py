def get_env(env_config):
    base_pkg = env_config.pop("base_pkg")

    assert base_pkg in ["robot-task-rl"]

    if base_pkg == "robot-task-rl":
        from environments.environment_robot_task import Environment

        def env_init():
            env = Environment(**env_config)
            return env
    else:
        raise NotImplementedError(f"Unknown base package: {base_pkg}")

    return env_init
