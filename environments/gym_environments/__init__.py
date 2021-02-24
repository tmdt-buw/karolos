def get_gym_env(env_config):

    task_config = env_config.pop("task_config")
    task_name = task_config.pop("name")

    if task_name == "pendulum":
        from environments.gym_environments.pendulum import Pendulum
        env = Pendulum(**task_config, render=env_config.pop("render", False))
    if task_name == "nlinkarm":
        from environments.gym_environments.NLinkArm import NLinkArm
        env = NLinkArm(**task_config)
    else:
        raise ValueError(f"Unknown gym environment: {task_name}")

    return env