
def get_robot(robot_config, bullet_client):
    robot_name = robot_config.pop("name")

    if robot_name == 'panda':
        from .Panda.panda import Panda
        robot = Panda(bullet_client, **robot_config)
    elif robot_name == 'ur5':
        from .UR5.ur5 import UR5
        robot = UR5(bullet_client, **robot_config)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    return robot