from .panda import Panda
from .ur5 import ur5

def get_robot(robot_config, bullet_client):
    robot_name = robot_config.pop("name")

    if robot_name == 'panda':
        robot = Panda(bullet_client, **robot_config)
    elif robot_name == 'ur5':
        robot = ur5(bullet_client, **robot_config)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    return robot