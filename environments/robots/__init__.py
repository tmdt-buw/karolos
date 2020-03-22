from .panda import Panda
from .panda_relative import PandaRelative

def get_robot(robot_config, bullet_client):
    robot_name = robot_config.pop("name")

    if robot_name == 'pandas':
        robot = Panda(bullet_client, **robot_config)
    elif robot_name == 'pandas_relative':
        robot = PandaRelative(bullet_client, **robot_config)
    else:
        raise ValueError()

    return robot