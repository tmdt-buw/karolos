def get_robot(robot_config, bullet_client=None):
    """
        Called upon initialization to get and create an instance of a robot
        :param robot_config: config of the robot
        :param bullet_client: pybullet client
        :return: a specified robots instance
        """

    robot_name = robot_config.pop("name")

    if robot_name == 'panda':
        from .robot_panda import RobotPanda
        robot = RobotPanda(bullet_client, **robot_config)
    elif robot_name == 'ur5':
        from .robot_ur5 import RobotUR5
        robot = RobotUR5(bullet_client, **robot_config)
    elif robot_name == 'iiwa':
        from .robot_iiwa import RobotIIWA
        robot = RobotIIWA(bullet_client, **robot_config)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    return robot
