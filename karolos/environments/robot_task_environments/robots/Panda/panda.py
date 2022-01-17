import os
from pathlib import Path

import pybullet as p
import pybullet_data as pd

try:
    from .. import RobotArm, Joint, Link
except:
    from karolos.environments.robot_task_environments.robots import RobotArm, Joint, Link


class Panda(RobotArm):
    def __init__(self, bullet_client, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):
        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "panda.urdf")

        joints_arm = {
            "panda_joint1": (0, (-2.8973, 2.8973), 2.1750, 87),
            "panda_joint2": (0.5, (-1.7628, 1.7628), 2.1750, 87),
            "panda_joint3": (0, (-2.8973, 2.8973), 2.1750, 87),
            "panda_joint4": (-0.5, (-3.0718, -0.0698), 2.1750, 87),
            "panda_joint5": (0, (-2.8973, 2.8973), 2.6100, 12),
            "panda_joint6": (1., (-0.0175, 3.7525), 2.6100, 12),
            "panda_joint7": (0.707, (-2.8973, 2.8973), 2.6100, 12),
        }

        joints_hand = {
            "panda_finger_joint1": (0.035, (0.0, 0.04), 0.05, 20),
            "panda_finger_joint2": (0.035, (0.0, 0.04), 0.05, 20),
        }

        links = {
            "panda_link1": Link(2.7, 0.01),
            "panda_link2": Link(2.73, 0.01),
            "panda_link3": Link(2.04, 0.01),
            "panda_link4": Link(2.08, 0.01),
            "panda_link5": Link(3.0, 0.01),
            "panda_link6": Link(1.3, 0.01),
            "panda_link7": Link(0.2, 0.01),
            "panda_hand": Link(0.81, 0.01),
            "panda_leftfinger": Link(0.1, 0.01),
            "panda_rightfinger": Link(0.1, 0.01),
            "tcp": Link(0.0, 0.01),
        }

        self.index_tcp = len(links) - 1

        super(Panda, self).__init__(bullet_client=bullet_client,
                                    urdf_file=urdf_file,
                                    joints_arm=joints_arm,
                                    joints_hand=joints_hand,
                                    links=links,
                                    offset=offset,
                                    sim_time=sim_time,
                                    scale=scale,
                                    parameter_distributions=parameter_distributions)


if __name__ == "__main__":
    import numpy as np

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    p.setRealTimeSimulation(0)

    p.setGravity(0, 0, -9.81)

    robot = Panda(p, sim_time=.1, scale=.1)

    while True:
        observation = robot.reset()

        action = -np.ones_like(robot.action_space.sample())

        for _ in range(25):
            observation = robot.step(action)

        for _ in range(25):
            observation = robot.step(-action)
