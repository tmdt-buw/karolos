import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Robot, Link


# todo implement domain randomization

class RobotUR5(Robot):
    def __init__(self, bullet_client=None, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):
        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for UR5")
            raise NotImplementedError()

        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "UR5/ur5.urdf")

        joints_arm = {
            "shoulder_pan_joint": (0, (-6.2831, 6.2831), 3.15, 300),
            "shoulder_lift_joint": (0, (-2.3561, 2.3561), 3.15, 150),
            "elbow_joint": (0, (-3.1415, 3.1415), 3.15, 150),
            "wrist_1_joint": (0, (-2.3561, 2.3561), 3.2, 28),
            "wrist_2_joint": (0, (-6.2831, 6.2831), 3.2, 28),
            "wrist_3_joint": (0, (-6.2831, 6.2831), 3.2, 28),
        }

        joints_hand = {
            # hand
            "left_inner_finger_joint": (0.3, (0., .0425), 2., 20),
            "right_inner_finger_joint": (0.3, (0., .0425), 2., 20),
        }

        links = {
            "shoulder_link": Link(3.7, 0.01),
            "upper_arm_link": Link(8.393, 0.01),
            "forearm_link": Link(2.275, 0.01),
            "wrist_1_link": Link(1.219, 0.01),
            "wrist_2_link": Link(1.219, 0.01),
            "wrist_3_link": Link(0.1879, 0.01),

            "robotiq_85_base_link": Link(0.1879, 0.01),
            "left_inner_finger": Link(0.1879, 0.01),
            "right_inner_finger": Link(0.1879, 0.01),

            "tcp": Link(0.0, 0.01),
        }

        dht_params = [
            {"d": .089159, "a": 0., "alpha": np.pi / 2.},
            {"d": 0., "a": -.425, "alpha": 0.},
            {"d": 0., "a": -.39225, "alpha": 0.},
            {"d": .10915, "a": 0., "alpha": np.pi / 2.},
            {"d": .09465, "a": 0., "alpha": -np.pi / 2.},
            {"d": .0823, "a": 0., "alpha": 0},
            {"theta": np.pi / 2, "d": .15, "a": 0., "alpha": np.pi},

            #{"theta": 0., "d": -.105, "a": 0., "alpha": np.pi},
        ]

        super(RobotUR5, self).__init__(bullet_client=bullet_client,
                                       urdf_file=urdf_file,
                                       joints_arm=joints_arm,
                                       joints_hand=joints_hand,
                                       links=links,
                                       dht_params=dht_params,
                                       offset=offset,
                                       sim_time=sim_time,
                                       scale=scale,
                                       parameter_distributions=parameter_distributions)
