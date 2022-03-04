import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Robot


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

        dht_params = [
            [None, 0, .15185, np.pi / 2],
            [None, -.24355, 0, 0],
            [None, -.2132, 0, 0],
            [None, 0, .13105, np.pi / 2],
            [None, 0, .08535, -np.pi / 2],
            [None, 0, .0921, 0]
        ]

        self.index_tcp = 10

        super(RobotUR5, self).__init__(bullet_client=bullet_client,
                                       urdf_file=urdf_file,
                                       joints_arm=joints_arm,
                                       joints_hand=joints_hand,
                                       dht_params=dht_params,
                                       offset=offset,
                                       sim_time=sim_time,
                                       scale=scale,
                                       parameter_distributions=parameter_distributions)

        # todo introduce friction

