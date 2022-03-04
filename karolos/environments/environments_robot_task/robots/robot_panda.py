import numpy as np
import os
from pathlib import Path
import sys

from pathlib import Path

import pybullet as p
import pybullet_data as pd

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Robot, Link


class RobotPanda(Robot):
    def __init__(self, bullet_client=None, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):
        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "Panda/panda.urdf")

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

        dht_params = [
            [None, .333, 0., 0.],
            [None, 0., 0., -np.pi / 2],
            [None, .316, 0., np.pi / 2],
            [None, 0., .0825, np.pi / 2],
            [None, .384, -.0825, -np.pi / 2],
            [None, 0., 0., np.pi / 2],
            [None, 0., .088, np.pi / 2],
            [0., .107, 0., 0.],
        ]

        # dht_params = [(
        #     (None, .333, 0., 0.), (
        #         (None, 0., 0., -np.pi / 2), (
        #             (None, .316, 0., np.pi / 2), (
        #                 (None, 0., .0825, np.pi / 2), (
        #                     (None, .384, -.0825, -np.pi / 2), (
        #                         (None, 0., 0., np.pi / 2), (
        #                             (None, 0., .088, np.pi / 2), (
        #                                 (0., .107, 0., 0.)
        #                             )
        #                         )
        #                     )
        #                 )
        #             )
        #         )
        #     )
        # )]

        self.index_tcp = len(links) - 1

        super(RobotPanda, self).__init__(bullet_client=bullet_client,
                                         urdf_file=urdf_file,
                                         joints_arm=joints_arm,
                                         joints_hand=joints_hand,
                                         links=links,
                                         dht_params=dht_params,
                                         offset=offset,
                                         sim_time=sim_time,
                                         scale=scale,
                                         parameter_distributions=parameter_distributions)