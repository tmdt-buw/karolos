import numpy as np
import os
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Robot, Link


class RobotPanda(Robot):
    def __init__(self, bullet_client, **kwargs):
        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent), "Panda/panda.urdf")

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
            {"d": .333, "a": 0., "alpha": 0., "proximal": True},
            {"d": 0., "a": 0., "alpha": -np.pi / 2, "proximal": True},
            {"d": .316, "a": 0., "alpha": np.pi / 2, "proximal": True},
            {"d": 0., "a": .0825, "alpha": np.pi / 2, "proximal": True},
            {"d": .384, "a": -.0825, "alpha": -np.pi / 2, "proximal": True},
            {"d": 0., "a": 0., "alpha": np.pi / 2, "proximal": True},
            {"d": 0., "a": .088, "alpha": np.pi / 2, "proximal": True},
            {"theta": 0., "d": .107, "a": 0., "alpha": 0., "proximal": True},
            {"theta": 0., "d": -.105, "a": 0., "alpha": np.pi, "proximal": True},
        ]

        self.index_tcp = len(links) - 1

        super(RobotPanda, self).__init__(
            urdf_file=urdf_file,
            joints_arm=joints_arm,
            joints_hand=joints_hand,
            links=links,
            dht_params=dht_params,
            bullet_client=bullet_client,
            **kwargs
        )
