import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Robot


# todo implement domain randomization

class RobotIIWA(Robot):
    def __init__(self, bullet_client=None, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):
        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for iiwa")
            raise NotImplementedError()

        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "iiwa/iiwa.urdf")

        joints_arm = {
            "lbr_iiwa_joint_1": (0, (-2.96705972839, 2.96705972839), 1.71042266695, 320),
            "lbr_iiwa_joint_2": (0, (-2.09439510239, 2.09439510239), 1.71042266695, 320),
            "lbr_iiwa_joint_3": (0, (-2.96705972839, 2.96705972839), 1.74532925199, 176),
            "lbr_iiwa_joint_4": (0, (-2.09439510239, 2.09439510239), 2.26892802759, 176),
            "lbr_iiwa_joint_5": (0, (-2.96705972839, 2.96705972839), 2.44346095279, 110),
            "lbr_iiwa_joint_6": (0, (-2.09439510239, 2.09439510239), 3.14159265359, 40),
            "lbr_iiwa_joint_7": (0, (-3.05432619099, 3.05432619099), 3.14159265359, 40),
        }

        joints_hand = {
            # hand
            "left_inner_finger_joint": (0.3, (0., .0425), 2., 20),
            "right_inner_finger_joint": (0.3, (0., .0425), 2., 20),
        }

        self.index_tcp = 10

        super(RobotIIWA, self).__init__(bullet_client=bullet_client,
                                        urdf_file=urdf_file,
                                        joints_arm=joints_arm,
                                        joints_hand=joints_hand,
                                        offset=offset,
                                        sim_time=sim_time,
                                        scale=scale,
                                        parameter_distributions=parameter_distributions)

        # todo introduce friction
