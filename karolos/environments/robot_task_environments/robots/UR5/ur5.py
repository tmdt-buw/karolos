import logging
import os
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd

try:
    from .. import RobotArm, Joint, KeyPointMode
except:
    from karolos.environments.robot_task_environments.robots import RobotArm, Joint, KeyPointMode


# todo implement domain randomization

class UR5(RobotArm):
    def __init__(self, bullet_client, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):
        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for UR5")
            raise NotImplementedError()

        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "ur5.urdf")

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
            "left_inner_finger_joint": (0.3, (-.0425, 0.), 2., 20),
            "right_inner_finger_joint": (0.3, (-.0425, 0.), 2., 20),
        }



        super(UR5, self).__init__(bullet_client=bullet_client,
                                  urdf_file=urdf_file,
                                  joints_arm=joints_arm,
                                  joints_hand=joints_hand,
                                  offset=offset,
                                  sim_time=sim_time,
                                  scale=scale,
                                  parameter_distributions=parameter_distributions,
                                  key_point_mode=KeyPointMode.WORLD_LINK_FRAME)

        # todo introduce friction


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    # p.setTimeStep(1. / 300.)

    p.setRealTimeSimulation(0)

    p.setGravity(0, 0, -9.81)

    robot = UR5(p, sim_time=.1, scale=.05)

    kp_arm, kp_hand = robot.get_key_points()

    for pos, ori in kp_arm:
        kp = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                     # halfExtents=[1,1,.001],
                                                     halfExtents=[.5, .005,
                                                                  .005],
                                                     rgbaColor=[1, 0, 0, 1],
                                                     ),
            baseMass=0,
        )

        p.resetBasePositionAndOrientation(kp, pos, [0, 0, 0, 1])

    while True:
        robot.bullet_client.stepSimulation()

    exit()

    cube = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                     halfExtents=[.02] * 3,
                                                                 rgbaColor=[1,
                                                                            0,
                                                                            0,
                                                                            1],
                                                                 ),

            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,
                                                           halfExtents=[.01] * 3,
                                                           ),
            baseMass=.1,
        )

    observation = None

    while True:
        # observation = robot.reset(np.zeros_like(robot.observation_space["joint_positions"].sample()))

        action = np.zeros_like(robot.action_space.sample())
        action[-1] = -1.

        for _ in range(5):
            observation = robot.step(action)

        p.resetBasePositionAndOrientation(
            cube, observation["tcp_position"], [0, 0, 0, 1])

        action[-1] = .7

        for _ in range(5):
            observation = robot.step(action)

        for _ in range(25):
            action = robot.action_space.sample() * .5
            action[-1] = .7

            observation = robot.step(action)

        p.resetBasePositionAndOrientation(
            cube, observation["tcp_position"], [0, 0, 0, 1])
