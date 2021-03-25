import logging
import os
import time
from collections import namedtuple

from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import spaces
from numpy.random import RandomState


# todo implement domain randomization

class UR5:
    def __init__(self, bullet_client, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for UR5")
            raise NotImplementedError()

        self.time_step = bullet_client.getPhysicsEngineParameters()["fixedTimeStep"]

        if not sim_time:
            sim_time = self.time_step

        self.scale = scale

        self.max_steps = int(sim_time / self.time_step)

        self.offset = offset

        self.bullet_client = bullet_client

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent),
                                 "ur5.urdf")

        self.model_id = bullet_client.loadURDF(urdf_file,
                                            np.array([0, 0, 0]) + self.offset,
                                            useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        self.joint_name2id = {}

        for jj in range(self.bullet_client.getNumJoints(self.model_id)):
            jointInfo = self.bullet_client.getJointInfo(self.model_id, jj)
            self.joint_name2id[jointInfo[1].decode("utf-8")] = jointInfo[0]

            # print(jointInfo[1].decode("utf-8"))

        Joint = namedtuple("Joint",
                           ["id", "initial_position", "limits", "max_velocity",
                            "max_torque"])

        self.joints_arm = {
            "shoulder_pan_joint": Joint(self.joint_name2id["shoulder_pan_joint"], 0, (-6.2831, 6.2831), 3.15, 300),
            "shoulder_lift_joint": Joint(self.joint_name2id["shoulder_lift_joint"], 0, (-2.3561, 2.3561), 3.15, 150),
            "elbow_joint": Joint(self.joint_name2id["elbow_joint"], 0, (-3.1415, 3.1415), 3.15, 150),
            "wrist_1_joint": Joint(self.joint_name2id["wrist_1_joint"], 0, (-2.3561, 2.3561), 3.2, 28),
            "wrist_2_joint": Joint(self.joint_name2id["wrist_2_joint"], 0, (-6.2831, 6.2831), 3.2, 28),
            "wrist_3_joint": Joint(self.joint_name2id["wrist_3_joint"], 0, (-6.2831, 6.2831), 3.2, 28),
        }

        self.joints_hand = {
            # hand
            "left_outer_knuckle_joint": Joint(self.joint_name2id["left_outer_knuckle_joint"], 0.3, (0.0, 0.725), 2., 15),
            "left_inner_knuckle_joint": Joint(self.joint_name2id["left_inner_knuckle_joint"], 0.3, (0.0, 0.8757), 2., 15),
            "right_outer_knuckle_joint": Joint(self.joint_name2id["right_outer_knuckle_joint"], 0.3, (0.0, 0.725), 2., 15),
            "right_inner_knuckle_joint": Joint(self.joint_name2id["right_inner_knuckle_joint"], 0.3, (0.0, 0.8757), 2., 15),
        }

        self.joints_passive = {
            # hand
            "left_outer_finger_joint": Joint(self.joint_name2id["left_outer_finger_joint"], 0.3, (0.0, 0.725), 2., 20),
            "left_inner_finger_joint": Joint(self.joint_name2id["left_inner_finger_joint"], 0.3, (0.0, 0.8757), 2., 20),
            "right_outer_finger_joint": Joint(self.joint_name2id["right_outer_finger_joint"], 0.3, (0.0, 0.725), 2., 20),
            "right_inner_finger_joint": Joint(self.joint_name2id["right_inner_finger_joint"], 0.3, (0.0, 0.8757), 2., 20),
        }

        constraint = self.bullet_client.createConstraint(self.model_id,
                                                         self.joint_name2id[
                                                             "left_inner_finger_loop"],
                                                         self.model_id,
                                                         self.joint_name2id[
                                                             "left_outer_finger_loop"],
                                                         self.bullet_client.JOINT_POINT2POINT,
                                                         [0, 0, 0],
                                                         [0, 0, 0], [0, 0, 0])
        self.bullet_client.changeConstraint(constraint, maxForce=1e4)

        constraint = self.bullet_client.createConstraint(self.model_id,
                                                         self.joint_name2id[
                                                             "right_inner_finger_loop"],
                                                         self.model_id,
                                                         self.joint_name2id[
                                                             "right_outer_finger_loop"],
                                                         self.bullet_client.JOINT_POINT2POINT,
                                                         [0, 0, 0],
                                                         [0, 0, 0], [0, 0, 0])
        self.bullet_client.changeConstraint(constraint, maxForce=1e4)

        from itertools import chain, permutations

        for linkA, linkB in permutations(chain(self.joints_hand.keys(), self.joints_passive.keys()), 2):
            self.bullet_client.setCollisionFilterPair(self.model_id,
                                                      self.model_id,
                                                      self.joint_name2id[
                                                          linkA],
                                                      self.joint_name2id[linkB],
                                                      False)

        self.bullet_client.setCollisionFilterPair(self.model_id, self.model_id,
                                                  self.joint_name2id[
                                                      "left_inner_finger_joint"],
                                                  self.joint_name2id["right_inner_finger_joint"], False)

        # todo introduce friction
        # self.bullet_client.setJointMotorControlArray(self.model_id,
        #                                              [joint.id for _, joint in self.joints_arm.items()],
        #                                              p.VELOCITY_CONTROL,
        #                                              forces=[0.1 * joint.max_torque
        #                                                      for _, joint in self.joints_arm.items()])

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     [joint.id for _, joint in self.joints_hand.items()],
                                                     p.VELOCITY_CONTROL,
                                                     forces=[0 * joint.max_torque
                                                             for _, joint in self.joints_hand.items()])

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     [joint.id for _, joint in
                                                      self.joints_passive.items()],
                                                     p.VELOCITY_CONTROL,
                                                     forces=[
                                                         0 * joint.max_torque
                                                         for _, joint in
                                                         self.joints_passive.items()])

        self.bullet_client.stepSimulation()

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=(len(self.joints_arm) + 1,))

        self.observation_space = spaces.Dict({
            "joint_positions": spaces.Box(-1., 1., shape=(len(self.joints_arm),)),
            "joint_velocities": spaces.Box(-1., 1., shape=(len(self.joints_arm),)),
            "tcp_position": spaces.Box(-1., 1., shape=(3,)),
            # "tcp_velocity": spaces.Box(-1., 1., shape=(3,)),
        })

        # reset to initial position
        self.reset()

    def reset(self, desired_state=None):
        """Reset robot to initial pose and return new state."""

        contact_points = True

        if desired_state is not None:

            desired_state = list(desired_state)

            for _, joint in self.joints_arm.items():
                joint_position = np.interp(desired_state.pop(0), [-1, 1],
                                           joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id,
                                                                 self.model_id)

        # reset until state is valid
        while contact_points:

            for _, joint in self.joints_hand.items():
                joint_position = self.random.uniform(*joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id,
                                                                 self.model_id)

        observation = self.get_observation()

        return observation

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action}"

        action_arm = action[:6]
        action_hand = action[-1]

        action_arm = list(action_arm * self.scale) # / self.max_steps)

        joint_ids = []
        target_arm_positions = []
        maxVelocities = []
        torques_arm = []

        for (_, joint), action_joint in zip(self.joints_arm.items(), action_arm):

            position, _, _, _ = self.bullet_client.getJointState(self.model_id, joint.id)

            normalized_joint_position = np.interp(position, joint.limits, [-1, 1])
            normalized_target_joint_position = np.clip(normalized_joint_position + action_joint, -1, 1)
            target_joint_position = np.interp(normalized_target_joint_position, [-1, 1], joint.limits)

            joint_ids.append(joint.id)
            target_arm_positions.append(target_joint_position)
            torques_arm.append(joint.max_torque)

            maxVelocities.append(joint.max_velocity)
            # if joint.id == 2:
            #     print(position, target_joint_position)

            # self.bullet_client.setJointMotorControl2(self.model_id,
            #                                              joint.id,
            #                                              p.POSITION_CONTROL,
            #                                              targetPosition=target_joint_position,
            #                                              maxVelocity=joint.max_velocity,
            #                                              force=joint.max_torque
            #                                              )


        print(target_arm_positions)

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     joint_ids,
                                                     p.POSITION_CONTROL,
                                                     targetPositions=target_arm_positions,
                                                     # maxVelocities=maxVelocities,
                                                     forces=torques_arm
                                                     )
        # joint_ids = []
        # target_hand_positions = []
        # torques_hand = []
        #
        # for _, joint in self.joints_hand.items():
        #
        #     position, _, _, _ = self.bullet_client.getJointState(self.model_id, joint.id)
        #
        #     normalized_joint_position = np.interp(position, joint.limits, [-1, 1])
        #     normalized_target_joint_position = np.clip(normalized_joint_position + action_hand, -1, 1)
        #     target_joint_position = np.interp(normalized_target_joint_position, [-1, 1], joint.limits)
        #
        #     joint_ids.append(joint.id)
        #     target_hand_positions.append(target_joint_position)
        #     torques_hand.append(joint.max_torque)
        #
        # self.bullet_client.setJointMotorControlArray(self.model_id,
        #                                              joint_ids,
        #                                              p.POSITION_CONTROL,
        #                                              targetPositions=target_hand_positions,
        #                                              forces=torques_hand
        #                                              )


        joint_ids = [joint.id for _, joint in self.joints_hand.items()]
        torques_fingers = [action_hand * joint.max_torque for _, joint in self.joints_hand.items()]

        for step in range(self.max_steps):
            self.bullet_client.setJointMotorControlArray(self.model_id,
                                                         joint_ids,
                                                         p.TORQUE_CONTROL,
                                                         forces=torques_fingers)

            self.bullet_client.stepSimulation()

            if self.bullet_client.getConnectionInfo()["connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

        observation = self.get_observation()

        return observation

    def get_observation(self):
        joint_positions, joint_velocities = [], []

        for _, joint in self.joints_arm.items():
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(
                self.model_id, joint.id)

            joint_positions.append(np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(
                np.interp(joint_velocity, [-joint.max_velocity, joint.max_velocity],
                          [-1, 1]))

        tcp_position, _, _, _, _, _, tcp_velocity, _ = \
            self.bullet_client.getLinkState(self.model_id, self.joint_name2id["tcp"],
                                            computeLinkVelocity=True)

        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)
        tcp_position = np.array(tcp_position)
        # tcp_velocity = np.array(tcp_velocity)

        observation = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "tcp_position": tcp_position,
            # "tcp_velocity": tcp_velocity
        }

        for key in observation:
            observation[key] = observation[key].clip(
                self.observation_space[key].low,
                self.observation_space[key].high)

        return observation


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

    robot = UR5(p, sim_time=.1, scale=.02)

    cube = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                     halfExtents=[.025] * 3,
                                                                 rgbaColor=[1,
                                                                            0,
                                                                            0,
                                                                            1],
                                                                 ),

            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,
                                                           halfExtents=[.025] * 3,
                                                           ),
            baseMass=.1,
        )

    while True:
        # observation = robot.reset(np.zeros_like(robot.observation_space["joint_positions"].sample()))

        action = np.zeros_like(robot.action_space.sample())
        action[-1] = -1.

        for _ in range(1):
            observation = robot.step(action)

        p.resetBasePositionAndOrientation(
            cube, observation["tcp_position"], [0, 0, 0, 1])

        for _ in range(25):
            action = robot.action_space.sample()  # * .1
            action[-1] = 1.

            observation = robot.step(action)

        p.resetBasePositionAndOrientation(
            cube, observation["tcp_position"], [0, 0, 0, 1])
