def get_robot(robot_config, bullet_client):
    robot_name = robot_config.pop("name")

    if robot_name == 'panda':
        from .Panda.panda import Panda
        robot = Panda(bullet_client, **robot_config)
    elif robot_name == 'ur5':
        from .UR5.ur5 import UR5
        robot = UR5(bullet_client, **robot_config)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    return robot


import logging
import os
import time
from collections import namedtuple

import numpy as np
import pybullet as p
from gym import spaces
from numpy.random import RandomState
from enum import Enum

Joint = namedtuple("Joint", ["id", "initial_position", "limits",
                             "max_velocity", "max_torque"])
Link = namedtuple("Link", ["mass", "linearDamping"])

class KeyPointMode(Enum):
    LINK_WORLD = 0
    WORLD_LINK_FRAME = 1

class RobotArm:
    def __init__(self, bullet_client, urdf_file,
                 joints_arm, joints_hand, links=None,
                 offset=(0, 0, 0), sim_time=0.,
                 scale=1., parameter_distributions=None,
                 key_point_mode=KeyPointMode.LINK_WORLD):

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        if parameter_distributions is None:
            parameter_distributions = {}
        self.parameter_distributions = parameter_distributions

        self.time_step = bullet_client.getPhysicsEngineParameters()[
            "fixedTimeStep"]

        if not sim_time:
            sim_time = self.time_step

        if sim_time < self.time_step:
            self.logger.warning(
                "time step of robot is smaller than time step of simulation. This might lead to unintended behavior.")

        self.scale = scale

        self.max_steps = int(sim_time / self.time_step)

        self.offset = np.array(offset)

        self.bullet_client = bullet_client

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

        self.model_id = bullet_client.loadURDF(urdf_file, self.offset,
                                               useFixedBase=True,
                                               flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        self.joint_name2id = {}

        for jj in range(self.bullet_client.getNumJoints(self.model_id)):
            jointInfo = self.bullet_client.getJointInfo(self.model_id, jj)
            self.joint_name2id[jointInfo[1].decode("utf-8")] = jointInfo[0]

            # print(jointInfo[1].decode("utf-8"))

        self.joints_arm = {}
        self.joints_hand = {}

        for joint_name, joint_args in joints_arm.items():
            self.joints_arm[joint_name] = Joint(self.joint_name2id[joint_name],
                                                *joint_args)

        for joint_name, joint_args in joints_hand.items():
            self.joints_hand[joint_name] = Joint(
                self.joint_name2id[joint_name], *joint_args)

        if links is None:
            self.links = {}
        else:
            self.links = links

        print(self.joints_arm)

        self.bullet_client.stepSimulation()

        # define spaces
        self.action_space = spaces.Box(-1., 1.,
                                       shape=(len(self.joints_arm) + 1,))

        self.observation_space = spaces.Dict({
            "joint_positions": spaces.Box(-1., 1.,
                                          shape=(len(self.joints_arm) +
                                                 len(self.joints_hand),)),
            "joint_velocities": spaces.Box(-1., 1.,
                                           shape=(len(self.joints_arm) +
                                                  len(self.joints_hand),)),
            "tcp_position": spaces.Box(-1., 1., shape=(3,)),
        })

        self.key_point_mode = key_point_mode

        # reset to initial position
        self.reset()

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action}"

        action_arm = action[:6]
        action_hand = action[-1]

        action_arm = list(action_arm * self.scale)  # / self.max_steps)

        joint_ids = []
        target_positions = []
        maxVelocities = []
        torques = []

        for (_, joint), action_joint in zip(self.joints_arm.items(),
                                            action_arm):
            position, _, _, _ = self.bullet_client.getJointState(self.model_id,
                                                                 joint.id)

            normalized_joint_position = np.interp(position, joint.limits,
                                                  [-1, 1])
            normalized_target_joint_position = np.clip(
                normalized_joint_position + action_joint, -1, 1)
            target_joint_position = np.interp(normalized_target_joint_position,
                                              [-1, 1], joint.limits)

            joint_ids.append(joint.id)
            target_positions.append(target_joint_position)
            torques.append(joint.max_torque)

            maxVelocities.append(joint.max_velocity)

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     joint_ids,
                                                     p.POSITION_CONTROL,
                                                     targetPositions=target_positions,
                                                     # maxVelocities=maxVelocities,
                                                     forces=torques
                                                     )

        joint_ids = []
        target_positions = []
        maxVelocities = []
        torques = []

        for _, joint in self.joints_hand.items():
            position, _, _, _ = self.bullet_client.getJointState(self.model_id,
                                                                 joint.id)

            target_joint_position = np.interp(action_hand, [-1, 1], joint.limits)

            joint_ids.append(joint.id)
            target_positions.append(target_joint_position)
            torques.append(joint.max_torque)

            maxVelocities.append(joint.max_velocity)

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     joint_ids,
                                                     p.POSITION_CONTROL,
                                                     targetPositions=target_positions,
                                                     forces=torques
                                                     )

        for step in range(self.max_steps):
            self.bullet_client.stepSimulation()

            if self.bullet_client.getConnectionInfo()[
                "connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

        observation = self.get_observation()

        return observation

    def reset(self, desired_state=None):
        """Reset robot to initial pose and return new state."""

        # domain randomization
        for parameter, distribution in self.parameter_distributions.items():

            std = distribution.get("std", 0)

            for link_id, link in self.links.items():
                mean = distribution.get("mean", getattr(link, parameter))

                parameter_value = np.random.normal(mean, std)

                self.bullet_client.changeDynamics(self.model_id, link_id,
                                                  **{
                                                      parameter: parameter_value})

        contact_points = True

        if desired_state is not None:

            desired_state_arm = desired_state[:6]
            desired_state_hand = desired_state[-1]

            for (_, joint), desired_state in zip(self.joints_arm.items(),
                                                 desired_state_arm):
                joint_position = np.interp(desired_state, [-1, 1],
                                           joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            for _, joint in self.joints_hand.items():
                joint_position = np.interp(desired_state_hand, [-1, 1],
                                           joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id,
                                                                 self.model_id)

        # reset until state is valid
        while contact_points:

            for _, joint in self.joints_arm.items():
                joint_position = self.random.uniform(*joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            for _, joint in self.joints_hand.items():
                joint_position = self.random.uniform(*joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id,
                                                                 self.model_id)

        observation = self.get_observation()

        return observation

    def get_observation(self):
        joint_positions, joint_velocities = [], []

        for _, joint in self.joints_arm.items():
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(
                self.model_id, joint.id)

            joint_positions.append(
                np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(np.interp(joint_velocity,
                                              [-joint.max_velocity,
                                               joint.max_velocity],
                                              [-1, 1]))

        for _, joint in self.joints_hand.items():
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(
                self.model_id, joint.id)

            joint_positions.append(
                np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(np.interp(joint_velocity,
                                              [-joint.max_velocity,
                                               joint.max_velocity],
                                              [-1, 1]))

        tcp_position, _, _, _, _, _, tcp_velocity, _ = \
            self.bullet_client.getLinkState(self.model_id,
                                            self.joint_name2id["tcp"],
                                            computeLinkVelocity=True)

        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)
        tcp_position = np.array(tcp_position) - self.offset

        observation = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "tcp_position": tcp_position,
        }

        for key in observation:
            observation[key] = observation[key].clip(
                self.observation_space[key].low,
                self.observation_space[key].high)

        return observation

    def get_key_points(self):
        joint_ids_arm = [joint.id for _, joint in self.joints_arm.items()]
        joint_ids_hand = [joint.id for _, joint in self.joints_hand.items()]

        linkStates_arm = self.bullet_client.getLinkStates(self.model_id, joint_ids_arm, False, True)
        linkStates_hand = self.bullet_client.getLinkStates(self.model_id, joint_ids_hand, False, True)

        if self.key_point_mode == KeyPointMode.LINK_WORLD:
            kp_arm = [(link[0] - self.offset, link[1]) for link in linkStates_arm]
            kp_hand = [(link[0] - self.offset, link[1]) for link in linkStates_hand]
        else:
            kp_arm = [(link[4] - self.offset, link[5]) for link in linkStates_arm]
            kp_hand = [(link[4] - self.offset, link[5]) for link in linkStates_hand]

        return kp_arm, kp_hand
