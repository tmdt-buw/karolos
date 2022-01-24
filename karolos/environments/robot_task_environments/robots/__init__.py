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
from enum import Enum
from itertools import chain

import klampt
import numpy as np
import pybullet as p
from gym import spaces
from klampt.math import so3
from klampt.model import ik

Joint = namedtuple("Joint", ["id", "initial_position", "limits",
                             "max_velocity", "max_torque"])
Link = namedtuple("Link", ["mass", "linearDamping"])


class STATUS_HAND(Enum):
    CLOSED = -1
    CLOSING = 0
    OPEN = 1


class RobotArm:
    def __init__(self, bullet_client, urdf_file,
                 joints_arm, joints_hand, links=None,
                 offset=(0, 0, 0), sim_time=0.,
                 scale=1., parameter_distributions=None):

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

        self.offset = offset

        self.bullet_client = bullet_client

        self.random = np.random.RandomState(int.from_bytes(os.urandom(4), byteorder='little'))

        self.model_id = bullet_client.loadURDF(urdf_file, self.offset,
                                               useFixedBase=True,
                                               flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        self.joint_name2id = {}

        for jj in range(self.bullet_client.getNumJoints(self.model_id)):
            jointInfo = self.bullet_client.getJointInfo(self.model_id, jj)
            self.joint_name2id[jointInfo[1].decode("utf-8")] = jointInfo[0]

        self.joints_arm = {}
        self.joints_hand = {}

        self.status_hand = STATUS_HAND.OPEN

        for joint_name, joint_args in joints_arm.items():
            self.joints_arm[joint_name] = Joint(self.joint_name2id[joint_name],
                                                *joint_args)

        for joint_name, joint_args in joints_hand.items():
            self.joints_hand[joint_name] = Joint(self.joint_name2id[joint_name], *joint_args)

        if links is None:
            self.links = {}
        else:
            self.links = links

        if not hasattr(self, "index_tcp"):
            self.index_tcp = len(self.links) - 1

        self.bullet_client.stepSimulation()

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=(len(self.joints_arm) + 1,))

        self.state_space = spaces.Dict({
            "joint_positions": spaces.Box(-1., 1., shape=(len(self.joints),)),
            "joint_velocities": spaces.Box(-1., 1., shape=(len(self.joints),)),
            "tcp_position": spaces.Box(-1., 1., shape=(3,)),
            "status_hand": spaces.Box(-1., 1., shape=(1,)),
        })

        # initialize ik

        self.ik_world = klampt.WorldModel()
        self.ik_world.loadElement(urdf_file)
        self.ik_model = self.ik_world.robot(0)
        self.ik_dof_joint_ids = [jj for jj in range(self.ik_model.numLinks()) if
                                 self.ik_model.getJointType(jj) == "normal"]

        assert len(self.ik_dof_joint_ids) == len(self.joints), "Mismatch between specified DOF and DOF found by Klampt!"

        # reset to initial position
        self.reset()

    @property
    def joints(self):
        return list(chain(self.joints_arm.values(), self.joints_hand.values()))

    def calculate_inverse_kinematics(self, tcp_position, tcp_orientation, initial_pose=None):
        if initial_pose is not None:
            assert len(initial_pose) == len(self.ik_dof_joint_ids)

            for ik_dof, pose in zip(self.ik_dof_joint_ids, initial_pose):
                self.ik_model.setDOFPosition(ik_dof, pose)

        obj = ik.objective(self.ik_model.link(self.ik_model.numLinks() - 1), t=list(tcp_position),
                           R=so3.from_quaternion(tcp_orientation))

        res = ik.solve_global(obj, activeDofs=self.ik_dof_joint_ids)

        if not res:
            return None
        else:
            return np.array([self.ik_model.getDOFPosition(jj) for jj in self.ik_dof_joint_ids])

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action}"

        action_arm = action[:-1]
        action_hand = action[-1]

        action_arm = list(action_arm * self.scale)  # / self.max_steps)

        joint_ids = []
        target_positions = []
        maxVelocities = []
        torques = []

        for (_, joint), action_joint in zip(self.joints_arm.items(), action_arm):
            position, _, _, _ = self.bullet_client.getJointState(self.model_id, joint.id)

            normalized_joint_position = np.interp(position, joint.limits, [-1, 1])
            normalized_target_joint_position = np.clip(normalized_joint_position + action_joint, -1, 1)
            target_joint_position = np.interp(normalized_target_joint_position, [-1, 1], joint.limits)

            joint_ids.append(joint.id)
            target_positions.append(target_joint_position)
            torques.append(joint.max_torque)

            maxVelocities.append(joint.max_velocity)

        for _, joint in self.joints_hand.items():
            # always keep hand open, gripping is handled in task
            target_joint_position = joint.limits[-1]

            joint_ids.append(joint.id)
            target_positions.append(target_joint_position)
            torques.append(joint.max_torque)

            maxVelocities.append(joint.max_velocity)

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     joint_ids,
                                                     p.POSITION_CONTROL,
                                                     targetPositions=target_positions,
                                                     # forces=torques
                                                     )

        if action_hand >= 0:
            self.status_hand = STATUS_HAND.OPEN
        elif self.status_hand == STATUS_HAND.OPEN:
            self.status_hand = STATUS_HAND.CLOSING
        else:
            self.status_hand = STATUS_HAND.CLOSED

        for step in range(self.max_steps):
            self.bullet_client.stepSimulation()

            joint_states = self.bullet_client.getJointStates(self.model_id, joint_ids)

            joint_positions = np.array([joint_state[0] for joint_state in joint_states])
            joint_velocities = np.array([joint_state[1] for joint_state in joint_states])

            if max(abs(joint_velocities)) < .01 and max(abs(joint_positions - target_positions)) < .01:
                break

            if self.bullet_client.getConnectionInfo()["connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

        state = self.get_state()

        return state

    def reset(self, desired_state=None):
        """Reset robot to initial pose and return new state."""

        # domain randomization
        for parameter, distribution in self.parameter_distributions.items():

            std = distribution.get("std", 0)

            for link_id, link in self.links.items():
                mean = distribution.get("mean", getattr(link, parameter))

                parameter_value = np.random.normal(mean, std)

                self.bullet_client.changeDynamics(self.model_id, link_id, **{parameter: parameter_value})

        contact_points = True

        if desired_state is not None:
            assert len(desired_state) == len(
                self.joints), f"Please provide {len(self.joints_arm)} values for the arm and 1 value for the hand!"

            desired_state_arm = desired_state[:len(self.joints_arm)]
            desired_state_hand = desired_state[len(self.joints_arm):]

            for (_, joint), desired_state in zip(self.joints_arm.items(), desired_state_arm):
                joint_position = np.interp(desired_state, [-1, 1], joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id, joint_position)

            if np.mean(desired_state_hand) >= 0:
                self.status_hand = STATUS_HAND.OPEN
            else:
                self.status_hand = STATUS_HAND.CLOSING

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id, self.model_id)

        # reset until state is valid
        while contact_points:
            for joint in self.joints:
                joint_position = self.random.uniform(*joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id, joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id, self.model_id)

        state = self.get_state()

        return state

    def get_state(self):
        joint_positions, joint_velocities = [], []

        for joint in self.joints:
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(self.model_id, joint.id)

            joint_positions.append(np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(np.interp(joint_velocity, [-joint.max_velocity, joint.max_velocity], [-1, 1]))

        tcp_position, _, _, _, _, _, tcp_velocity, _ = self.bullet_client.getLinkState(self.model_id,
                                                                                       self.joint_name2id["tcp"],
                                                                                       computeLinkVelocity=True)

        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)
        tcp_position = np.array(tcp_position)

        state = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "tcp_position": tcp_position,
            "status_hand": np.array(self.status_hand.value)
        }

        for key in state:
            state[key] = state[key].clip(
                self.state_space[key].low,
                self.state_space[key].high)

        return state

    def get_key_points(self, useLinkWorld=True):
        joint_ids_arm = [joint.id for _, joint in self.joints_arm.items()]
        joint_ids_hand = [joint.id for _, joint in self.joints_hand.items()]

        linkStates_arm = self.bullet_client.getLinkStates(self.model_id, joint_ids_arm, False, True)
        linkStates_hand = self.bullet_client.getLinkStates(self.model_id, joint_ids_hand, False, True)

        if useLinkWorld:
            kp_arm = [link[:2] for link in linkStates_arm]
            kp_hand = [link[:2] for link in linkStates_hand]
        else:
            kp_arm = [link[4:6] for link in linkStates_arm]
            kp_hand = [link[4:6] for link in linkStates_hand]

        return kp_arm, kp_hand
