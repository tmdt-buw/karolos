import logging
import os
import time
from collections import namedtuple
from enum import Enum
from itertools import chain

import klampt
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from gym import spaces
from klampt.math import so3
from klampt.model import ik

Joint = namedtuple("Joint", ["id", "initial_position", "limits",
                             "max_velocity", "max_torque"])
Link = namedtuple("Link", ["mass", "linearDamping"])


class Robot:
    """
    Parent class of all robots.
    """
    class STATUS_HAND(Enum):
        CLOSED = -1
        CLOSING = 0
        OPEN = 1

    def __init__(self, urdf_file, joints_arm, joints_hand, links=None, dht_params=None, offset=(0, 0, 0), sim_time=0.,
                 scale=1.,
                 parameter_distributions=None, bullet_client=None):

        if bullet_client is None:
            bullet_client = bc.BulletClient()

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 300.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        self.bullet_client = bullet_client

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        if parameter_distributions is None:
            parameter_distributions = {}
        self.parameter_distributions = parameter_distributions

        self.time_step = bullet_client.getPhysicsEngineParameters()["fixedTimeStep"]

        if not sim_time:
            sim_time = self.time_step

        if sim_time < self.time_step:
            self.logger.warning(
                "time step of robot is smaller than time step of simulation. This might lead to unintended behavior.")

        self.scale = scale

        self.max_steps = int(sim_time / self.time_step)

        self.offset = np.array(offset)

        self.random = np.random.RandomState(int.from_bytes(os.urandom(4), byteorder='little'))

        self.model_id = bullet_client.loadURDF(urdf_file, self.offset,
                                               useFixedBase=True,
                                               # flags=p.URDF_MAINTAIN_LINK_ORDER)
                                               flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        self.joint_name2id = {}

        for jj in range(self.bullet_client.getNumJoints(self.model_id)):
            jointInfo = self.bullet_client.getJointInfo(self.model_id, jj)
            self.joint_name2id[jointInfo[1].decode("utf-8")] = jointInfo[0]

        self.joints_arm = {}
        self.joints_hand = {}

        self.status_hand = Robot.STATUS_HAND.OPEN

        for joint_name, joint_args in joints_arm.items():
            self.joints_arm[joint_name] = Joint(self.joint_name2id[joint_name], *joint_args)

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
        self.action_space = spaces.Dict({
            "arm": spaces.Box(-1., 1., shape=(len(self.joints),), dtype=np.float64),
            "hand": spaces.Box(-1., 1., shape=(1,), dtype=np.float64)
        })

        self.state_space = spaces.Dict({
            "arm": spaces.Dict({
                "joint_positions": spaces.Box(-1., 1., shape=(len(self.joints_arm),), dtype=np.float64)
            }),
            "hand": spaces.Box(-1., 1., shape=(1,), dtype=np.float64),
        })

        # initialize ik

        self.ik_world = klampt.WorldModel()
        self.ik_world.loadElement(urdf_file)
        self.ik_model = self.ik_world.robot(0)
        self.ik_dof_joint_ids = [jj for jj in range(self.ik_model.numLinks()) if
                                 self.ik_model.getJointType(jj) == "normal"]

        assert len(self.ik_dof_joint_ids) == len(self.joints), "Mismatch between specified DOF and DOF found by Klampt!"

        if dht_params is not None:
            self.dht_params = dht_params

    def __del__(self):
        self.bullet_client.removeBody(self.model_id)

    @property
    def joints(self):
        return self.joints_arm.values()

    def normalize_joints(self, joint_positions):
        return np.array([np.interp(joint_position, joint.limits, [-1, 1]) for joint, joint_position in
                         zip(self.joints, joint_positions)])

    def unnormalize_joints(self, joint_positions):
        return np.array([np.interp(joint_position, [-1, 1], joint.limits) for joint, joint_position in
                         zip(self.joints, joint_positions)])

    def calculate_inverse_kinematics(self, tcp_position, tcp_orientation, initial_pose=None, iters=1000):
        if initial_pose is not None:
            assert len(initial_pose) == len(self.ik_dof_joint_ids)

            conf = self.ik_model.getConfig()
            for ik_dof, pose in zip(self.ik_dof_joint_ids, initial_pose):
                conf[ik_dof] = pose
            self.ik_model.setConfig(conf)

        obj = ik.objective(self.ik_model.link(self.ik_model.numLinks() - 1), t=list(tcp_position),
                           R=so3.from_quaternion(tcp_orientation))

        res = ik.solve_global(obj, iters=iters, activeDofs=self.ik_dof_joint_ids)

        if not res:
            return None

        return np.array([self.ik_model.getDOFPosition(jj) for jj in self.ik_dof_joint_ids])

    def step(self, action: np.ndarray):
        """
        Execute an action using the robotic arm
        :param action: the action
        :return: an observation
        """
        assert action in self.action_space, f"{self.action_space} {action}"

        action_arm = action["arm"]
        action_hand = action["hand"]

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

        if np.mean(action_hand) >= 0:
            self.status_hand = Robot.STATUS_HAND.OPEN
        elif self.status_hand == Robot.STATUS_HAND.OPEN:
            self.status_hand = Robot.STATUS_HAND.CLOSING
        else:
            self.status_hand = Robot.STATUS_HAND.CLOSED

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

    def reset(self, desired_state=None, force=False):
        """Reset robot to initial pose and return new state."""

        # domain randomization
        for parameter, distribution in self.parameter_distributions.items():

            std = distribution.get("std", 0)

            for link_id, link in self.links.items():
                mean = distribution.get("mean", getattr(link, parameter))

                parameter_value = np.random.normal(mean, std)

                self.bullet_client.changeDynamics(self.model_id, link_id, **{parameter: parameter_value})

        if desired_state is None:
            desired_state = {}

        def complete_state(state_dict, space):
            for key in space:
                if key not in state_dict:
                    state_dict[key] = space[key].sample()
                elif type(space[key]) == spaces.Dict:
                    state_dict[key] = complete_state(state_dict[key], space[key])
            return state_dict

        desired_state = complete_state(desired_state, self.state_space)

        # reset until state is valid
        while True:
            for (_, joint), desired_state_joint in zip(self.joints_arm.items(),
                                                       desired_state["arm"]["joint_positions"]):
                joint_position = np.interp(desired_state_joint, [-1, 1], joint.limits)

                self.bullet_client.resetJointState(self.model_id, joint.id, joint_position)

            if desired_state["hand"] >= 0:
                self.status_hand = Robot.STATUS_HAND.OPEN
            else:
                self.status_hand = Robot.STATUS_HAND.CLOSING

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.model_id, self.model_id)

            if not contact_points or force:
                break
            else:
                # try reset again with random state
                desired_state = self.state_space.sample()

        state = self.get_state()

        self.bullet_client.setJointMotorControlArray(self.model_id,
                                                     [joint.id for joint in self.joints],
                                                     p.VELOCITY_CONTROL,
                                                     targetVelocities=np.zeros(len(self.joints)),
                                                     # forces=torques
                                                     )

        return state

    def get_tcp_pose(self):
        tcp_position, tcp_orientation, _, _, _, _, tcp_velocity, _ = self.bullet_client.getLinkState(self.model_id,
                                                                                                     self.joint_name2id[
                                                                                                         "tcp"],
                                                                                                     computeLinkVelocity=True)

        tcp_position = np.array(tcp_position) - self.offset
        tcp_orientation = np.array(tcp_orientation)

        return tcp_position, tcp_orientation

    def get_state(self):
        """
        Convert pybullet robot state to observation
        :return: the observation
        """
        joint_positions, joint_velocities = [], []

        for joint in self.joints:
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(self.model_id, joint.id)

            joint_positions.append(np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(np.interp(joint_velocity, [-joint.max_velocity, joint.max_velocity], [-1, 1]))

        joint_positions = np.array(joint_positions)
        # joint_velocities = np.array(joint_velocities)

        state = {
            "arm": {"joint_positions": joint_positions},
            "hand": np.array(self.status_hand.value)
        }

        def clip_state(state_dict, space):
            for key in state_dict:
                if type(state_dict[key]) == dict:
                    state_dict[key] = clip_state(state_dict[key], space[key])
                else:
                    state_dict[key] = state_dict[key].clip(space[key].low,
                                                           space[key].high)
            return state_dict

        state = clip_state(state, self.state_space)

        return state
