import logging
import os
import time
from collections import namedtuple

import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import spaces
from numpy.random import RandomState

class Panda(gym.Env):
    tolerance_joint_rotation = np.pi / 180
    tolerance_joint_linear = 0.001

    def __init__(self, bullet_client, dof=3, state_mode='full',
                 use_gripper=False, mirror_finger_control=False, offset=(0, 0, 0), time_step=1. / 240.,
                 sim_time=.1, scale=0.1, domain_randomization=None):

        if domain_randomization is None:
            domain_randomization = {}

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        assert dof in [2, 3]
        self.dof = dof
        assert state_mode in ['full', 'reduced', 'tcp']
        if mirror_finger_control:
            assert use_gripper, 'cannot mirror control if not using gripper'

        self.state_mode = state_mode
        self.use_gripper = use_gripper
        self.mirror_finger = mirror_finger_control
        self.time_step = time_step
        self.scale = scale
        self.domain_randomization = domain_randomization

        self.max_steps = int(sim_time / time_step)

        self.offset = offset

        self.bullet_client = bullet_client

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

        # load robot in simulation
        self.robot = bullet_client.loadURDF("robots/panda/panda.urdf",
                                            np.array([0, 0, 0]) + self.offset,
                                            useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        Joint = namedtuple("Joint",
                           ["initial_position", "limits", "max_velocity",
                            "torque"])

        self.joints = {
            0: Joint(0, (-2.8973, 2.8973), 2.1750, 87),
            1: Joint(0.5, (-1.7628, 1.7628), 2.1750, 87),
            2: Joint(0, (-2.8973, 2.8973), 2.1750, 87),
            3: Joint(-0.5, (-3.0718, -0.0698), 2.1750, 87),
            4: Joint(0, (-2.8973, 2.8973), 2.6100, 12),
            5: Joint(1., (-0.0175, 3.7525), 2.6100, 12),
            6: Joint(0.707, (-2.8973, 2.8973), 2.6100, 12),

            # finger 1 & 2
            8: Joint(0.035, (0.0, 0.04), 0.05, 70),
            9: Joint(0.035, (0.0, 0.04), 0.05, 70),
        }

        Link = namedtuple("Link", ["mass", "linear_damping"])

        self.links = {
            0: Link(2.7, 0.01),
            1: Link(2.73, 0.01),
            2: Link(2.04, 0.01),
            3: Link(2.08, 0.01),
            4: Link(3.0, 0.01),
            5: Link(1.3, 0.01),
            6: Link(0.2, 0.01),
            7: Link(0.81, 0.01),
            8: Link(0.1, 0.01),
            9: Link(0.1, 0.01),
            10: Link(0.0, 0.01),
        }

        for joint_id, joint in self.joints.items():
            self.bullet_client.changeDynamics(self.robot, joint_id,
                                              linearDamping=0)
            self.bullet_client.changeDynamics(self.robot, joint_id,
                                              angularDamping=0)

        self.standard()

        # define controllable parameters
        if self.dof == 2:
            self.ids_controllable = np.array([1, 3, 5])
        else:
            self.ids_controllable = np.arange(7)

        if use_gripper:
            if self.mirror_finger:
                self.ids_controllable = np.concatenate(
                    (self.ids_controllable, [8]))
            else:
                self.ids_controllable = np.concatenate(
                    (self.ids_controllable, [8, 9]))

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=(len(self.joints_arm) + 1,))

        self.observation_space = spaces.Dict({
            "joint_positions": spaces.Box(-1., 1., shape=(len(self.joints),)),
            "joint_velocities": spaces.Box(-1., 1., shape=(len(self.joints),)),
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

            for joint_id, joint in self.joints.items():
                if joint_id in self.ids_controllable:
                    joint_position = np.interp(desired_state.pop(0), [-1, 1],
                                               joint.limits)
                else:

                    if self.mirror_finger and joint_id == 9:
                        self.bullet_client.resetJointState(self.robot, joint_id,
                                                           joint_position)
                        continue

                    joint_position = joint.initial_position

                self.bullet_client.resetJointState(self.robot, joint_id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.robot,
                                                                 self.robot)

        # reset until state is valid
        while contact_points:

            for joint_id, joint in self.joints.items():

                if joint_id in self.ids_controllable:
                    joint_position = self.random.uniform(*joint.limits)
                else:

                    if self.mirror_finger and joint_id == 9:
                        self.bullet_client.resetJointState(self.robot, joint_id,
                                                           joint_position)
                        continue

                    joint_position = joint.initial_position

                self.bullet_client.resetJointState(self.robot, joint_id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.robot,
                                                                 self.robot)

        observation = self.get_observation()

        return observation

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action}"

        action = list(action * self.scale)

        # duplicate action
        if self.mirror_finger:
            action.append(action[-1])

        for joint_id, joint in self.joints.items():

            if joint_id in self.ids_controllable:
                position, _, _, _ = self.bullet_client.getJointState(
                    self.robot, joint_id)

                action_joint = action.pop(0)

                normalized_joint_position = np.interp(position, joint.limits,
                                                      [-1, 1])
                normalized_target_joint_position = normalized_joint_position + action_joint
                normalized_target_joint_position = np.clip(
                    normalized_target_joint_position, -1, 1)
                target_joint_position = np.interp(
                    normalized_target_joint_position, [-1, 1], joint.limits)

            else:

                if self.mirror_finger and joint_id == 9:
                    position, _, _, _ = self.bullet_client.getJointState(
                        self.robot, joint_id)
                    action_joint = action.pop(0)
                    normalized_joint_position = np.interp(position,
                                                          joint.limits,
                                                          [-1, 1])
                    normalized_target_joint_position = normalized_joint_position + action_joint
                    normalized_target_joint_position = np.clip(
                        normalized_target_joint_position, -1, 1)
                    target_joint_position = np.interp(
                        normalized_target_joint_position, [-1, 1],
                        joint.limits)
                    self.bullet_client.setJointMotorControl2(self.robot,
                                                             joint_id,
                                                             self.bullet_client.POSITION_CONTROL,
                                                             target_joint_position,
                                                             force=joint.torque)
                    continue

                target_joint_position = joint.initial_position

            self.bullet_client.setJointMotorControl2(self.robot, joint_id,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     target_joint_position,
                                                     force=joint.torque)

        for step in range(self.max_steps):
            self.bullet_client.stepSimulation()
            if self.bullet_client.getConnectionInfo()[
                "connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

        observation = self.get_observation()

        return observation

    def get_observation(self):
        joint_positions, joint_velocities = [], []

        for joint_id, joint in self.joints.items():
            joint_position, joint_velocity, _, _ = self.bullet_client.getJointState(
                self.robot, joint_id)

            joint_positions.append(np.interp(joint_position, joint.limits, [-1, 1]))
            joint_velocities.append(
                np.interp(joint_velocity, [-joint.max_velocity, joint.max_velocity],
                          [-1, 1]))

        tcp_position, _, _, _, _, _, tcp_velocity, _ = \
            self.bullet_client.getLinkState(self.robot, 10,
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

    def randomize(self):

        for link_id, link in self.links.items():
            if 'linear_damping' in self.domain_randomization.keys():
                linear_damping = max(0, np.random.normal(self.domain_randomization['linear_damping']['mean'],
                                                         self.domain_randomization['linear_damping']['std']))
            else:
                linear_damping = link.linear_damping

            if 'mass' in self.domain_randomization.keys():
                mass = np.random.normal(link.mass, link.mass * self.domain_randomization['mass']['std_factor'])
            else:
                mass = link.mass

            self.bullet_client.changeDynamics(self.robot, link_id, linearDamping=linear_damping)
            self.bullet_client.changeDynamics(self.robot, link_id, mass=mass)

    def standard(self):

        for link_id, link in self.links.items():

            if 'linear_damping' in self.domain_randomization.keys():
                linear_damping = self.domain_randomization['linear_damping']['mean']
            else:
                linear_damping = link.linear_damping

            self.bullet_client.changeDynamics(self.robot, link_id, linearDamping=linear_damping)
            self.bullet_client.changeDynamics(self.robot, link_id, mass=link.mass)


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    time_step = 1. / 60.
    p.setTimeStep(time_step)
    p.setRealTimeSimulation(0)

    robot = Panda(p, dof=3, time_step=time_step, sim_time=.1, scale=.1)

    initial_pose = np.ones(8)

    while True:
        # action = robot.action_space.sample()

        robot.reset(initial_pose)

        for i in range(50):
            action = -np.ones_like(robot.action_space.sample())

            obs = robot.step(action)

            # print(min(obs[9:]), max(obs[9:]))
