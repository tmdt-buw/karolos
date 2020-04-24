from collections import namedtuple
import gym
from gym import spaces
import logging
import numpy as np
import pybullet as p
import pybullet_data as pd
import time


class Panda(gym.Env):
    tolerance_joint_rotation = np.pi / 180
    tolerance_joint_linear = 0.001

    def __init__(self, bullet_client, dof=3, state_mode='full',
                 use_gripper=False, offset=(0, 0, 0), time_step=1. / 240.,
                 sim_time=.1, scale=0.1):

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        assert dof in [2, 3]
        self.dof = dof
        assert state_mode in ['full', 'reduced', 'tcp']
        self.state_mode = state_mode
        self.use_gripper = use_gripper
        self.time_step = time_step
        self.scale = scale

        self.max_steps = int(sim_time / time_step)

        self.offset = offset

        self.bullet_client = bullet_client

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

            # hand
            9: Joint(0.035, (0, 0.04), 0.05, 70),
            10: Joint(0.035, (0, 0.04), 0.05, 70),
        }

        for joint_id, joint in self.joints.items():
            self.bullet_client.changeDynamics(self.robot, joint_id,
                                              linearDamping=0,
                                              angularDamping=0)

        # define controllable parameters
        if self.dof == 2:
            self.ids_controllable = np.array([1, 3, 5])
        else:
            self.ids_controllable = np.arange(7)

        if use_gripper:
            self.ids_controllable = np.concatenate(
                (self.ids_controllable, [9, 10]))

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=
        self.ids_controllable.shape)

        if state_mode == 'full':
            self.observation_space = spaces.Box(-1., 1., shape=(
                2 * len(self.joints),))
        elif state_mode == 'reduced':
            self.observation_space = spaces.Box(-1., 1., shape=(
                2 * len(self.ids_controllable),))

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
                    joint_position = np.random.uniform(*joint.limits)
                else:
                    joint_position = joint.initial_position

                self.bullet_client.resetJointState(self.robot, joint_id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.robot,
                                                                 self.robot)

        observation = self.get_observation()

        return observation

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        action = list(action * self.scale)

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
        positions, velocities = [], []

        for joint_id, joint in self.joints.items():
            if self.state_mode == 'reduced' and joint_id not in self.ids_controllable:
                continue

            position, velocity, _, _ = self.bullet_client.getJointState(
                self.robot, joint_id)

            positions.append(np.interp(position, joint.limits, [-1, 1]))
            velocities.append(
                np.interp(velocity, [-joint.max_velocity, joint.max_velocity],
                          [-1, 1]))

        positions = np.array(positions)
        velocities = np.array(velocities)

        observation = np.concatenate((positions, velocities))

        observation = observation.clip(self.observation_space.low,
                                       self.observation_space.high)
        return observation

    def get_position_tcp(self):

        state_fingers = self.bullet_client.getLinkStates(self.robot, [9, 10])

        positions_fingers = [state_finger[0] for state_finger in state_fingers]

        position_tcp = np.mean(positions_fingers, axis=0)

        return position_tcp


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
