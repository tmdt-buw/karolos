import logging
import os
import time
from collections import namedtuple

import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import spaces
from numpy.random import RandomState

# todo implement domain randomization

class Kuka_iiwa:
    def __init__(self, bullet_client, offset=(0, 0, 0), sim_time=0., scale=1.,
                 parameter_distributions=None):

        self.logger = logging.Logger(f"robot:panda:{bullet_client}")

        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for iiwa")
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
        self.robot = bullet_client.loadURDF("robots/kuka_lbr_iiwa/iiwa.urdf",
                                            np.array([0, 0, 0]) + self.offset,
                                            useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)

        Joint = namedtuple("Joint",
                           ["initial_position", "limits", "max_velocity",
                            "max_torque"])

        self.joints = {
            0: Joint(0, (-2.96705972839, 2.96705972839), 1.71042266695, 320),
            1: Joint(0, (-2.09439510239, 2.09439510239), 1.71042266695, 320),
            2: Joint(0, (-2.96705972839, 2.96705972839), 1.74532925199, 176),
            3: Joint(0, (-2.09439510239, 2.09439510239), 2.26892802759, 176),
            4: Joint(0, (-2.96705972839, 2.96705972839), 2.44346095279, 110),
            5: Joint(0, (-2.09439510239, 2.09439510239), 3.14159265359, 40),
            6: Joint(0, (-3.05432619099, 3.05432619099), 3.14159265359, 40),

        }

        self.joints_arm = list(range(7))
        self.joints_fingers = []

        # todo introduce friction
        self.bullet_client.setJointMotorControlArray(self.robot,
                                                     self.joints_fingers,
                                                     p.VELOCITY_CONTROL,
                                                     forces=[0 * self.joints[idx].max_torque
                                                         for idx in self.joints_fingers])

        self.bullet_client.stepSimulation()

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=(len(self.joints_arm) + 1,))

        self.observation_space = spaces.Dict({
            "joint_positions": spaces.Box(-1., 1., shape=(len(self.joints_arm) + len(self.joints_fingers),)),
            "joint_velocities": spaces.Box(-1., 1., shape=(len(self.joints_arm) + len(self.joints_fingers),)),
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
                joint_position = np.interp(desired_state.pop(0), [-1, 1],
                                           joint.limits)

                self.bullet_client.resetJointState(self.robot, joint_id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.robot,
                                                                 self.robot)

        # reset until state is valid
        while contact_points:

            for joint_id, joint in self.joints.items():
                joint_position = self.random.uniform(*joint.limits)

                self.bullet_client.resetJointState(self.robot, joint_id,
                                                   joint_position)

            self.bullet_client.stepSimulation()
            contact_points = self.bullet_client.getContactPoints(self.robot,
                                                                 self.robot)

        observation = self.get_observation()

        return observation

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), f"{action}"

        action_arm = action[:7]
        action_fingers = action[-1]

        action_arm = list(action_arm * self.scale) # / self.max_steps)

        target_arm_positions = []
        torques_arm = []

        for joint_id, action_joint in zip(self.joints_arm, action_arm):

            position, _, _, _ = self.bullet_client.getJointState(self.robot, joint_id)

            normalized_joint_position = np.interp(position, self.joints[joint_id].limits, [-1, 1])
            normalized_target_joint_position = np.clip(normalized_joint_position + action_joint, -1, 1)
            target_joint_position = np.interp(normalized_target_joint_position, [-1, 1], self.joints[joint_id].limits)

            target_arm_positions.append(target_joint_position)
            torques_arm.append(self.joints[joint_id].max_torque)

        self.bullet_client.setJointMotorControlArray(self.robot,
                                                     self.joints_arm,
                                                     p.POSITION_CONTROL,
                                                     targetPositions=target_arm_positions,
                                                     forces=torques_arm
                                                     )

        torques_fingers = [action_fingers * self.joints[joint_id].max_torque for joint_id in self.joints_fingers]

        for step in range(self.max_steps):

            self.bullet_client.setJointMotorControlArray(self.robot,
                                                         self.joints_fingers,
                                                         p.TORQUE_CONTROL,
                                                         forces=torques_fingers)

            self.bullet_client.stepSimulation()

            if self.bullet_client.getConnectionInfo()["connectionMethod"] == p.GUI:
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
            self.bullet_client.getLinkState(self.robot, 6,
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
        logging.warning("Domain randomization not implemented for UR5")

    def standard(self):
        pass

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    p.setTimeStep(1. / 300.)

    p.setRealTimeSimulation(0)

    p.setGravity(0, 0, -9.81)

    robot = Kuka_iiwa(p, sim_time=.1, scale=.1)

    while True:
        observation = robot.reset()

        action = -np.ones_like(robot.action_space.sample())

        for _ in range(25):
            observation = robot.step(action)

        for _ in range(25):
            observation = robot.step(-action)