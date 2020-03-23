import gym
from gym import spaces
import numpy as np
import pybullet as p
import logging
import time


class Panda(gym.Env):
    tolerance_joint_rotation = np.pi / 180
    tolerance_joint_linear = 0.001

    def __init__(self, bullet_client, dof=3, state_mode='full',
                 use_gripper=False, offset=(0, 0, 0), time_step=1. / 240.,
                 sim_time=1, scale=0.1):

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
                                            flags=p.URDF_USE_SELF_COLLISION)

        for joint_id in range(self.bullet_client.getNumJoints(self.robot)):
            self.bullet_client.changeDynamics(self.robot, joint_id,
                                              linearDamping=0,
                                              angularDamping=0)

        # define robot parameters
        self.limits_joints_arm = np.array(
            [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
             (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
             (-2.8973, 2.8973)])
        self.limits_joints_hand = np.array([(0, 0.04), (0, 0.04)])

        self.torques_arm = np.array([87, 87, 87, 87, 12, 12, 12])
        self.forces_hand = np.array([70, 70])

        self.initial_joints_arm = np.array([0, 0.5, 0, -0.5, 0, 1., 0.707])
        self.initial_joints_hand = np.array([0.035, 0.035])

        self.ids_joints_arm = np.arange(7)
        self.ids_joints_hand = np.array([9, 10])

        # define controllable parameters
        if self.dof == 2:
            self.ids_joints_arm_controllable = np.array([1, 3, 5])
        else:
            self.ids_joints_arm_controllable = self.ids_joints_arm

        if use_gripper:
            self.ids_joints_hand_controllable = self.ids_joints_hand
        else:
            self.ids_joints_hand_controllable = []

        # define spaces
        self.action_space = spaces.Box(-1., 1., shape=(
            len(self.ids_joints_arm_controllable) + self.use_gripper,))

        if state_mode == 'full':
            self.observation_space = spaces.Box(-1., 1., shape=(
                len(self.ids_joints_arm) + len(self.ids_joints_hand),))
        elif state_mode == 'reduced':
            self.observation_space = spaces.Box(-1., 1., shape=(
                len(self.ids_joints_arm_controllable) + len(
                    self.ids_joints_hand_controllable),))
        elif state_mode == 'tcp':
            self.observation_space = spaces.Box(-1., 1., shape=(3,))

        # reset to initial position
        self.reset()

    def get_id(self):
        return self.robot

    def get_camera_image(self):

        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 5
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect,
                                                         nearplane,
                                                         farplane)
        com_p, com_o, _, _, _, _ = self.bullet_client.getLinkState(self.robot,
                                                                   8,
                                                                   computeForwardKinematics=True)
        com_p = np.array(com_p) + np.array([0.1, 0, 0])
        rot_matrix = self.bullet_client.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (-.5, 0, 1)
        init_up_vector = (1, 0, 0)
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self.bullet_client.computeViewMatrix(com_p,
                                                           com_p + 0.1 * camera_vector,
                                                           up_vector)
        img = self.bullet_client.getCameraImage(200, 200, view_matrix,
                                                projection_matrix)
        return img

    def reset(self):
        """Reset robot to initial pose and return new state."""

        # todo reset to random pose
        for joint_id, initial_state in zip(self.ids_joints_arm,
                                           self.initial_joints_arm):
            self.bullet_client.resetJointState(self.robot, joint_id,
                                               initial_state)

        for joint_id, initial_state in zip(self.ids_joints_hand,
                                           self.initial_joints_hand):
            self.bullet_client.resetJointState(self.robot, joint_id,
                                               initial_state)

        observation = self.get_observation()

        return observation

    def step(self, action: np.ndarray):

        assert self.action_space.contains(action)

        action = action * self.scale

        rel_action_arm, rel_action_hand = np.split(action, [
            len(self.ids_joints_arm_controllable)])

        # todo extend to gripper movement
        target_joints_hand = self.initial_joints_hand.copy()

        state_arm, _ = self.get_joint_state()
        positions_arm, _ = state_arm

        target_joints_arm = positions_arm.copy()

        for id_joint, action in zip(self.ids_joints_arm_controllable,
                                    rel_action_arm):
            joint_index = \
                np.argwhere(self.ids_joints_arm == id_joint).ravel()[0]

            limits_joint = self.limits_joints_arm[joint_index]

            joint_range = limits_joint[1] - limits_joint[0]

            target_joints_arm[joint_index] += action * joint_range
            target_joints_arm[joint_index] = np.clip(
                target_joints_arm[joint_index], limits_joint[0],
                limits_joint[1])

        self.move_joints_to_target(target_joints_arm,
                                   target_joints_hand)

        observation = self.get_observation()

        return observation

    def convert_intervals(self, value, interval_origin, interval_target):

        value_mapped = value - interval_origin[0]
        value_mapped = value_mapped / (interval_origin[1] - interval_origin[0])
        value_mapped = value_mapped * (interval_target[1] - interval_target[0])
        value_mapped = value_mapped + interval_target[0]

        return value_mapped

    def joint_included_in_state(self, id_joint):

        if self.state_mode == 'full':
            return True
        elif self.state_mode == 'reduced':
            return id_joint in self.ids_joints_arm_controllable or \
                   id_joint in self.ids_joints_hand_controllable

    def get_observation(self):

        # self.get_camera_image()

        state_arm, state_hand = self.get_joint_state()

        positions_arm, velocities_arm = state_arm
        positions_hand, velocities_hand = state_hand

        success = True

        observation_arm = []

        for id_joint, position_arm, limits_joint in zip(self.ids_joints_arm,
                                                        positions_arm,
                                                        self.limits_joints_arm):

            position_inside_limits = limits_joint[0] <= position_arm <= \
                                     limits_joint[1]

            # position_inside_limits = position_inside_limits or np.any(
            #     np.isclose(position_arm, limits_joint, atol=0, rtol=0.01))

            if not position_inside_limits:
                self.logger.debug(
                    f"Arm joint {id_joint} outside limits {position_arm} {limits_joint}")

            success = success and position_inside_limits

            if self.joint_included_in_state(id_joint):
                observation_joint = self.convert_intervals(position_arm,
                                                           limits_joint,
                                                           [-1, 1])
                observation_arm.append(observation_joint)

        observation_hand = []

        for id_joint, position_hand, limits_joint in zip(self.ids_joints_hand,
                                                         positions_hand,
                                                         self.limits_joints_hand):

            position_inside_limits = limits_joint[0] <= position_hand <= \
                                     limits_joint[1]

            # position_inside_limits = position_inside_limits or np.any(
            #     np.isclose(position_hand, limits_joint, atol=0, rtol=0.01))

            if not position_inside_limits:
                self.logger.debug(
                    f"Hand outside limits {position_hand} {limits_joint}")

            success = success and position_inside_limits

            if self.joint_included_in_state(id_joint):
                observation_joint = self.convert_intervals(position_hand,
                                                           limits_joint,
                                                           [-1, 1])
                observation_hand.append(observation_joint)

        observation = np.concatenate((observation_arm, observation_hand))

        observation = observation.clip(self.observation_space.low,
                                       self.observation_space.high)
        return observation

    def move_joints_to_target(self, target_joints_arm, target_joints_hand):

        for id_joint, target_joint, torque in zip(
                self.ids_joints_arm,
                target_joints_arm, self.torques_arm):
            self.bullet_client.setJointMotorControl2(self.robot, id_joint,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     target_joint,
                                                     force=torque)
        for id_joint, target_joint, force in zip(
                self.ids_joints_hand,
                target_joints_hand, self.forces_hand):
            self.bullet_client.setJointMotorControl2(self.robot, id_joint,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     target_joint, force=force)

        for step in range(self.max_steps):
            self.bullet_client.stepSimulation()
            if self.bullet_client.getConnectionInfo()[
                "connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

    def get_joint_state(self):

        positions_arm, velocities_arm = [], []

        for j in self.ids_joints_arm:
            position, velocity, _, _ = self.bullet_client.getJointState(
                self.robot, j)
            positions_arm.append(position)
            velocities_arm.append(velocity)

        positions_hand, velocities_hand = [], []

        for j in self.ids_joints_hand:
            position, velocity, _, _ = self.bullet_client.getJointState(
                self.robot, j)
            positions_hand.append(position)
            velocities_hand.append(velocity)

        positions_arm = np.array(positions_arm)
        velocities_arm = np.array(velocities_arm)
        positions_hand = np.array(positions_hand)
        velocities_hand = np.array(velocities_hand)

        state_arm = (positions_arm, velocities_arm)
        state_hand = (positions_hand, velocities_hand)

        return state_arm, state_hand

    def get_position_tcp(self):

        positions_links = []

        for joint_id in self.ids_joints_hand:
            state_link = self.bullet_client.getLinkState(self.robot, joint_id)

            position_link = state_link[0]

            positions_links.append(position_link)

        position_tcp = np.mean(positions_links, axis=0)

        return position_tcp


if __name__ == "__main__":
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    time_step = 1. / 60.
    p.setTimeStep(time_step)
    p.setRealTimeSimulation(0)

    robot = Panda(p, dof=3, state_mode='reduced', time_step=time_step,
                  sim_time=1.)

    while True:
        # action = robot.action_space.sample()

        robot.reset()

        action = np.ones(robot.action_space.shape) * np.random.choice([-1,1])

        obs = robot.step(action)
