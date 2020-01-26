import pybullet as p
import numpy as np
import gym
from gym import spaces
import time


class Panda(gym.Env):

    def __init__(self, bullet_client, dof=3, state_mode='full',
                 use_gripper=False, offset=(0, 0, 0), time_step=1. / 240.):

        assert dof in [2, 3]
        self.dof = dof
        assert state_mode in ['full', 'reduced', 'tcp']
        self.state_mode = state_mode
        self.use_gripper = use_gripper
        self.time_step = time_step

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

    def __reset(self):

        for joint_id, initial_state in zip(self.ids_joints_arm,
                                           self.initial_joints_arm):
            self.bullet_client.resetJointState(self.robot, joint_id,
                                               initial_state)

        for joint_id, initial_state in zip(self.ids_joints_hand,
                                           self.initial_joints_hand):
            self.bullet_client.resetJointState(self.robot, joint_id,
                                               initial_state)

    def reset(self):
        """Reset robot to initial pose and return new state."""

        success = False
        observation = None

        # reset until state is valid
        while not success:
            self.__reset()
            success, observation = self.get_observation()

        return observation

    def step(self, action):

        assert self.action_space.contains(action)

        action_arm, action_hand = np.split(action, [
            len(self.ids_joints_arm_controllable)])

        target_joints_arm = self.initial_joints_arm.copy()
        target_joints_hand = self.initial_joints_hand.copy()

        success_target_determination = True

        for id_joint, action in zip(self.ids_joints_arm_controllable,
                                    action_arm):

            joint_position = \
                np.argwhere(self.ids_joints_arm == id_joint).ravel()[0]

            limits_joint = self.limits_joints_arm[joint_position]

            action = self.convert_intervals(action, [-1, 1], limits_joint)

            if action < limits_joint[0] or limits_joint[1] < action:
                action_clipped = np.clip(action, limits_joint[0], limits_joint[1])

                if not np.allclose(action_clipped, action):
                    success_target_determination = False

                action = action_clipped


            target_joints_arm[joint_position] = action

        success_movement = self.move_joints_to_target(target_joints_arm,
                                                      target_joints_hand)

        success_observation, observation = self.get_observation()

        success = success_movement and success_observation and success_target_determination

        return success, observation

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
                print(id_joint, position_arm, limits_joint)

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
                print(position_hand, limits_joint)

            success = success and position_inside_limits

            if self.joint_included_in_state(id_joint):
                observation_joint = self.convert_intervals(position_hand,
                                                           limits_joint,
                                                           [-1, 1])
                observation_hand.append(observation_joint)

        observation = np.concatenate((observation_arm, observation_hand))

        if not self.observation_space.contains(observation):
            observation = observation.clip(self.observation_space.low,
                                           self.observation_space.high)
            return False, observation
        else:
            return True, observation

    def move_joints_to_target(self, target_joints_arm, target_joints_hand,
                              max_steps=100):
        for id_joint, target_joint, torque in zip(
                self.ids_joints_arm,
                target_joints_arm, self.torques_arm):
            self.bullet_client.setJointMotorControl2(self.robot, id_joint,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     target_joint, force=torque)
        for id_joint, target_joint, force in zip(
                self.ids_joints_hand,
                target_joints_hand, self.forces_hand):
            self.bullet_client.setJointMotorControl2(self.robot, id_joint,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     target_joint, force=force)

        success = False
        step = 0

        while not success and step < max_steps:
            p.stepSimulation()
            if p.getConnectionInfo()["connectionMethod"] == p.GUI:
                time.sleep(self.time_step)

            state_arm, state_hand = self.get_joint_state()

            positions_arm, velocities_arm = state_arm
            positions_hand, velocities_hand = state_hand

            if np.allclose(velocities_arm, 0, atol=1e-1) and np.allclose(
                    velocities_hand, 0, atol=1e-1):

                if np.allclose(positions_arm, target_joints_arm,
                               atol=1e-1) and np.allclose(positions_hand,
                                                          target_joints_hand,
                                                          atol=1e-1):
                    success = True

            step += 1

        return success

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
    # p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    time_step = 1. / 60.
    p.setTimeStep(time_step)
    p.setRealTimeSimulation(0)

    robot = Panda(p, dof=2, state_mode='reduced')

    for ii in np.linspace(-1, 1, 5):
        for jj in np.linspace(-1, 1, 5):
            for kk in np.linspace(-1, 1, 5):

                print(ii, jj, kk)

                success = True

                obs = robot.reset()
                # p.stepSimulation()

                action = np.array([ii,jj,kk])

                success, obs = robot.step(action)


        # while success:
        #
        #     action = robot.action_space.sample()
        #     # action = np.array([0.28,.71,-.46])
        #     # action = -np.array([.99,.99,.99])
        #
        #     success, obs = robot.step(action)
        #
        #     print(success, obs)
        #
        #     p.stepSimulation()
