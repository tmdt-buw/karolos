from environments.tasks.task import Task
import numpy as np
from gym import spaces
import os
from numpy.random import RandomState

class Pick_Place(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0), dof=1,
                 only_positive=False, sparse_reward=False, max_steps=10):

        super(Pick_Place, self).__init__(bullet_client=bullet_client,
                                   gravity=[0, 0, -9.81],
                                   # assuming 1kg box weight
                                   offset=offset,
                                   dof=dof,
                                   only_positive=only_positive,
                                   sparse_reward=sparse_reward)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        # position target, position object

        self.observation_space = spaces.Box(-1, 1, shape=(6,))

        # add plane to place object on
        bullet_client.loadURDF("plane.urdf")

        self.target = self.bullet_client.loadURDF("objects/sphere.urdf",
                                                  useFixedBase=True)
        self.object = self.bullet_client.loadURDF("objects/box.urdf")

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))
        self.max_steps = max_steps

    def reset(self, robot=None, desired_state=None):

        super(Pick_Place, self).reset()
        contact_points = True

        if desired_state is not None:
            # gripped flag irrelevant for initial state
            desired_state_object, desired_state_target = np.split(
                desired_state[:-1], 2)

            assert np.linalg.norm(
                desired_state_object) < 0.8, "desired_state puts object out of reach."

            assert np.linalg.norm(
                desired_state_target) < 0.8, "desired_state puts target out of reach."

            self.bullet_client.resetBasePositionAndOrientation(
                self.object, desired_state_object, [0, 0, 0, 1])

            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state_target, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.robot, self.target)
            else:
                contact_points = False

        while contact_points:
            object_position = np.zeros(3)
            target_position = np.zeros(3)

            for dimension, limits in enumerate(self.limits[:self.dof]):

                if self.only_positive:
                    limits = np.clip(limits, 0, None)

                object_position[dimension] = self.random.uniform(*limits)
                target_position[dimension] = self.random.uniform(*limits)

            #target_position[-1] = 0

            if np.linalg.norm(object_position) < 0.8 and np.linalg.norm(
                    target_position) < 0.8:
                object_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    self.object, object_position, [0, 0, 0, 1])

                target_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    self.target, target_position, [0, 0, 0, 1])
                self.bullet_client.stepSimulation()
            else:
                continue

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.robot, self.target)
            else:
                contact_points = False

        return self.get_observation()

    def get_target(self):

        position_target, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)

        position_target = np.array(position_target)

        return position_target

    def get_object(self):
        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object )

        return position_object

    def get_goals(self, robot, success=True):
        if success:
            achieved_goal = self.get_object()
        else:
            achieved_goal = None

        desired_goal = self.get_target()
        return achieved_goal, desired_goal

    # def get_gripped(self, robot):
    #     # only called by get_object -> get_object = current state of task
    #     # object has to be between gripper
    #     grip = False
    #     assert robot is not None, "provide robot to check if object is gripped"
    #     grip_pos_1, _, _, _ = self.bullet_client.getJointState(robot.robot, 8)
    #     grip_pos_2, _, _, _ = self.bullet_client.getJointState(robot.robot, 9)
    #
    #     position_object, _ = self.bullet_client.getBasePositionAndOrientation(
    #         self.object)
    #
    #     position_tcp = robot.get_position_tcp()
    #
    #     print('#'*20,'gripped', grip_pos_1, '-', grip_pos_2, '-', position_object)
    #     print('#'*20,'    tcp', position_tcp)
    #
    #     return grip

    def get_observation(self):

        position_target = self.get_target()
        position_object = self.get_object()

        observation_object = [np.interp(position, limits, [-1, 1])
                              for position, limits in
                              zip(position_object, self.limits)]

        observation_target = [np.interp(position, limits, [-1, 1])
                              for position, limits in
                              zip(position_target, self.limits)]

        observation = np.concatenate((observation_object, observation_target))
        #print('#'*20,'observations', observation, observation_object, observation_target)
        observation = observation.clip(self.observation_space.low,
                                       self.observation_space.high)

        return observation

    def compute_reward(self, achieved_goal, desired_goal):

        if achieved_goal is not None:
            distance = np.linalg.norm(achieved_goal - desired_goal)

            #gripped_state = achieved_goal[-1]
            #assert gripped_state in [0, 1], f'gripped state in reward not 0/1: {achieved_goal}'

            goal_reached = distance < 0.05
            done = goal_reached or self.step_counter >= self.max_steps

            if goal_reached:
                reward = 1.
            else:
                if self.sparse_reward:
                    reward = -1

                else:
                    reward = np.exp(-distance * 3.5) * 2 - 1
                    reward /= self.max_steps
                    #reward += int(gripped_state) - 1 # either -1 or 0 reward
        else:
            reward = -1.
            goal_reached = False
            done = True

        reward = np.clip(reward, -1, 1)

        return reward, done, goal_reached