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
                        robot.robot, self.object)
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

        position_object = np.array(position_object)

        return position_object

    def get_observation(self):
        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        observation = {"position": position_object}

        return observation

    def get_status(self, robot):
        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        position_object_desired, _ = \
            self.bullet_client.getBasePositionAndOrientation(self.target)

        position_object_desired = np.array(position_object_desired)

        achieved_goal = {"position": position_object}

        desired_goal = {"position": position_object_desired}

        done = self.step_counter >= self.max_steps

        return achieved_goal, desired_goal, done
