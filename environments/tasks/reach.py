import numpy as np
from gym import spaces
import os
from environments.tasks.task import Task
from numpy.random import RandomState

class Reach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 dof=1, only_positive=False, sparse_reward=False,
                 max_steps=100, domain_randomization=None):

        if domain_randomization is None:
            domain_randomization = {}

        super(Reach, self).__init__(bullet_client=bullet_client,
                                    gravity=[0, 0, 0],
                                    domain_randomization=domain_randomization,
                                    offset=offset,
                                    dof=dof,
                                    only_positive=only_positive,
                                    sparse_reward=sparse_reward)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(0,))
        self.goal_space = spaces.Box(-1, 1, shape=(3,))

        self.target = self.bullet_client.loadURDF("objects/sphere.urdf",
                                                  useFixedBase=True)

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

        self.max_steps = max_steps

    def reset(self, robot=None, desired_state=None):

        super(Reach, self).reset()

        contact_points = True

        if desired_state is not None:
            desired_state = [np.interp(value, [-1, 1], limits)
                             for value, limits in
                             zip(desired_state, self.limits)]

            assert np.linalg.norm(
                desired_state) < 0.85, \
                f"desired_state puts target out of reach. " \
                f"{np.linalg.norm(desired_state)}"

            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.robot, self.target)
            else:
                contact_points = False

        while contact_points:

            target_position = np.random.uniform(-1, 1, 3)

            for dimension in range(self.dof):
                if self.only_positive:
                    target_position[dimension] = self.random.uniform(0,
                                                                   self.limits[
                                                                       dimension, 1])
                else:
                    target_position[dimension] = self.random.uniform(
                        *self.limits[dimension])

            if np.linalg.norm(target_position) < 0.8:
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

    def get_observation(self):
        return np.array([])

    def get_status(self, robot):
        achieved_goal = robot.get_position_tcp()

        desired_goal, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)

        desired_goal = np.array(desired_goal)

        distance_tcp_object = np.linalg.norm(achieved_goal - desired_goal)
        goal_reached = distance_tcp_object < 0.05

        done = goal_reached or self.step_counter >= self.max_steps

        return achieved_goal, desired_goal, goal_reached, done


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    time_step = 1. / 60.
    p.setTimeStep(time_step)
    p.setRealTimeSimulation(0)

    task = Reach(p, dof=3)

    while True:
        p.stepSimulation()

        time.sleep(time_step)

        success = True

        obs = task.reset()

        p.stepSimulation()
