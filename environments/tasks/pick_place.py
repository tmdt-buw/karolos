import os

import numpy as np
from gym import spaces
from numpy.random import RandomState

from environments.tasks.task import Task


class Pick_Place(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0), max_steps=100):

        super(Pick_Place, self).__init__(bullet_client=bullet_client,
                                         offset=offset, max_steps=max_steps)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.observation_space = spaces.Dict({
            "position": spaces.Box(-1, 1, shape=(3,)),
        })
        self.goal_space = spaces.Dict({
            "position": spaces.Box(-1, 1, shape=(3,)),
        })

        self.target = self.bullet_client.loadURDF("objects/sphere.urdf",
                                                  useFixedBase=True)
        self.object = self.bullet_client.loadURDF("objects/cube.urdf")

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    def reset(self, robot=None, desired_state=None):

        super(Pick_Place, self).reset()

        if desired_state is not None:
            # gripped flag irrelevant for initial state
            desired_state_object, desired_state_target = np.split(
                np.array(desired_state), 2)

            desired_state_object = [np.interp(value, [-1, 1], limits)
                                    for value, limits in
                                    zip(desired_state_object, self.limits)]
            desired_state_target = [np.interp(value, [-1, 1], limits)
                                    for value, limits in
                                    zip(desired_state_target, self.limits)]

            assert np.linalg.norm(desired_state_object) < 0.8, \
                f"desired_state puts object out of reach. {desired_state_object}"

            assert np.linalg.norm(desired_state_target) < 0.8, \
                f"desired_state puts target out of reach. {desired_state_target}"

            self.bullet_client.resetBasePositionAndOrientation(
                self.object, desired_state_object, [0, 0, 0, 1])

            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state_target, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.robot, self.object)

                assert not contact_points, f"desired_state puts object and " \
                                           f"robot in collision"

        else:
            contact_points = True

            while contact_points:
                object_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])
                target_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])

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


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = Pick_Place(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(5. / time_step):
            p.stepSimulation()

            time.sleep(p.getPhysicsEngineParameters()["fixedTimeStep"])
