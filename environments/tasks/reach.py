import numpy as np
from gym import spaces
import os
from environments.tasks.task import Task
from numpy.random import RandomState

class Reach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=100, domain_randomization=None):

        if domain_randomization is None:
            domain_randomization = {}

        super(Reach, self).__init__(bullet_client=bullet_client,
                                    gravity=[0, 0, 0],
                                    domain_randomization=domain_randomization,
                                    offset=offset,
                                    max_steps=max_steps)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(0,))
        self.goal_space = spaces.Dict({
            "tcp_position": spaces.Box(-1, 1, shape=(3,)),
            # "tcp_velocity": spaces.Box(-1, 1, shape=(3,))
        })

        self.target = self.bullet_client.loadURDF("objects/cube.urdf",
                                                  useFixedBase=True)

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

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

            target_position = np.random.uniform(self.limits[:, 0],
                                                self.limits[:, 1])

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
        observation = robot.get_observation()
        achieved_goal = {
            "position": observation["tcp_position"],
            # "velocity": observation["tcp_velocity"]
        }

        position_desired, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)
        position_desired = np.array(position_desired)

        # velocity_desired = np.zeros_like(observation["tcp_velocity"])

        desired_goal = {
            "position": position_desired,
            # "velocity": velocity_desired
        }

        done = self.step_counter >= self.max_steps

        return achieved_goal, desired_goal, done


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    task = Reach(p)

    while True:
        obs = task.reset()
        p.stepSimulation()

        time.sleep(p.getPhysicsEngineParameters()["fixedTimeStep"])
