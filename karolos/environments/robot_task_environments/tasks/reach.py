from gym import spaces
import numpy as np
from numpy.random import RandomState
import os

from karolos.utils import unwind_dict_values

try:
    from . import Task
except:
    from karolos.environments.robot_task_environments.tasks import Task


class Reach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=100, parameter_distributions=None):

        super(Reach, self).__init__(bullet_client=bullet_client,
                                    parameter_distributions=parameter_distributions,
                                    offset=offset,
                                    max_steps=max_steps)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(3,))

        self.target = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE,
                                                     radius=.03,
                                                     rgbaColor=[0, 1, 1, 1],
                                                     ),
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,
                                                           radius=.03,
                                                           ),
        )

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    @staticmethod
    def success_criterion(goal_info):
        goal_achieved = unwind_dict_values(goal_info["achieved"])
        goal_desired = unwind_dict_values(goal_info["desired"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        return goal_distance < 0.05

    def reward_function(self, goal_info, done, **kwargs):
        if self.success_criterion(goal_info):
            reward = 1.
        elif done:
            reward = -1.
        else:
            goal_achieved = unwind_dict_values(goal_info["achieved"])
            goal_desired = unwind_dict_values(goal_info["desired"])

            reward = np.exp(
                -5 * np.linalg.norm(goal_achieved - goal_desired)) - 1

        return reward

    def reset(self, robot=None, observation_robot=None, desired_state=None):

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
                    robot.model_id, self.target)
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
                    robot.model_id, self.target)
            else:
                contact_points = False

        return self.get_status(observation_robot)

    def get_status(self, observation_robot=None):

        position_desired, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)
        position_desired = np.array(position_desired)

        observation = {
            "goal": {"position": position_desired}
        }

        if observation_robot is None:
            achieved_goal = {"position": None}
        else:
            achieved_goal = {
                "position": observation_robot["tcp_position"],
                # "velocity": observation["tcp_velocity"]
            }

        desired_goal = {
            "position": position_desired,
            # "velocity": np.zeros_like(observation["tcp_velocity"])
        }

        done = self.step_counter >= self.max_steps

        goal_info = {
            'achieved': achieved_goal,
            'desired': desired_goal,
        }

        return observation, goal_info, done


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = Reach(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
