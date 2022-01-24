"""
Basically Pick&Place without 3rd dimension if full robots are used
TODO dicuss if we can replace with pick_place task
"""


from gym import spaces
import numpy as np
from numpy.random import RandomState
import os

from utils import unwind_dict_values

try:
    from . import Task
except ImportError:
    from karolos.environments.robot_task_environments.tasks import Task


class Push(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                  max_steps=100, parameter_distributions=None):

        super(Push, self).__init__(bullet_client=bullet_client,
                                   offset=offset,
                                   max_steps=max_steps,
                                   parameter_distributions=parameter_distributions)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        # position target, position object
        self.state_space = spaces.Dict({
            'object_position': spaces.Box(-1, 1, shape=(3,))
        })
        self.goal_space = spaces.Dict({
            "target_position": spaces.Box(-1, 1, shape=(3,))
        })

        # add plane to place box on
        bullet_client.loadURDF("plane.urdf")

        self.target = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE,
                                                     radius=.03,
                                                     rgbaColor=[0, 1, 1, 1],
                                                     ),
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,
                                                           radius=.03,
                                                           ),
        )

        self.object = p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                     halfExtents=[.025] * 3,
                                                     ),

            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,
                                                           halfExtents=[
                                                                           .025] * 3,
                                                           ),
            baseMass=.1,
        )

        # self.bullet_client.changeDynamics(self.object, -1, mass=0)
        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    @staticmethod
    def success_criterion(goal):
        # position object
        goal_achieved = unwind_dict_values(goal["achieved"])
        # desired pos of object
        goal_desired = unwind_dict_values(goal["desired"])
        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        # more error margin
        return goal_distance < 0.04

    def reward_function(self, done, goal, **kwargs):
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:

            goal_achieved_obj = unwind_dict_values(goal["achieved"])
            goal_desired = unwind_dict_values(goal["desired"])

            reward = np.exp(
                -5 * np.linalg.norm(goal_desired - goal_achieved_obj)) - 1

        return reward

    def reset(self, robot=None, state_robot=None, desired_state=None):

        super(Push, self).reset()

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

            desired_state_object[-1] = 0  # put on floor
            desired_state_target[-1] = 0  # put on floor

            self.bullet_client.resetBasePositionAndOrientation(
                self.object, desired_state_object, [0, 0, 0, 1])

            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state_target, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.model_id, self.object)

                assert not contact_points, f"desired_state puts object and " \
                                           f"robot in collision"

        else:
            contact_points = True

            while contact_points:
                object_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])
                object_position[-1] = 0  # put object on floor and use gravity
                target_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])
                target_position[-1] = 0  # put object on floor and use gravity

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
                        robot.model_id, self.object)
                else:
                    contact_points = False

        return self.get_status(state_robot)

    def get_status(self, state_robot=None):
        if state_robot is None:
            position_tcp = None
        else:
            position_tcp = state_robot["tcp_position"]

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        position_object_desired, _ = \
            self.bullet_client.getBasePositionAndOrientation(self.target)

        position_object_desired = np.array(position_object_desired)

        state = {
            "position": position_object,
            "goal": position_object_desired
        }

        achieved_goal = {
            "object_position": position_object,
            "tcp_position": position_tcp
        }

        desired_goal = {
            "object_position": position_object_desired,
            "tcp_position": position_object
        }

        goal_info = {
            'achieved': achieved_goal,
            'desired': desired_goal,
        }

        done = self.step_counter >= self.max_steps

        return state, goal_info, done


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = Push(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
