from gym import spaces
import numpy as np
from numpy.random import RandomState
import os

import numpy as np
import pybullet as p
from gym import spaces
from numpy.random import RandomState

from karolos.utils import unwind_dict_values

try:
    from . import Task
except ImportError:
    from karolos.environments.robot_task_environments.tasks import Task


class Pick_Place(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=100, parameter_distributions=None):

        super(Pick_Place, self).__init__(bullet_client=bullet_client,
                                         parameter_distributions=parameter_distributions,
                                         offset=offset, max_steps=max_steps,
                                         gravity=(0, 0, -9.8))

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.observation_space = spaces.Dict({
            "position": spaces.Box(-1, 1, shape=(3,)),
            "goal": spaces.Box(-1, 1, shape=(3,)),
        })

        self.target = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_SPHERE, radius=.03, rgbaColor=[0, 1, 1, 1]),
        )

        self.tcp_target_constraint = None

        self.object = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_BOX, halfExtents=[.025] * 3),

            baseCollisionShapeIndex=bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=[.025] * 3),
            baseMass=.1,
        )

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    @staticmethod
    def success_criterion(goal):
        # goal -> object + tcp
        goal_achieved = unwind_dict_values(goal["achieved"]["object_position"])
        # desired -> both should be at same desired position
        goal_desired = unwind_dict_values(goal["desired"]["object_position"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        # more error margin
        return goal_distance < 0.05

    def reward_function(self, goal_info, done, **kwargs):
        # 0.2* dist(tcp, obj), 0.8*dist(obj,goal)
        if self.success_criterion(goal_info):
            reward = 1.
        elif done:
            reward = -1.
        else:

            goal_achieved_object = unwind_dict_values(
                goal_info["achieved"]['object_position'])
            goal_achieved_tcp = unwind_dict_values(
                goal_info["achieved"]['tcp_position'])
            goal_desired_object = unwind_dict_values(
                goal_info["desired"]['object_position'])
            goal_desired_tcp = unwind_dict_values(
                goal_info["desired"]['tcp_position'])

            # 0.8 * dist(obj, goal_obj), how close obj is to obj_goal
            reward_object = np.exp(-5 * np.linalg.norm(goal_desired_object -
                                                       goal_achieved_object)) - 1

            # 0.2 * dist(obj, tcp), how close tcp is to obj
            reward_tcp = np.exp(-5 * np.linalg.norm(goal_achieved_tcp -
                                                    goal_desired_tcp)) - 1

            # scale s.t. reward in [-1,1]
            reward = .8 * reward_object + .2 * reward_tcp
        return reward

    def reset(self, robot=None, observation_robot=None, desired_state=None):

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

            desired_state_object[-1] = 0  # put on floor
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

        return self.get_status(observation_robot)

    def step(self, observation_robot, robot):
        if observation_robot["status_hand"] == 1:  # open
            if self.tcp_target_constraint is not None:
                self.bullet_client.removeConstraint(self.tcp_target_constraint)
                self.tcp_target_constraint = None
        elif observation_robot["status_hand"] == 0:  # closing
            assert self.tcp_target_constraint is None

            position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)

            # todo: make theshold a parameter
            if np.linalg.norm(observation_robot["tcp_position"] - np.array(position_object)) < .03:
                self.tcp_target_constraint = self.bullet_client.createConstraint(self.object, -1,
                                                                                 robot.model_id, robot.index_tcp,
                                                                                 p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                                                 parentFramePosition=[0, 0, 0],
                                                                                 childFramePosition=[0, 0, 0])
                self.bullet_client.changeConstraint(self.tcp_target_constraint, maxForce=1e5)

        return super(Pick_Place, self).step(observation_robot, robot)

    def get_status(self, observation_robot=None):
        if observation_robot is None:
            position_tcp = None
        else:
            position_tcp = observation_robot["tcp_position"]

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        position_object_desired, _ = \
            self.bullet_client.getBasePositionAndOrientation(self.target)

        position_object_desired = np.array(position_object_desired)

        observation = {
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

        return observation, goal_info, done


if __name__ == "__main__":
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = Pick_Place(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
