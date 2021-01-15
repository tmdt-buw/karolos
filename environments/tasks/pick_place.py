import os

import numpy as np
from gym import spaces
from numpy.random import RandomState

from environments.tasks.task import Task
from utils import unwind_dict_values


class Pick_Place(Task):

    @staticmethod
    def success_criterion(goal):
        # goal -> object + tcp
        goal_achieved = unwind_dict_values(goal["achieved"]["object_position"])
        # desired -> both should be at same desired position
        goal_desired = unwind_dict_values(goal["desired"]["object_position"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        # more error margin
        return goal_distance < 0.05

    def reward_function(self, done, goal, **kwargs):
        # 0.2* dist(tcp, obj), 0.8*dist(obj,goal)
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:

            goal_achieved_object = unwind_dict_values(
                goal["achieved"]['object_position'])
            goal_achieved_tcp = unwind_dict_values(
                goal["achieved"]['tcp_position'])
            goal_desired_object = unwind_dict_values(
                goal["desired"]['object_position'])
            goal_desired_tcp = unwind_dict_values(
                goal["desired"]['tcp_position'])

            # 0.8 * dist(obj, goal_obj), how close obj is to obj_goal
            reward_object = np.exp(-5 * np.linalg.norm(goal_desired_object -
                                                    goal_achieved_object)) - 1

            # 0.2 * dist(obj, tcp), how close tcp is to obj
            reward_tcp = np.exp(-5 * np.linalg.norm(goal_achieved_tcp -
                                                    goal_desired_tcp)) - 1

            # scale s.t. reward in [-1,1]
            reward = .8 * reward_object + .2 * reward_tcp
        return reward

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
        })
        self.goal_space = spaces.Dict({
            "object_position": spaces.Box(-1, 1, shape=(3,)),
            "tcp_position": spaces.Box(-1, 1, shape=(3,))
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

            desired_state_object[-1] = 0  # put on floor
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

        observation = {"object_position": position_object}

        return observation

    def get_status(self, observation):
        position_tcp = observation["tcp_position"]

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        position_object_desired, _ = \
            self.bullet_client.getBasePositionAndOrientation(self.target)

        position_object_desired = np.array(position_object_desired)

        achieved_goal = {
            "object_position": position_object,
            "tcp_position": position_tcp
        }

        desired_goal = {
            "object_position": position_object_desired,
            "tcp_position": position_object
        }

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

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
