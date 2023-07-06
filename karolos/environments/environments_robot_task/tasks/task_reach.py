import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from gym import spaces
from numpy.random import RandomState

sys.path.append(str(Path(__file__).resolve().parent))
from task import Task

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils import unwind_dict_values

class TaskReach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=25, accuracy=0.05, parameter_distributions=None):

        super(TaskReach, self).__init__(bullet_client=bullet_client,
                                        parameter_distributions=parameter_distributions,
                                        offset=offset,
                                        max_steps=max_steps)

        self.accuracy = accuracy

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.state_space = spaces.Dict({
        })

        self.goal_space = spaces.Dict({
            "achieved": spaces.Dict({
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
            }),
            "desired": spaces.Dict({
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
            })
        })

        self.target = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_SPHERE,
                                                                 radius=self.accuracy,
                                                                 rgbaColor=[0, 1, 1, 1],
                                                                 ),
        )

        self.random = RandomState(int.from_bytes(os.urandom(4), byteorder='little'))

    def __del__(self):
        self.bullet_client.removeBody(self.target)

    def success_criterion(self, goal, *args, **kwargs):
        goal_achieved = unwind_dict_values(goal["achieved"])
        goal_desired = unwind_dict_values(goal["desired"])

        goal_achieved = np.array(
            [np.interp(value, [-1, 1], limits) for value, limits in zip(goal_achieved, self.limits)])
        goal_desired = np.array(
            [np.interp(value, [-1, 1], limits) for value, limits in zip(goal_desired, self.limits)])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        return goal_distance < self.accuracy

    def reward_function(self, goal, done, *args, **kwargs):
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:
            goal_achieved = unwind_dict_values(goal["achieved"])
            goal_desired = unwind_dict_values(goal["desired"])

            goal_achieved = np.array(
                [np.interp(value, [-1, 1], limits) for value, limits in zip(goal_achieved, self.limits)])
            goal_desired = np.array(
                [np.interp(value, [-1, 1], limits) for value, limits in zip(goal_desired, self.limits)])

            reward = np.exp(-1 * np.linalg.norm(goal_achieved - goal_desired)) - 1
            reward /= self.max_steps

        return reward

    def reset(self, desired_state=None, desired_goal=None, robot=None, state_robot=None, force=False):

        super(TaskReach, self).reset()

        if desired_goal is None:
            desired_goal = {}

        def complete_state(state_dict, space):
            for key in space:
                if key not in state_dict:
                    state_dict[key] = space[key].sample()
                elif type(space[key]) == spaces.Dict:
                    state_dict[key] = complete_state(state_dict[key], space[key])
            return state_dict

        desired_goal = complete_state(desired_goal, self.goal_space)

        while True:
            desired_target_position = np.array([np.interp(value, [-1, 1], limits) for value, limits in
                                                zip(desired_goal["desired"]["position"], self.limits)])

            self.bullet_client.resetBasePositionAndOrientation(self.target, desired_target_position + self.offset,
                                                               [0, 0, 0, 1])
            self.bullet_client.stepSimulation()

            if robot and not force:
                contact_points = self.bullet_client.getContactPoints(robot.model_id, self.target)
            else:
                contact_points = False

            if not contact_points and np.linalg.norm(desired_target_position) < 0.8:
                break
            else:
                # try reset again with random state
                desired_goal = self.goal_space.sample()

        state, goal, done, info = self.get_state(state_robot, robot)

        return state, goal, info

    def get_state(self, state_robot=None, robot=None):
        position_achieved = None

        if state_robot is not None and robot is not None:
            position_achieved, _ = robot.get_tcp_pose()
            position_achieved = np.array(
                [np.interp(value, limits, [-1, 1]) for value, limits in zip(position_achieved, self.limits)])

        position_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_desired = np.array(position_desired)
        position_desired -= self.offset

        position_desired = np.array(
            [np.interp(value, limits, [-1, 1]) for value, limits in zip(position_desired, self.limits)])

        state = {}

        goal = {
            'achieved': {
                "position": position_achieved,
            },
            'desired': {
                "position": position_desired,
            },
        }

        done = self.step_counter >= self.max_steps

        info = {
            "steps": self.step_counter
        }

        return state, goal, done, info


if __name__ == "__main__":
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = TaskReach(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
