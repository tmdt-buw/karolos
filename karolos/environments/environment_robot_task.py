import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from gym import spaces

sys.path.append(str(Path(__file__).resolve().parents[0]))
from environment import Environment

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from .environments_robot_task.robots import get_robot
    from .environments_robot_task.tasks import get_task
except:
    from environments_robot_task.robots import get_robot
    from environments_robot_task.tasks import get_task

class EnvironmentRobotTask(Environment):
    """
    A class to combine a robot instance with a task instance
    to define a RL environment
    """
    def __init__(self, robot_config, task_config, render=False,
                 bullet_client=None, **kwargs):
        self.render = render

        self.task_config = task_config
        self.robot_config = robot_config

        if bullet_client is None:
            connection_mode = p.GUI if render else p.DIRECT

            bullet_client = bc.BulletClient(connection_mode)

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 300.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        self.bullet_client = bullet_client

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

        self.action_space = self.robot.action_space

        self.state_space = spaces.Dict({
            'robot': self.robot.state_space,
            'task': self.task.state_space,
        })

        self.goal_space = self.task.goal_space

        self.reward_function = self.task.reward_function
        self.success_criterion = self.task.success_criterion

    def __exit__(self):
        del self.robot
        del self.task

    def reset(self, desired_state=None, desired_goal=None):
        """
        Reset the environment and return new state
        """

        if desired_state is None:
            desired_state = {}

        state_robot = self.robot.reset(desired_state.get("robot"))
        state_task, goal, info = self.task.reset(desired_state.get("task"), desired_goal, self.robot, state_robot)

        state = {
            'robot': state_robot,
            'task': state_task
        }

        return state, goal, info

    def step(self, action):
        state_robot = self.robot.step(action)
        state_task, goal, done, info = self.task.step(state_robot, self.robot)

        state = {
            'robot': state_robot,
            'task': state_task
        }

        return state, goal, done, info
