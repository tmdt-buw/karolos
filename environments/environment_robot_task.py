import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from gym import spaces

from environments.robots import get_robot
from environments.tasks import get_task


class Environment(gym.Env):

    def __init__(self, task_config, robot_config, render=False,
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

        bullet_client.loadURDF("plane.urdf")

        self.bullet_client = bullet_client

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

        self.action_space = self.robot.action_space

        self.observation_space = spaces.Dict({
            'robot': self.robot.observation_space,
            'task': self.task.observation_space,
            'goal': self.task.goal_space
        })

        self.reward_function = self.task.reward_function
        self.success_criterion = self.task.success_criterion

    def reset(self, desired_state=None):
        """
        Reset the environment and return new state
        """

        try:
            if desired_state is not None:
                observation_robot = self.robot.reset(desired_state["robot"])
                observation_task = self.task.reset(self.robot,
                                                   desired_state["task"])
            else:
                observation_robot = self.robot.reset()
                observation_task = self.task.reset(self.robot)

        except AssertionError as e:
            return e

        state = {
            'task': observation_task,
            'robot': observation_robot
        }

        achieved_goal, desired_goal, _ = \
            self.task.get_status(observation_robot)

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal
        }

        return state, goal

    def render(self, mode='human'):
        ...

    def step(self, action):
        observation_robot = self.robot.step(action)
        observation_task, achieved_goal, desired_goal, done = self.task.step(observation_robot)

        state = {
            'task': observation_task,
            'robot': observation_robot
        }

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal,
        }

        return state, goal, done


if __name__ == "__main__":
    import time

    env_kwargs = {
        "render": True,
        "task_config": {
            "name": "pick_place",
            "max_steps": 25
        },
        "robot_config": {
            "name": "panda",
            "scale": .1,
            "sim_time": .1
        }
    }

    env = Environment(**env_kwargs)

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )
    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    action = np.zeros_like(env.action_space.sample())

    while True:
        obs = env.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)

            observation, goal, done = env.step(action)

            reward = env.reward_function(False, goal)
