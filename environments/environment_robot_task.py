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

            time_step = 1. / 60.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        self.bullet_client = bullet_client

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

        self.action_space = self.robot.action_space

        self.observation_space = spaces.Dict({
            'robot': self.robot.observation_space,
            'task': self.task.observation_space,
            'goal': self.task.goal_space
        })

    def reset(self, params=None):
        """Reset the environment and return new state
        """

        desired_state = params[0]
        domain_randomization = params[1]

        # call domain randomization before reset
        if domain_randomization:
            self._domain_randomize()
        else:
            # for test runs
            self._domain_standard()

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

        achieved_goal, desired_goal, goal_reached, done = \
            self.task.get_status(self.robot)

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal,
            'reached': goal_reached
        }

        return state, goal

    def render(self, mode='human'):
        ...

    def step(self, action):
        observation_robot = self.robot.step(action)
        observation_task = self.task.step(self.robot)

        achieved_goal, desired_goal, goal_reached, done = \
            self.task.get_status(self.robot)

        state = {
            'task': observation_task,
            'robot': observation_robot
        }

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal,
            'reached': goal_reached
        }

        return state, goal, done

    def _domain_randomize(self):

        self.robot.randomize()
        self.task.randomize()

    def _domain_standard(self):

        self.robot.standard()
        self.task.standard()

if __name__ == "__main__":

    env_kwargs1 = {
        "render": True,
        "task_config": {"name": "pick_place",
                        "dof": 3,
                        "only_positive": False,
                        "max_steps": 50,
                        },
        "robot_config": {
            "use_gripper": True,
            "mirror_finger_control": True,
            "name": "panda",
            "dof": 3,
            "sim_time": .1,
            "scale": .1
        }
    }

    env1 = Environment(**env_kwargs1)

    action1 = np.zeros_like(env1.action_space.sample())
    action1[0] = 1

    action2 = np.zeros_like(env1.action_space.sample())
    action2[0] = -1

    while True:

        done = False

        desired_states = {"robot": [-1, 1, 1, 1, 1, 1, 1, .01, .01],
                         "task": np.array([.5, .5, .5, .5, .5, .5, 0])}


        # desired_state = {"robot": env1.robot.observation_space.sample(),
        #                  "task": env1.task.observation_space.sample()}

        obs = env1.reset(params=None)


        action1 = env1.action_space.sample()

        for i in range(25):
            observation, goal, done = env1.step(action1)

        for i in range(25):
            observation, goal, done = env1.step(action2)
