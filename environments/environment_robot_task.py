from environments.robots import get_robot
from environments.tasks import get_task
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
import pybullet_utils.bullet_client as bc


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

        self.observation_dict = spaces.Dict({
            'state': spaces.Dict({
                'task': self.task.observation_space,
                'robot': self.robot.observation_space,
                'agent_state': spaces.Box(-1, 1, shape=tuple(
                    np.array(self.task.observation_space.shape) +
                    np.array(self.robot.observation_space.shape)))
            }),
            'goal': spaces.Dict({
                'task': self.task.observation_space,
                'achieved_goal': self.task.observation_space,
                'desired_goal': self.task.observation_space,
                'reached': spaces.MultiBinary(1)
            }),
            'success': spaces.Dict({
                'robot': spaces.MultiBinary(1),
                'task': spaces.MultiBinary(1)
            }),
            'her': spaces.Dict({
                "achieved_goal": spaces.MultiBinary(1),
                "reward": spaces.Box(-1, 1, shape=(1,)),
                "done": spaces.MultiBinary(1)
            })
        })

        self.observation_space = tuple(
            np.array(self.task.observation_space.shape) +
            np.array(self.robot.observation_space.shape))
        self.observation_space = spaces.Box(-1, 1,
                                            shape=self.observation_space)

    def reset(self, desired_state=None):
        """Reset the environment and return new state
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

        observation_state = {
            'state': {
                'task': observation_task,
                'robot': observation_robot,
                'agent_state': np.concatenate(
                    (observation_robot, observation_task))
            }
        }

        return observation_state

    def render(self, mode='human'):
        ...

    def step(self, action):
        observation_robot = self.robot.step(action)
        observation_task = self.task.step(self.robot)

        observation = {}

        achieved_goal, desired_goal = self.task.get_goals(self.robot)

        observation["goal"] = {
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

        reward, done, goal_reached = self.task.compute_reward(achieved_goal,
                                                              desired_goal)
        # done, goal_reached = self.task.compute_done(achieved_goal,
        #                                             desired_goal)

        observation["goal"]["reached"] = goal_reached

        # todo make generic by merging task observation and achived goal
        
        observation["state"] = {
            'task': observation_task,
            'robot': observation_robot,
            'agent_state': np.concatenate(
                (observation_robot, observation_task))
        }

        return observation, reward, done


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
    num=10
    while True:

        done = False

        desired_state = {"robot": [-1, 1, 1, 1, 1, 1, 1, .01, .01],
                         "task": np.array([.5, .5, .5, .5, .5, .5, 0])}


        # desired_state = {"robot": env1.robot.observation_space.sample(),
        #                  "task": env1.task.observation_space.sample()}

        obs = env1.reset(desired_state=None)


        action1 = env1.action_space.sample()

        for i in range(num):
            obs, reward, done = env1.step(action1)
            print(obs['state']['robot'])
            input()

        input('1')
        action1[0] = -1
        for i in range(num):
            obs, reward, done = env1.step(action1)
        input('2')