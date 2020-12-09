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

    def reset(self, params=None, domain_randomization=None):
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

        achieved_goal, desired_goal, done = \
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
        observation_task = self.task.step(self.robot)

        achieved_goal, desired_goal, done = \
            self.task.get_status(observation_robot)

        state = {
            'task': observation_task,
            'robot': observation_robot
        }

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal,
        }

        return state, goal, done

    def _domain_randomize(self):

        self.robot.randomize()
        self.task.randomize()

    def _domain_standard(self):

        self.robot.standard()
        self.task.standard()

if __name__ == "__main__":

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

    while False:
        observation, goal = env.reset()
        goal_positiom = np.random.random(9)

        for tt in np.arange(
                2 / p.getPhysicsEngineParameters()["fixedTimeStep"]):

            positions_joints = observation["robot"]["joint_positions"]
            velocities_joints = observation["robot"]["joint_velocities"]

            action = (goal_positiom - positions_joints) * (
                    1 + 2 * (1 - np.abs(velocities_joints)))

            action = np.clip(action, env.action_space.low,
                             env.action_space.high)

            observation, goal, done = env.step(action)

            if np.all((goal_positiom - positions_joints) < 0.1):
                print(tt)
                break

    time_step = env_kwargs["robot_config"]["sim_time"]

    actions = []

    action = np.zeros_like(env.action_space.sample())
    action[1] = 1
    action[-1] = 1

    for _ in np.arange(.2 / time_step):
        actions.append(action)

    action = np.zeros_like(env.action_space.sample())
    action[-1] = -1

    for _ in np.arange(.2 / time_step):
        actions.append(action)

    action = np.zeros_like(env.action_space.sample())
    action[1] = -1
    action[-1] = -1

    for _ in np.arange(.5 / time_step):
        actions.append(action)

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    while True:
        desired_state = {"robot": [0, 0.2, 0, -0.3, 0, .3, .2, 1, 1],
                         "task": np.array([.7, 0, -.9, -.5, .0, .3])}

        # observation = env.reset(desired_state=desired_state)
        obs = env.reset(params=None)

        for action in actions:
            observation, goal, done = env.step(action)

            print(np.linalg.norm(
                goal["desired"]["position"] - goal["achieved"]["position"]))
