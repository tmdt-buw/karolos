from environments.robots.panda import Panda
from environments.tasks.reach import Reach
import gym
import numpy as np
from environments.environment_robot_task import Environment
import multiprocessing as mp


class EnvironmentWrapper(gym.Env):

    def __init__(self, **kwargs_env):
        env_cls = Environment

        self.action_space_ = None
        self.observation_space_ = None

        pipe_wrapper, pipe_env = mp.Pipe()

        self.pipe = pipe_wrapper

        p = mp.Process(target=self.run, args=(env_cls, kwargs_env, pipe_env),
                       daemon=True)
        p.start()

    def run(self, env_cls, kwargs_env, pipe):

        env = env_cls(**kwargs_env)

        while True:

            func, params = pipe.recv()

            # print(func)

            if func == "close":
                break
            elif func == "reset":
                pipe.send(env.reset())
            elif func == "step":
                pipe.send(env.step(params))
            elif func == "render":
                pipe.send(env.render(params))
            elif func == "action space":
                # for i in range(3):
                #     print(env.action_space.sample())
                pipe.send(env.action_space)
            elif func == "observation space":
                pipe.send(env.observation_space)
            else:
                raise NotImplementedError(func)

    def reset(self):
        """Reset the environment and return new state
        """

        self.pipe.send(["reset", None])

        observation = self.pipe.recv()

        return observation

    def render(self, mode='human'):
        self.pipe.send(["render", mode])
        self.pipe.recv()

    def step(self, action):

        self.pipe.send(["step", action])

        observation, reward, done, info = self.pipe.recv()

        info = {"info": 1} # EnvInfo()

        observation = np.array(observation)
        reward = np.array(reward)
        done = np.array(done).astype(float)

        # EnvStep = namedtuple("EnvStep", ["observation", "reward", "done", "env_info"])

        return observation, reward, done, info
        # return EnvStep(observation, reward, done, info)

    @property
    def action_space(self):

        if self.action_space_ is None:
            self.pipe.send(["action space", None])
            self.action_space_ = self.pipe.recv()

        return self.action_space_

    @property
    def observation_space(self):
        if self.observation_space_ is None:
            self.pipe.send(["observation space", None])
            self.observation_space_ = self.pipe.recv()

        return self.observation_space_

    def test_compatability(self):
        # todo check if task can be completed with robot (dimensionalities)
        ...


if __name__ == "__main__":

    task = Reach
    robot = Panda

    env_kwargs1 = {
        "task_cls": task,
        "robot_cls": robot,
        "render": True,
        "kwargs_task": {"dof": 2},
        "kwargs_robot": {"dof": 3}
    }

    env_kwargs2 = {
        "task_cls": task,
        "robot_cls": robot,
        "render": False,
        "kwargs_task": {"dof": 1},
        "kwargs_robot": {"dof": 2}
    }

    env1 = EnvironmentWrapper(Environment, env_kwargs1)
    env2 = EnvironmentWrapper(Environment, env_kwargs2)

    while True:

        done1 = False
        done2 = False

        obs1 = env1.reset()
        obs2 = env2.reset()

        step = 0

        while not done1 or not done2:
            action1 = env1.action_space.sample()
            obs1, reward1, done1, info1 = env1.step(action1)

            action2 = env2.action_space.sample()
            obs2, reward2, done2, info2 = env2.step(action2)

            step += 1
