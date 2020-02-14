import datetime
from stable_baselines import SAC
# from stable_baselines.ddpg import LnMlpPolicy
from stable_baselines.sac import LnMlpPolicy
from stable_baselines.common import set_global_seeds
import tensorflow as tf
import numpy as np
from environments.environment_robot_task import Environment
from tasks.reach import Reach
from robots.panda import Panda
from multiprocessing import cpu_count
import os


def make_env(env_kwargs, rank, seed=0):
    """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

    def _init():

        env = Environment(**env_kwargs)
        return env

    set_global_seeds(seed)
    return _init

def callback(locals_, globals_):

    if locals_["step"] % save_interval == 0:

        model.save(os.path.join(modeldir, f"{locals_['step']}.tf"))

    if locals_["step"] % test_interval == 0:
        episode_rewards = []

        for tt in range(10):
            episode_reward = 0
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='test_reward', simple_value=np.mean(episode_rewards))])
        locals_['writer'].add_summary(summary, locals_["step"])

    return True


if __name__ == "__main__":
    experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"./logs/sb/{experiment_id}"

    modeldir = f"./models/sb/{experiment_id}"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    task = Reach
    robot = Panda

    env_kwargs = {
        "task_cls": task,
        "robot_cls": robot,
        "render": False,
        "kwargs_task": {"dof": 1},
        "kwargs_robot": {"dof": 2}
    }

    num_cpu = cpu_count()

    # env = SubprocVecEnv([make_env(env_kwargs, i) for i in range(num_cpu)])
    env = Environment(**env_kwargs)

    # model = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log=logdir)

    test_interval = 1_000
    save_interval = 10_000

    model.learn(total_timesteps=1_000_000, reset_num_timesteps=False, callback=callback)
