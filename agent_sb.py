import datetime
from stable_baselines import SAC
from stable_baselines.sac import LnMlpPolicy
import tensorflow as tf
import pybullet as p
import numpy as np

from environment import Environment
from tasks.reach import Reach
from robots.panda import Panda

experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"./logs/sb/{experiment_id}"

modeldir = f"./models/sb/{experiment_id}"


import os

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

task = Reach
robot = Panda

env_kwargs = {
    "task": task,
    "robot": robot,
    "render": False,
    "kwargs_task": {"dof": 2},
    "kwargs_robot": {"dof": 3}
}

env = Environment(p, **env_kwargs)

model = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log=logdir)

# model.load("model_20200126-145502_490000.tf")

test_interval = 1_000
save_interval = 10_000


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

model.learn(total_timesteps=1_000_000, reset_num_timesteps=False, callback=callback)
