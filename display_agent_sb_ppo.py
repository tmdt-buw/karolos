from stable_baselines import SAC, PPO2

from environments.environment_robot_task import Environment
from tasks.reach import Reach
from robots.panda import Panda

import os
import pybullet as p
import pybullet_data as pd

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# Instantiate env
task = Reach
robot = Panda

env_kwargs = {
    "task_cls": task,
    "robot_cls": robot,
    "render": True,
    "kwargs_task": {"dof": 1},
    "kwargs_robot": {"dof": 2}
}

env = Environment(**env_kwargs, bullet_client=p)
# Define and Train the agent

models_folder = "models/sb"

experiments = os.listdir(models_folder)
last_experiment = os.path.join(models_folder, max(experiments))

models = [int(m.split(".")[0]) for m in os.listdir(last_experiment)]
last_model = max(models)

model = PPO2.load(os.path.join(last_experiment, f"{last_model}.tf"))

while True:

    obs = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs)

        obs, rewards, done, info = env.step(action)

        print(action, rewards)