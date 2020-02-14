from stable_baselines import SAC

from environments.environment_robot_task import Environment
from tasks.reach import Reach
from robots.panda import Panda

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
    "kwargs_task": {"dof": 2},
    "kwargs_robot": {"dof": 3}
}

env = Environment(**env_kwargs, bullet_client=p)
# Define and Train the agent

model = SAC.load(f"models/model_20200126-201228_240000.tf")

print(model.get_parameters())

while True:

    obs = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)