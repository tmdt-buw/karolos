import json
import os
import os.path as osp
import pybullet as p
import pybullet_data as pd
from agents import get_agent
from environments import get_env
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# todo make experiment a parameter
experiments = os.listdir("results")
experiment_folder = osp.join("results", max(experiments))

print(experiment_folder)

with open(osp.join(experiment_folder, "config.json")) as f:
    config = json.load(f)

env_config = config.pop("env_config")
agent_config = config.pop("agent_config")

env_config["bullet_client"] = p

env_config["robot_config"]["sim_time"] = 0.1
env_config["task_config"]["max_steps"] = 100
print(env_config)

env = get_env(env_config)()
agent = get_agent(agent_config, env.observation_space,
                  env.action_space)

models_folder = osp.join(experiment_folder, "models")

agent.load(osp.join(models_folder, max(os.listdir(models_folder))))

while True:

    observation = env.reset()

    done = False

    while not done:


        observation = [observation['state']['agent_state']]

        # print(observation)
        action = agent.predict(observation, deterministic=True)
        
        action = action[0]

        observation, rewards, done = env.step(action)

        # print(np.linalg.norm(observation['state']['task']), rewards)

