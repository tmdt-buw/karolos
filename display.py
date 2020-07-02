import json
import os
import os.path as osp

import pybullet as p
import pybullet_data as pd

from agents import get_agent
from environments import get_env

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# todo make experiment a parameter
experiments = os.listdir("results")
experiment_folder = osp.join("results", "20200702-143606", "(0.0005, 0.0005)_32_8_5e-05_0.0025")

print(experiment_folder)

with open(osp.join(experiment_folder, "config.json")) as f:
    config = json.load(f)

env_config = config.pop("env_config")
agent_config = config.pop("agent_config")

env_config["bullet_client"] = p

# env_config["robot_config"]["sim_time"] = 0.1
# env_config["task_config"]["max_steps"] = 100
print(env_config)
print(agent_config)

env = get_env(env_config)
agent = get_agent("sac", agent_config, env.observation_space,
                  env.action_space, ".")

models_folder = osp.join(experiment_folder, "models")

# models_folder = osp.join(models_folder, "37003283_0.500")

models_folder = osp.join(models_folder, max(os.listdir(models_folder)))

agent.load(models_folder)

desired_states = [
    {"robot": [0, 0, 0, 0, 0, 0, 0, 0], "task": [.5, .5, 0]},
    {"robot": [0, 0, 0, 0, 0, 0, 0, 0], "task": [.5, -.5, 0]},
    {"robot": [0.5, 0, 0, 0, 0, 0, 0, 0], "task": [.5, .5, 0]}
]

replay = 10

for desired_state in desired_states:

    for _ in range(replay):
        observation = env.reset(desired_state)

        done = False

        while not done:
            observation = [observation['state']['agent_state']]

            # print(observation)
            action = agent.predict(observation, deterministic=True)

            action = action[0]

            observation, reward, done = env.step(action)
