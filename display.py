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
experiment_folder = osp.join("results", max(experiments))

with open(osp.join(experiment_folder, "config.json")) as f:
    config = json.load(f)

env_config = config.pop("env_config")
agent_config = config.pop("agent_config")

env_config["bullet_client"] = p

env = get_env(env_config)()
agent = get_agent(agent_config, env.observation_space,
                  env.action_space)

models_folder = osp.join(experiment_folder, "models")

agent.load(osp.join(models_folder, max(os.listdir(models_folder))))

while True:

    obs = env.reset()

    done = False

    while not done:
        action = agent.predict(obs, deterministic=True)

        obs, rewards, done, info = env.step(action)

        print(action, rewards)
