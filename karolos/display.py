import json
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data as pd

from karolos.utils import unwind_dict_values

from karolos.agents import get_agent
from karolos.environments import get_env

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pd.getDataPath())

# todo make experiment a parameter
# experiments = os.listdir("results")

#
experiment_folder = osp.join("./environments/shell_environment/results", "clip_1_1")

# diagonal escape
# experiment_folder = osp.join("./environments/shell_environment/results", "clip_4_5")

# sliding trajectory
experiment_folder = osp.join("./environments/shell_environment/results", "clip_4_6")


print(experiment_folder)
print(os.listdir(experiment_folder))

with open(osp.join(experiment_folder, "config.json")) as f:
    config = json.load(f)

env_config = config.pop("env_config")
agent_config = config.pop("agent_config")

env_config["render"] = True

print(env_config)
print(agent_config)

env = get_env(env_config)
agent = get_agent(agent_config, env.observation_space,
                  env.action_space, reward_function=lambda x: 0, experiment_dir=".")

models_folder = osp.join(experiment_folder, "models")

models_folder = osp.join(models_folder, max(os.listdir(models_folder)))
print(models_folder)

agent.load(models_folder)

while True:
    state, goal_info = env.reset()
    done = False

    actions = []

    while not done:
        # print(state)

        state = unwind_dict_values(state)

        action = agent.predict([state], deterministic=True)[0]

        actions.append(action.copy())

        next_state, goal_info, done = env.step(action)

        done |= env.success_criterion(goal_info)

        # if done:

        print(action)

        state = next_state

    print(env.reward_function(goal_info, done))

    # for action in reversed(actions):
    #     # rotation_q = p.getQuaternionFromEuler(action[-3:] * env.max_rotation)
    #     # _, rotation_q_inv = p.invertTransform([0,0,0], rotation_q)
    #
    #     action[:3] = -action[:3]
    #     action[3:] = -action[3:]
    #     print(action)
    #     # action[-3:] = np.array(p.getEulerFromQuaternion(rotation_q_inv)) / env.max_rotation
    #
    #     env.step(action)

    print()

