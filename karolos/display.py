import json
import os
import os.path as osp

from utils import unwind_dict_values

from agents import get_agent
from environments import get_env

# todo make experiment a parameter
print(os.listdir("."))
experiment_folder = osp.join("results/train_sac_panda_reach/20230706-125802/")

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
agent = get_agent(agent_config, env.state_space, env.goal_space,
                  env.action_space, reward_function=lambda x: 0, experiment_dir=".")

models_folder = osp.join(experiment_folder, "models")
models_folder = osp.join(models_folder, 'final')
print(models_folder)

agent.load(models_folder)

while True:
    state, goal, info = env.reset()
    done = False

    actions = []

    while not done:
        # print(state)

        state = unwind_dict_values(state)
        goal = unwind_dict_values(goal)

        action, = agent.predict([state], [goal], deterministic=True)
        action = action[0] # convert batch of size 1 to single action

        actions.append(action.copy())

        next_state, goal, done, info = env.step(action)

        done |= env.success_criterion(goal)

        state = next_state

    print(f"Last episode reward: {env.reward_function(goal, done)}")

    print()

