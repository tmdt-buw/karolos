import sys
import os

sys.path.insert(0, os.path.abspath("."))

from collections import defaultdict
import datetime
from environments.orchestrator import Orchestrator
import json
from multiprocessing import cpu_count
import numpy as np
import os
import os.path as osp
from torch.utils.tensorboard.writer import SummaryWriter
from agents import get_agent

from tqdm import tqdm


class Trainer:

    def get_env(self, env_config):
        base_pkg = env_config.pop("base_pkg")

        assert base_pkg in ["robot-task-rl"]

        if base_pkg == "robot-task-rl":
            from environments.environment_wrapper import Environment

            def env_init():
                env = Environment(**env_config)
                return env
        else:
            raise NotImplementedError(f"Unknown base package: {base_pkg}")

        return env_init

    def check_experiment(self, directory, config):
        """
        1. check if config eq config in result dir, if not -> overwrite or rename experiment
        2. return result dir
        """

        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        print('Found experiment folder with same name')

        with open(directory + '/config.json', 'r+') as fp:
            exp_config = json.load(fp)

        if not ordered(exp_config) == ordered(config):
            print('Experiment config and given config do not match:')
            print(ordered(exp_config))
            print(ordered(config))
            dec = input('Quit <any> or New Experiment <n> ?')
            if dec == 'n' or dec == 'N':
                new_exp = osp.join("..", exp_config[
                    'results_dir']) + datetime.datetime.now().strftime(
                    "_%Y%m%d-%H%M%S")
                print('adding datetime to Experiment folder: ', new_exp, '\n')
                return new_exp, config, True
            else:
                exit('quitting')
        else:
            print(
                'Experiment config matches given config, loading experiment config')
            return osp.join("..", exp_config['results_dir']), exp_config, False

    def __init__(self, training_config):

        assert training_config["base_pkg"] in ["stable-baselines"]

        # create results directories
        try:
            results_dir = osp.join('results/',
                                   training_config.pop('experiment_name'))
            training_config['results_dir'] = results_dir
        except KeyError:
            results_dir = training_config.pop('results_dir')

        results_dir = osp.join(".", results_dir)
        load_new_agent = True

        if not osp.exists(results_dir):
            os.makedirs(results_dir)
        else:
            results_dir, training_config, load_new_agent = \
                self.check_experiment(results_dir, training_config)
            if load_new_agent:
                os.makedirs(results_dir)

        models_dir = osp.join(results_dir, "models")
        log_dir = results_dir

        if not osp.exists(models_dir):
            os.makedirs(models_dir)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir)

        if load_new_agent:
            with open(osp.join(results_dir, 'config.json'), 'w') as f:
                json.dump(training_config, f)

        # get environment
        env_config = training_config.pop("env_config")

        nb_envs = env_config.pop("nb_envs", 1)
        env_orchestrator = Orchestrator(env_config, nb_envs)

        # get agents
        agent_config = training_config.pop("agent_config")

        # add action and state spaces to config
        agent = get_agent(agent_config,
                          env_orchestrator.observation_space,
                          env_orchestrator.action_space)

        if not load_new_agent:
            print('\nloading previous agent from', models_dir, '\n')
            agent.load(models_dir, train_mode=True)

        # reset all
        env_responses = env_orchestrator.reset_all()

        steps = defaultdict(int)

        states = {}
        actions = {}
        episodic_reward = {}

        requests = []

        nb_tests = training_config["nb_tests"]
        test_interval = training_config["test_interval"]
        save_interval = training_config["save_interval_steps"]
        next_test_timestep = 0

        best_success_ratio = 0.5

        assert nb_tests >= nb_envs

        pbar = tqdm(total=training_config["total_timesteps"])

        while sum(steps.values()) < training_config["total_timesteps"]:

            next_save_timestep = (sum(
                steps.values()) // save_interval + 1) * save_interval

            # Save
            if sum(steps.values()) >= next_save_timestep:
                agent.save(models_dir)
                print("\nsaved agent in", models_dir, '\n')

            # Test
            if sum(steps.values()) >= next_test_timestep:

                next_test_timestep = \
                    (sum(steps.values()) // test_interval + 1) * test_interval

                # reset all
                env_responses = env_orchestrator.reset_all()

                results_episodes = []
                tests_launched = nb_envs

                while len(results_episodes) < nb_tests:

                    for env_id, response in env_responses:
                        func, data = response

                        if func == "reset":
                            states[env_id] = data
                            actions.pop(env_id, None)
                            episodic_reward[env_id] = []
                        elif func == "step":
                            state, reward, done = data

                            self.writer.add_scalar('test reward step',
                                                   reward,
                                                   sum(steps.values()) + 1
                                                   )

                            episodic_reward[env_id].append(reward)
                            previous_state = states.pop(env_id, None)
                            if previous_state is not None:

                                action = actions.pop(env_id)

                                experience = (previous_state['state']['agent_state'],
                                              action, reward,
                                              state['state']['agent_state'],
                                              done)
                                agent.add_experience([experience])

                                if training_config["use_hindsight_experience_replay"] and "her" in state:
                                    her_goal = state["her"]["achieved_goal"]

                                    her_previous_state = previous_state['state']['agent_state'].copy()
                                    her_previous_state[
                                    -len(her_goal):] = her_goal

                                    her_reward = state["her"]["reward"]

                                    her_state = state['state']['agent_state'].copy()
                                    her_state[-len(her_goal):] = her_goal

                                    her_done = state["her"]["done"]

                                    experience = (her_previous_state,
                                                  action, her_reward,
                                                  her_state, her_done)
                                    agent.add_experience([experience])

                            if done:
                                episode_reward = sum(episodic_reward[env_id])

                                self.writer.add_scalar('test reward episode',
                                                       episode_reward,
                                                       sum(steps.values()) + 1
                                                       )

                                results_episodes.append(state["goal"]["reached"])
                                if tests_launched < nb_tests:
                                    requests.append((env_id, "reset", None))
                                    tests_launched += 1
                            else:
                                states[env_id] = state
                            steps[env_id] += 1
                        else:
                            raise NotImplementedError(
                                f"Undefined behavior for {env_id}"
                                f" | {response}")

                    required_predictions = list(
                        set(states.keys()) - set(actions.keys()))

                    if required_predictions:
                        observations = [states[env_id]['state']['agent_state']
                                            for env_id in
                                            required_predictions]

                        observations = np.stack(observations)

                        # predictions = agent.random_action(observations)
                        predictions = agent.predict(observations,
                                                    deterministic=True)

                        for env_id, prediction in zip(required_predictions,
                                                      predictions):
                            actions[env_id] = prediction
                            requests.append((env_id, "step", prediction))

                    env_responses = env_orchestrator.send_receive(requests)
                    requests = []

                success_ratio = sum(results_episodes) / len(results_episodes)

                self.writer.add_scalar('test success ratio',
                                       success_ratio,
                                       sum(steps.values()) + 1
                                       )

                if success_ratio >= best_success_ratio:
                    best_success_ratio = success_ratio
                    agent.save(os.path.join(models_dir,
                                            f"{sum(steps.values()) + 1}_"
                                            f"{success_ratio:.3f}"))

                # reset all
                env_responses = env_orchestrator.reset_all()

            # Train

            for env_id, response in env_responses:
                func, data = response

                if func == "reset":
                    states[env_id] = data
                    actions.pop(env_id, None)
                    episodic_reward[env_id] = []
                elif func == "step":
                    state, reward, done = data

                    self.writer.add_scalar('train reward step',
                                           reward,
                                           sum(steps.values()) + 1
                                           )

                    episodic_reward[env_id].append(reward)
                    previous_state = states.pop(env_id, None)
                    if previous_state is not None:
                        action = actions.pop(env_id)

                        experience = (previous_state['state']['agent_state'],
                                      action, reward,
                                      state['state']['agent_state'], done)
                        agent.add_experience([experience])

                        if training_config["use_hindsight_experience_replay"] and "her" in state:
                            her_goal = state["her"]["achieved_goal"]

                            her_previous_state = previous_state['state']['agent_state'].copy()
                            her_previous_state[-len(her_goal):] = her_goal

                            her_reward = state["her"]["reward"]

                            her_state = state['state']['agent_state'].copy()
                            her_state[-len(her_goal):] = her_goal

                            her_done = state["her"]["done"]

                            experience = (her_previous_state,
                                          action, her_reward,
                                          her_state, her_done)
                            agent.add_experience([experience])

                    if done:
                        episode_reward = sum(episodic_reward[env_id])

                        self.writer.add_scalar('train reward episode',
                                               episode_reward,
                                               sum(steps.values()) + 1
                                               )

                        requests.append((env_id, "reset", None))
                    else:
                        states[env_id] = state
                    steps[env_id] += 1
                else:
                    raise NotImplementedError(
                        f"Undefined behavior for {env_id} | {response}")

            required_predictions = list(
                set(states.keys()) - set(actions.keys()))

            if required_predictions:
                observations = [states[env_id]['state']['agent_state'] for env_id in
                                required_predictions]
                observations = np.stack(observations)

                predictions = agent.predict(observations, deterministic=False)

                for env_id, prediction in zip(required_predictions,
                                              predictions):
                    actions[env_id] = prediction
                    requests.append((env_id, "step", prediction))

            env_responses = env_orchestrator.send_receive(requests)
            requests = []

            agent.learn()

            pbar.update(sum(steps.values()) - pbar.n)
            pbar.refresh()


if __name__ == "__main__":
    results_dir = osp.join("results",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    training_config = {
        "base_pkg": "stable-baselines",
        "algorithm": "SAC",
        "test_interval": 500_000,
        "nb_tests": 100,
        "total_timesteps": 25_000_000,
        "save_interval_steps": 1_000_000,
        "results_dir": results_dir,
        "use_hindsight_experience_replay": False,
        "agent_config": {
            "algorithm": "sac",
            "soft_q_lr": 0.0005,
            "policy_lr": 0.0005,
            "alpha": 1,
            "alpha_lr": 0.0005,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "gamma": 0.95,
            "auto_entropy": True,
            "memory_size": 100_000,
            "tau": 0.0025,
            "hidden_dim": 25,
            "hidden_layers": 4,
            "seed": 192
        },
        "env_config": {
            "nb_envs": cpu_count(),
            "base_pkg": "robot-task-rl",
            "render": False,
            "task_config": {"name": "reach",
                            "dof": 3,
                            "only_positive": False,
                            "sparse_reward": False,
                            "max_steps": 10
                            },
            "robot_config": {
                "name": "pandas",
                "dof": 3
            }
        }
    }

    trainer = Trainer(training_config)
