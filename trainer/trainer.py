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

    def similar_config(self, resdir, conf):

        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        with open(resdir + '/config.json', 'r+') as fp:
            exp_config = json.load(fp)

        if not ordered(exp_config) == ordered(conf):
            return False
        else:
            return True

    def process_responses(self, env_responses, train: bool):

        requests = []
        results_episodes = []

        for env_id, response in env_responses:
            func, data = response

            if func == "reset":
                self.states[env_id] = data
                self.actions.pop(env_id, None)
                self.episodic_reward[env_id] = []
            elif func == "step":
                state, reward, done = data

                self.writer.add_scalar(
                    f'{"train" if train else "test"} reward step',
                    reward,
                    sum(self.steps.values()) + 1
                )

                self.episodic_reward[env_id].append(reward)
                previous_state = self.states.pop(env_id, None)
                if previous_state is not None:
                    action = self.actions.pop(env_id)

                    experience = (previous_state['state']['agent_state'],
                                  action, reward,
                                  state['state']['agent_state'], done)
                    self.agent.add_experience([experience])

                    if training_config[
                        "use_hindsight_experience_replay"] and "her" in state:
                        her_goal = state["her"]["achieved_goal"]

                        her_previous_state = previous_state['state'][
                            'agent_state'].copy()
                        her_previous_state[-len(her_goal):] = her_goal

                        her_reward = state["her"]["reward"]

                        her_state = state['state']['agent_state'].copy()
                        her_state[-len(her_goal):] = her_goal

                        her_done = state["her"]["done"]

                        experience = (her_previous_state,
                                      action, her_reward,
                                      her_state, her_done)
                        self.agent.add_experience([experience])

                if done:
                    episode_reward = sum(self.episodic_reward[env_id])

                    self.writer.add_scalar(
                        f'{"train" if train else "test"} reward episode',
                        episode_reward,
                        sum(self.steps.values()) + 1
                    )

                    results_episodes.append(state["goal"]["reached"])

                    requests.append((env_id, "reset", None))
                else:
                    self.states[env_id] = state
                self.steps[env_id] += 1
            else:
                raise NotImplementedError(
                    f"Undefined behavior for {env_id} | {response}")

        required_predictions = list(
            set(self.states.keys()) - set(self.actions.keys()))

        if required_predictions:
            observations = [self.states[env_id]['state']['agent_state'] for
                            env_id in
                            required_predictions]
            observations = np.stack(observations)

            predictions = self.agent.predict(observations,
                                             deterministic=not train)

            for env_id, prediction in zip(required_predictions,
                                          predictions):
                self.actions[env_id] = prediction
                requests.append((env_id, "step", prediction))

        return requests, results_episodes

    def __init__(self, training_config):

        results_dir = training_config.pop('results_dir')
        reload_agent = training_config.pop('reload_previous_agent')

        # get environment
        env_config = training_config["env_config"]

        nb_envs = env_config["nb_envs"]
        env_orchestrator = Orchestrator(env_config, nb_envs)

        # get agents
        agent_config = training_config["agent_config"]

        # add action and state spaces to config
        self.agent = get_agent(agent_config,
                          env_orchestrator.observation_space,
                          env_orchestrator.action_space)

        models_dir = osp.join(results_dir, "models")

        if not osp.exists(results_dir):
            # first run of experiment, keep new agent, save config
            os.makedirs(results_dir)

            with open(osp.join(results_dir, 'config.json'), 'w') as f:
                json.dump(training_config, f)
            self.agent.save(models_dir)
        else:
            # experiment exists, check similar configs, if not similar make new results_dir_MMDDHHMM
            # if similar, check reload_agent option:
            # if true->reload previous agent, if false->make new results_dir_MMDDHHMM

            print('\n################################')

            if not self.similar_config(results_dir, training_config):
                print('Experiment config and given config do not match')
                results_dir = osp.join(results_dir,
                                       datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))

                os.makedirs(results_dir)
                models_dir = osp.join(results_dir, "models")

                print('New directory for this experiment:', results_dir)
                print('Loading new Agent')

                with open(osp.join(results_dir, 'config.json'), 'w') as f:
                    json.dump(training_config, f)
                self.agent.save(models_dir)
            else:
                if reload_agent:
                    print('Re-loading previous agent from', models_dir)
                    self.agent.load(models_dir, train_mode=True)

                else:
                    results_dir = osp.join(results_dir,
                                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

                    os.makedirs(results_dir)
                    models_dir = osp.join(results_dir, "models")
                    print('New directory for this experiment:', results_dir)
                    print('Loading new Agent')
                    with open(osp.join(results_dir, 'config.json'), 'w') as f:
                        json.dump(training_config, f)
                    self.agent.save(models_dir)
            print('################################\n')

        log_dir = results_dir

        if not osp.exists(models_dir):
            os.makedirs(models_dir)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir)

        # reset all
        env_responses = env_orchestrator.reset_all()

        self.steps = defaultdict(int)

        self.states = {}
        self.actions = {}
        self.episodic_reward = {}

        nb_tests = training_config["nb_tests"]
        test_interval = training_config["test_interval"]
        save_interval = training_config["save_interval_steps"]
        next_test_timestep = 0

        best_success_ratio = 0.5

        assert nb_tests >= nb_envs

        pbar = tqdm(total=training_config["total_timesteps"])

        while sum(self.steps.values()) < training_config["total_timesteps"]:

            next_save_timestep = (sum(
                self.steps.values()) // save_interval + 1) * save_interval

            # Save
            if sum(self.steps.values()) >= next_save_timestep:
                self.agent.save(models_dir)
                print("\nsaved agent in", models_dir, '\n')

            # Test
            if sum(self.steps.values()) >= next_test_timestep:

                next_test_timestep = \
                    (sum(
                        self.steps.values()) // test_interval + 1) * test_interval

                # reset all
                env_responses = env_orchestrator.reset_all()

                # subtract tests already launched in each environment
                tests_to_run = nb_tests - nb_envs
                concluded_tests = []

                while len(concluded_tests) < nb_tests:

                    requests, results_episodes = self.process_responses(
                        env_responses, train=False)
                    concluded_tests += results_episodes

                    for ii in reversed(range(len(requests))):
                        if requests[ii][1] == "reset":
                            if tests_to_run:
                                tests_to_run -= 1
                            else:
                                del requests[ii]

                    env_responses = env_orchestrator.send_receive(requests)

                # evaluate test
                success_ratio = np.mean(concluded_tests)

                self.writer.add_scalar('test success ratio',
                                       success_ratio,
                                       sum(self.steps.values()) + 1
                                       )

                if success_ratio >= best_success_ratio:
                    best_success_ratio = success_ratio
                    self.agent.save(os.path.join(models_dir,
                                                 f"{sum(self.steps.values()) + 1}_"
                                                 f"{success_ratio:.3f}"))

                # reset all
                env_responses = env_orchestrator.reset_all()

            # Train
            requests, _ = self.process_responses(env_responses, train=True)

            env_responses = env_orchestrator.send_receive(requests)

            self.agent.learn()

            pbar.update(sum(self.steps.values()) - pbar.n)
            pbar.refresh()


if __name__ == "__main__":
    results_dir = osp.join(os.path.dirname(os.path.abspath(__file__)),"../results")
    #res_dir = osp.join(res_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_dir = osp.join(results_dir, 'test')

    training_config = {
        "algorithm": "SAC",
        "test_interval": 500_000,
        "nb_tests": 100,
        "total_timesteps": 25_000_000,
        "save_interval_steps": 1_000_000,
        "results_dir": results_dir,
        "reload_previous_agent": False,
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
                            "max_steps": 100
                            },
            "robot_config": {
                "name": "pandas",
                "dof": 3,
                "sim_time": .1,
                "scale": .1
            }
        }
    }

    trainer = Trainer(training_config)
