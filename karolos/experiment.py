import datetime
import json
import logging
import os
import os.path as osp
import random
import sys
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter


from tqdm import tqdm

try:
    from IPython import get_ipython

    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
except:
    pass


sys.path.append(str(Path(__file__).resolve().parent))

from agents import get_agent
from environments.orchestrator import Orchestrator
from utils import unwind_dict_values, set_seed


class Experiment:
    @staticmethod
    def get_initial_state(random: bool, env_id=None):
        """
        Initial state generator, if initial state is None a random initial state is chosen
        The goal (e.g. TCP end point with reach task) can also be specified
        :param random:
        :param env_id:
        :return: initial state or None
        """
        initial_state = None

        return initial_state

    @staticmethod
    def similar_config(config_1, config_2):
        """
        check similarity between config1 and config2
        required when re-loading previous experiments
        :param config_1:
        :param config_2:
        :return: bool if configs are similar
        """

        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        if not ordered(config_1) == ordered(config_2):
            return False
        else:
            return True

    def _save_transition_onpol(self, goal, done, info, env_id):
        # todo remove this and move into ppo agent!
        experience = {
            "states": torch.Tensor(
                unwind_dict_values(self.trajectories[env_id][-3][0])
            ),
            # "actions": torch.Tensor(self.trajectories[env_id]["mdp"][-2]),
            "rewards": torch.as_tensor(
                self.reward_function(goal=goal, done=done, info=info)
            ),
            "terminals": torch.as_tensor(done, dtype=torch.bool),

            # contains values if generalized advantage estimation in ppo is true
            # therefore, agent_specific_info should be defined by the on-pol algorithm
            **self.state_infos[env_id],
        }
        self.agent.add_experiences([experience], env_id)
        self.state_infos[env_id] = None

    def process_responses(self, env_responses, mode: str):
        """
        handle the responses sent back by environments
        and return a request for each environment, e.g. the agents action or a reset command
        :param env_responses: list of tuples (env_id, response) of env reponses
        :param mode: "train" or "test"
        :return: requests to each environment as list of tuples (env_id, request)
        """
        requests = []
        results_episodes = []
        failed_runs = 0

        for env_id, response in env_responses:
            func, data = response

            if func == "reset":
                if type(data) == AssertionError:
                    warnings.warn(f"Resetting the environment resulted in AssertionError: {data}.\n"
                                  f"This might indicate issues, if applicable, in the choice of desired initial states."
                                  f"The environment will be reset again."
                                  )
                    requests.append((env_id, "reset", self.get_initial_state(mode != "test", env_id)))
                    failed_runs += 1
                else:
                    self.trajectories.pop(env_id, None)

                    if mode == "expert":
                        state, goal, info = data
                        if info.get("expert_action", None) is None:
                            requests.append((env_id, "reset", self.get_initial_state(mode != "test", env_id)))
                            failed_runs += 1
                            continue

                    self.trajectories[env_id].append(data)
            elif func == "step":
                state, goal, done, info = data
                self.trajectories[env_id].append((state, goal, info))

                success = self.success_criterion(state=state, goal=goal, done=done, info=info)
                done |= success

                if mode == "expert":
                    if info.get("expert_action", None) is None or (done and not success):
                        requests.append((env_id, "reset", self.get_initial_state(mode != "test", env_id)))
                        self.steps[env_id] -= len(self.trajectories.pop(env_id, [])) // 2 - 1
                        failed_runs += 1
                        continue

                if done:
                    trajectory = self.trajectories.pop(env_id)

                    rewards = self.agent.add_experience_trajectory(trajectory)

                    self.writer.add_scalar(f'{mode} reward episode', sum(rewards), sum(self.steps.values()) + 1)

                    results_episodes.append(self.success_criterion(state=state, goal=goal, done=done, info=info))

                    requests.append((env_id, "reset", self.get_initial_state(mode != "test", env_id)))

                self.steps[env_id] += 1

                total_steps = sum(self.steps.values())
                env_id_load = self.steps[env_id] / total_steps
                env_id_load *= len(self.steps)

                self.writer.add_scalar(f'env_id load/{env_id}', env_id_load, total_steps)
            else:
                raise NotImplementedError(
                    f"Undefined behavior for {env_id} | {response}")

        required_predictions = [env_id for env_id in self.trajectories.keys() if len(self.trajectories[env_id]) % 2]

        if required_predictions:
            # todo: check how/if random & expert mode can work with agents that require meta information (e.g. ppo)
            if mode == "random":
                predictions = [self.agent.action_space.sample() for _ in range(len(required_predictions))]
                predictions = np.stack(predictions),
            elif mode == "expert":
                predictions = []

                for env_id in required_predictions:
                    state, goal, info = self.trajectories[env_id][-1]

                    predictions.append(info["expert_action"])

                predictions = np.stack(predictions),
            else:
                states = []
                goals = []

                for env_id in required_predictions:
                    state, goal, _ = self.trajectories[env_id][-1]

                    state = unwind_dict_values(state)
                    goal = unwind_dict_values(goal)

                    states.append(state)
                    goals.append(goal)

                states = np.stack(states)
                goals = np.stack(goals)

                predictions = self.agent.predict(states, goals, deterministic=mode == "test")

            for env_id, *prediction in zip(required_predictions, *predictions):
                self.trajectories[env_id].append(prediction)
                requests.append((env_id, "step", prediction[0]))

        return requests, results_episodes, failed_runs

    def __init__(self, experiment_config):
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.INFO)

        self.experiment_config = experiment_config

    def generate_trajectories(self, number_trajectories, mode):

        env_responses = self.orchestrator.reset_all(partial(self.get_initial_state, random=mode != "test"))

        # subtract tests already launched in each environment
        remaining_runs = number_trajectories - len(self.orchestrator)

        # remove excessive tests
        while remaining_runs < 0:
            excessive_runs = min(abs(remaining_runs), len(env_responses))

            remaining_runs += excessive_runs
            env_responses = env_responses[:-excessive_runs]

            env_responses += self.orchestrator.receive()

        concluded_runs = []

        pbar = tqdm(total=number_trajectories, desc=mode, leave=False)

        while len(concluded_runs) < number_trajectories:
            requests, results_episodes, failed_runs = self.process_responses(env_responses, mode=mode)
            concluded_runs += results_episodes
            pbar.update(len(results_episodes))

            remaining_runs += failed_runs

            for ii in reversed(range(len(requests))):
                if requests[ii][1] == "reset":
                    if remaining_runs:
                        remaining_runs -= 1
                    else:
                        del requests[ii]

            env_responses = self.orchestrator.send_receive(requests)

        pbar.close()

        return concluded_runs

    def run(self, results_dir="./results", experiment_name=None, seed=None):
        """
        Runs the experiment
        :param results_dir: directory where results are saved
        :param experiment_name: name of experiment and directory name
        :param seed: seed for random, numpy, cuda and pytorch
        """
        experiment_config = self.experiment_config.copy()

        if seed is None:
            seed = random.randint(0, 10000)
        set_seed(seed)

        experiment_config['seed'] = seed

        # Initialize results folder structure
        if not experiment_name:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            self.logger.warning(f"Experiment name not specified. Using {experiment_name}.")

        experiment_dir = osp.join(results_dir, experiment_name)

        if os.path.exists(experiment_dir):
            appendix = 1

            while os.path.exists(f"{experiment_dir}_{appendix}"):
                appendix += 1

            self.logger.warning(f"Result directory already exists {experiment_dir}. "
                                f"Using directory {experiment_dir}_{appendix} instead.")

            experiment_dir += f"_{appendix}"

        models_dir = osp.join(experiment_dir, "models")

        os.makedirs(experiment_dir)
        os.makedirs(models_dir)

        with open(osp.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(self.experiment_config, f)

        env_config = experiment_config["env_config"]
        number_processes = experiment_config.get("number_processes", 1)
        number_threads = experiment_config.get("number_threads", 1)

        with Orchestrator(env_config, number_processes, number_threads) as self.orchestrator:

            self.reward_function = self.orchestrator.reward_function
            self.success_criterion = self.orchestrator.success_criterion

            # get agents
            agent_config = experiment_config["agent_config"]

            self.agent = get_agent(agent_config,
                                   self.orchestrator.state_space,
                                   self.orchestrator.goal_space,
                                   self.orchestrator.action_space,
                                   self.reward_function,
                                   experiment_dir)

            models_dir = osp.join(experiment_dir, "models")

            base_experiment = experiment_config.pop('base_experiment', None)

            if base_experiment is not None:
                base_experiment_dir = osp.join(results_dir, base_experiment["experiment"])

                with open(osp.join(base_experiment_dir, 'config.json'), 'r') as f:
                    base_experiment_config = json.load(f)

                if self.similar_config(env_config, base_experiment_config["env_config"]) and self.similar_config(
                        agent_config, base_experiment_config["agent_config"]):

                    base_experiment_models_dir = osp.join(base_experiment_dir, "models")

                    agent_id = base_experiment.get("agent", max(os.listdir(base_experiment_models_dir)))

                    self.agent.load(osp.join(base_experiment_models_dir, agent_id))
                else:
                    raise ValueError("Configurations do not match!")

            self.writer = SummaryWriter(experiment_dir, 'trainer')

            # reset all
            env_responses = self.orchestrator.reset_all(partial(self.get_initial_state, random=True))

            self.steps = defaultdict(int)
            self.trajectories = defaultdict(list)

            number_tests = experiment_config.get("number_tests", 1)
            test_interval = experiment_config["test_interval"]
            next_test_timestep = 0

            best_success_ratio = 0.0

            pbar = tqdm(total=experiment_config["total_timesteps"], desc="Progress")

            while sum(self.steps.values()) < experiment_config["total_timesteps"]:

                # Test
                if sum(self.steps.values()) >= next_test_timestep:

                    next_test_timestep = (sum(self.steps.values()) // test_interval + 1) * test_interval

                    concluded_tests = self.generate_trajectories(number_tests, mode="test")

                    # evaluate test
                    success_ratio = np.mean(concluded_tests)

                    self.writer.add_scalar('test success ratio', success_ratio, sum(self.steps.values()) + 1)

                    if success_ratio >= best_success_ratio:
                        best_success_ratio = success_ratio
                        self.agent.save(os.path.join(models_dir, f"{success_ratio:.3f}_{sum(self.steps.values()) + 1}"))

                    # reset all
                    env_responses = self.orchestrator.reset_all(partial(self.get_initial_state, random=True))

                # Train
                requests, results_episodes, _ = self.process_responses(env_responses, mode="train")

                env_responses = self.orchestrator.send_receive(requests)

                self.agent.train(sum(self.steps.values()))

                pbar.update(sum(self.steps.values()) - pbar.n)
                pbar.refresh()

            self.agent.save(os.path.join(models_dir, f"final"))
