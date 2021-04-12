import os
import sys

sys.path.insert(0, os.path.abspath("."))

from collections import defaultdict
import datetime
from functools import partial
import json
import logging
from multiprocessing import cpu_count
import numpy as np
import os.path as osp
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from karolos.agents import get_agent
from karolos.environments.orchestrator import Orchestrator
from karolos.utils import unwind_dict_values


class Manager:
    @staticmethod
    def get_initial_state(random: bool, env_id=None):
        # return None for random sampling of initial state

        # initial_state = {
        #     'robot': self.env_orchestrator.observation_dict['state'][
        #         'robot'].sample(),
        #     'task': self.env_orchestrator.observation_dict['state'][
        #         'task'].sample()
        # }

        initial_state = None

        return initial_state

    @staticmethod
    def similar_config(config_1, config_2):

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

    def process_responses(self, env_responses, mode: str):

        requests = []
        results_episodes = []

        for env_id, response in env_responses:
            func, data = response

            if func == "reset":
                if type(data) == AssertionError:
                    requests.append((env_id, "reset",
                                     self.get_initial_state(mode != "test",
                                                            env_id)
                                     ))
                else:
                    self.trajectories.pop(env_id, None)
                    self.trajectories[env_id].append(data)
            elif func == "step":
                state, goal_info, done = data
                self.trajectories[env_id].append((state, goal_info))

                done |= self.success_criterion(goal_info)

                if done:
                    trajectory = self.trajectories.pop(env_id)

                    rewards = self.agent.add_experience_trajectory(trajectory)

                    self.writer.add_scalar(
                        f'{mode} reward episode', sum(rewards),
                        sum(self.steps.values()) + 1
                    )

                    results_episodes.append(self.success_criterion(goal_info))

                    requests.append(
                        (env_id, "reset",
                         self.get_initial_state(mode != "test", env_id)
                         ))

                self.steps[env_id] += 1
            else:
                raise NotImplementedError(
                    f"Undefined behavior for {env_id} | {response}")

        required_predictions = [env_id for env_id in self.trajectories.keys()
                                if len(self.trajectories[env_id]) % 2]

        if required_predictions:

            if mode == "random":
                predictions = [self.agent.action_space.sample() for _ in
                               range(len(required_predictions))]
            else:
                states = []

                for env_id in required_predictions:
                    state, _ = self.trajectories[env_id][-1]

                    state = unwind_dict_values(state)

                    states.append(state)

                states = np.stack(states)

                predictions = self.agent.predict(states,
                                                 deterministic=mode == "test")

            for env_id, prediction in zip(required_predictions,
                                          predictions):
                self.trajectories[env_id].append(prediction)
                requests.append((env_id, "step", prediction))

        return requests, results_episodes

    def __init__(self, training_config, results_dir="./results",
                 experiment_name=None):

        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)

        if not experiment_name:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            logger.warning(f"Experiment name not specified. "
                           f"Using {experiment_name}.")

        experiment_dir = osp.join(results_dir, experiment_name)

        if os.path.exists(experiment_dir):
            appendix = 1

            while os.path.exists(f"{experiment_dir}_{appendix}"):
                appendix += 1

            logger.warning(
                f"Result directory already exists {experiment_dir}. "
                f"Using directory {experiment_dir}_{appendix} instead.")

            experiment_dir += f"_{appendix}"

        models_dir = osp.join(experiment_dir, "models")

        os.makedirs(experiment_dir)
        os.makedirs(models_dir)

        with open(osp.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(training_config, f)

        env_config = training_config["env_config"]
        number_processes = training_config.get("number_processes", 1)
        number_threads = training_config.get("number_threads", 1)

        with Orchestrator(env_config, number_processes,
                          number_threads) as self.orchestrator:

            self.reward_function = self.orchestrator.reward_function
            self.success_criterion = self.orchestrator.success_criterion

            # get agents
            agent_config = training_config["agent_config"]

            self.agent = get_agent(agent_config,
                                   self.orchestrator.observation_space,
                                   self.orchestrator.action_space,
                                   self.reward_function,
                                   experiment_dir)

            models_dir = osp.join(experiment_dir, "models")

            base_experiment = training_config.pop('base_experiment', None)

            if base_experiment is not None:

                base_experiment_dir = osp.join(results_dir,
                                               base_experiment["experiment"])

                with open(osp.join(base_experiment_dir, 'config.json'), 'r') as f:
                    base_experiment_config = json.load(f)

                if self.similar_config(env_config, base_experiment_config[
                    "env_config"]) and self.similar_config(agent_config,
                                                           base_experiment_config[
                                                               "agent_config"]):

                    base_experiment_models_dir = osp.join(base_experiment_dir,
                                                          "models")

                    agent_id = base_experiment.get("agent", max(
                        os.listdir(base_experiment_models_dir)))

                    self.agent.load(osp.join(base_experiment_models_dir, agent_id))
                else:
                    raise ValueError("Configurations do not match!")

            self.writer = SummaryWriter(experiment_dir, 'trainer')

            # reset all
            env_responses = self.orchestrator.reset_all(
                partial(self.get_initial_state, random=True))

            self.steps = defaultdict(int)
            self.trajectories = defaultdict(list)

            number_tests = training_config["number_tests"]
            test_interval = training_config["test_interval"]
            next_test_timestep = 0

            best_success_ratio = 0.0

            pbar = tqdm(total=training_config["total_timesteps"], desc="Progress")

            while sum(self.steps.values()) < training_config["total_timesteps"]:

                # Test
                if sum(self.steps.values()) >= next_test_timestep:

                    next_test_timestep = (sum(self.steps.values()) //
                                          test_interval + 1) * test_interval

                    # reset all
                    env_responses = self.orchestrator.reset_all(
                        partial(self.get_initial_state, random=False))

                    # subtract tests already launched in each environment
                    tests_to_run = number_tests - len(self.orchestrator)

                    # remove excessive tests
                    while tests_to_run < 0:
                        excessive_tests = min(abs(tests_to_run),
                                              len(env_responses))

                        tests_to_run += excessive_tests
                        env_responses = env_responses[:-excessive_tests]

                        env_responses += self.orchestrator.receive()

                    concluded_tests = []

                    pbar_test = tqdm(total=number_tests, desc="Test", leave=False)

                    while len(concluded_tests) < number_tests:

                        for response in env_responses:
                            func, data = response[1]

                            if func == "reset" and type(
                                    data) == AssertionError:
                                tests_to_run += 1

                        requests, results_episodes = self.process_responses(
                            env_responses, mode="test")
                        concluded_tests += results_episodes
                        pbar_test.update(len(results_episodes))

                        for ii in reversed(range(len(requests))):
                            if requests[ii][1] == "reset":
                                if tests_to_run:
                                    tests_to_run -= 1
                                else:
                                    del requests[ii]

                        env_responses = self.orchestrator.send_receive(
                            requests)

                    pbar_test.close()

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
                    env_responses = self.orchestrator.reset_all(
                        partial(self.get_initial_state, random=True))

                # Train
                requests, results_episodes = self.process_responses(
                    env_responses, mode="train")

                env_responses = self.orchestrator.send_receive(requests)

                self.agent.train(sum(self.steps.values()))

                pbar.update(sum(self.steps.values()) - pbar.n)
                pbar.refresh()

            self.agent.save(os.path.join(models_dir, f"final"))


if __name__ == "__main__":

    learning_rates = [(0.0005, 0.0005)]
    hidden_layer_sizes = [32]
    network_depths = [8]
    entropy_regularization_learning_rates = [5e-5]
    taus = [0.0025]
    her_ratios = [0.0]

    training_config = {
        "total_timesteps": 5_000_000,
        "test_interval": 500_000,
        "number_tests": 100,

        "agent_config": {
            "algorithm": "sac",

            "policy_structure": [('linear', 32), ('relu', None)] * 8,
            "critic_structure": [('linear', 32), ('relu', None)] * 8
        },
        "env_config": {
            "environment": "karolos",
            "render": True,
            "task_config": {
                "name": "reach",
            },
            "robot_config": {
                "name": "panda",
            }
        }
    }

    trainer = Manager(training_config, "results")
