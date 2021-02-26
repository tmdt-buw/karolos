import os
import sys

sys.path.insert(0, os.path.abspath("."))

from agents import get_agent
from collections import defaultdict
import datetime
from environments.orchestrator import Orchestrator
import json
import numpy as np
import os
import os.path as osp
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from functools import partial
from utils import unwind_dict_values
from multiprocessing import cpu_count
import logging

class Trainer:
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

    def trajectory2experiences(self, trajectory, her_ratio=0.):
        assert len(trajectory) % 2

        experiences = []
        experiences_her = []

        trajectory_length = len(trajectory) // 2

        for trajectory_step in range(trajectory_length):
            state, _ = trajectory[trajectory_step * 2]
            action = trajectory[trajectory_step * 2 + 1]
            next_state, goal = trajectory[trajectory_step * 2 + 2]
            done = trajectory_step == len(trajectory) // 2 - 1
            reward = self.reward_function(state=state, action=action,
                                         next_state=next_state, done=done,
                                         goal=goal)

            reward /= trajectory_length

            state = unwind_dict_values(state)
            goal = unwind_dict_values(goal["desired"])
            next_state = unwind_dict_values(next_state)

            experience = {
                "state": state,
                "goal": goal,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            experiences.append(experience)

        for _ in range(int(her_ratio * len(trajectory) // 2)):
            # sample random step
            trajectory_step = np.random.randint(len(trajectory) // 2)

            state, _ = trajectory[trajectory_step * 2]
            action = trajectory[trajectory_step * 2 + 1]
            next_state, goal = trajectory[trajectory_step * 2 + 2]
            done = trajectory_step == len(trajectory) // 2 - 1

            # sample future goal
            goal_step = np.random.randint(trajectory_step,
                                          len(trajectory) // 2)
            _, future_goal = trajectory[goal_step * 2]

            goal = goal.copy()
            goal["desired"] = future_goal["achieved"]

            reward = self.reward_function(state=state, action=action,
                                         next_state=next_state, done=done,
                                         goal=goal)

            reward /= trajectory_length

            state = unwind_dict_values(state)
            goal = unwind_dict_values(goal["desired"])
            next_state = unwind_dict_values(next_state)

            experience = {
                "state": state,
                "goal": goal,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            experiences_her.append(experience)

        return experiences, experiences_her

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
                state, goal, done = data
                self.trajectories[env_id].append((state, goal))

                done |= self.success_criterion(goal)

                if done:
                    experiences, experiences_her = self.trajectory2experiences(
                        self.trajectories.pop(env_id), self.her_ratio)

                    reward = sum(
                        [experience["reward"] for experience in experiences])

                    self.agent.add_experiences(experiences)
                    self.agent.add_experiences(experiences_her)

                    self.writer.add_scalar(
                        f'{mode} reward episode', reward,
                        sum(self.steps.values()) + 1
                    )

                    results_episodes.append(self.success_criterion(goal))

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
                goals = []

                for env_id in required_predictions:
                    state, goal = self.trajectories[env_id][-1]

                    state = unwind_dict_values(state)
                    goal = unwind_dict_values(goal["desired"])

                    states.append(state)
                    goals.append(goal)

                states = np.stack(states)
                goals = np.stack(goals)

                predictions = self.agent.predict(states, goals,
                                                 deterministic=mode == "test")

            for env_id, prediction in zip(required_predictions,
                                          predictions):
                self.trajectories[env_id].append(prediction)
                requests.append((env_id, "step", prediction))

        return requests, results_episodes

    def __init__(self, training_config, results_dir="./results", experiment_name=None):

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

            logger.warning(f"Result directory already exists {experiment_dir}. "
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
                                   experiment_dir)

            models_dir = osp.join(experiment_dir, "models")

            base_experiment = training_config.pop('base_experiment', None)

            if base_experiment is not None:

                base_experiment_dir = osp.join(log_dir,
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
            self.her_ratio = training_config.get("her_ratio", 0)
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

    import itertools



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

    trainer = Trainer(training_config, "../results", experiment_name="reach_sac_default")