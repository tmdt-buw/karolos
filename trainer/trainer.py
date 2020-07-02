import os
import sys

sys.path.insert(0, os.path.abspath("."))

from agents import get_agent
from collections import defaultdict
import datetime
from environments.orchestrator import Orchestrator
import json
from multiprocessing import cpu_count
import numpy as np
import os
import os.path as osp
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from functools import partial

log_dir = results_dir = osp.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../results")


class Trainer:
    def get_initial_state(self, random: bool, env_id=None):
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
                                     self.get_initial_state(mode != "test", env_id)))
                else:
                    self.states[env_id] = data
                    self.actions.pop(env_id, None)
                    self.episodic_reward[env_id] = []
            elif func == "step":
                state, reward, done = data

                # self.writer.add_scalar(
                #     f'{"train" if train else "test"} reward step',
                #     reward,
                #     sum(self.steps.values()) + 1
                # )

                self.episodic_reward[env_id].append(reward)
                previous_state = self.states.pop(env_id, None)
                if previous_state is not None:
                    action = self.actions.pop(env_id)

                    experience = (previous_state['state']['agent_state'],
                                  action, reward,
                                  state['state']['agent_state'], done)
                    self.agent.add_experience([experience])

                    if training_config.get("use_hindsight_experience_replay",
                                           False) and "her" in state:
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
                    # episode_reward = sum(self.episodic_reward[env_id])
                    #
                    # self.writer.add_scalar(
                    #     f'{"train" if train else "test"} reward episode',
                    #     episode_reward,
                    #     sum(self.steps.values()) + 1
                    # )

                    results_episodes.append(state["goal"]["reached"])

                    requests.append(
                        (env_id, "reset",
                         self.get_initial_state(mode != "test", env_id)))
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

            if mode == "random":
                predictions = [self.agent.action_space.sample() for _ in range(observations.shape[0])]
                # print(predictions.shape)
            else:
                predictions = self.agent.predict(observations,
                                             deterministic=mode == "test")

            for env_id, prediction in zip(required_predictions,
                                          predictions):
                self.actions[env_id] = prediction
                requests.append((env_id, "step", prediction))

        return requests, results_episodes

    def __init__(self, training_config, experiment_name):

        global log_dir

        experiment_dir = osp.join(log_dir, experiment_name)
        models_dir = osp.join(experiment_dir, "models")

        os.makedirs(experiment_dir)
        os.makedirs(models_dir)

        with open(osp.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(training_config, f)

        # get environment
        number_envs = training_config["number_envs"]
        env_config = training_config["env_config"]

        self.env_orchestrator = Orchestrator(env_config, number_envs)

        # get agents
        agent_config = training_config["agent_config"]

        algorithm = training_config["algorithm"]
        self.agent = get_agent(algorithm, agent_config,
                               self.env_orchestrator.observation_space,
                               self.env_orchestrator.action_space, experiment_dir)

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
        env_responses = self.env_orchestrator.reset_all(
            partial(self.get_initial_state, random=True))

        self.steps = defaultdict(int)

        self.states = {}
        self.actions = {}
        self.episodic_reward = {}

        number_tests = training_config["number_tests"]
        test_interval = training_config["test_interval"]
        next_test_timestep = 0

        best_success_ratio = 0.0

        # todo allow for less tests
        assert number_tests >= number_envs

        pbar = tqdm(total=training_config["total_timesteps"])

        mode = "train"

        while sum(self.steps.values()) < training_config["total_timesteps"]:

            # Test
            if sum(self.steps.values()) >= next_test_timestep:

                next_test_timestep = (sum(self.steps.values()) //
                                      test_interval + 1) * test_interval

                if sum(self.steps.values()):
                    mode = "test"
                else:
                    mode = "random"

                # reset all
                env_responses = self.env_orchestrator.reset_all(
                    partial(self.get_initial_state, random=False))

                # subtract tests already launched in each environment
                tests_to_run = number_tests - number_envs
                concluded_tests = []

                while len(concluded_tests) < number_tests:

                    for response in env_responses:
                        func, data = response[1]

                        if func == "reset" and type(data) == AssertionError:
                            tests_to_run += 1

                    requests, results_episodes = self.process_responses(
                        env_responses, mode=mode)
                    concluded_tests += results_episodes

                    for ii in reversed(range(len(requests))):
                        if requests[ii][1] == "reset":
                            if tests_to_run:
                                tests_to_run -= 1
                            else:
                                del requests[ii]

                    env_responses = self.env_orchestrator.send_receive(
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
                env_responses = self.env_orchestrator.reset_all(
                    partial(self.get_initial_state, random=True))

                mode = "train"

            # Train
            requests, _ = self.process_responses(env_responses, mode=mode)

            env_responses = self.env_orchestrator.send_receive(requests)

            self.agent.learn(sum(self.steps.values()) + 1)

            pbar.update(sum(self.steps.values()) - pbar.n)
            pbar.refresh()


if __name__ == "__main__":
    experiment_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 0.0005
    learning_rates = [0.0005]
    hidden_layer_sizes = [32]
    network_depths = [8]
    alpha_learning_rates = [1e-5]

    import itertools

    for params in itertools.product(learning_rates, hidden_layer_sizes, network_depths, alpha_learning_rates):

        experiment_name = experiment_date + "/" + "_".join(list(
            map(str, params)))

        learning_rate, hidden_layer_size, network_depth, alpha_learning_rate = params

        training_config = {
            "total_timesteps": 5_000_000,
            "test_interval": 500_000,
            "number_tests": 100,
            # "base_experiment": {
            #     "experiment": "20200513-145010",
            #     "agent": 0,
            # },
            # "algorithm": "sac",
            # "agent_config": {
            #     "learning_rate_critic": learning_rate,
            #     "learning_rate_policy": learning_rate,
            #     "alpha": 1,
            #     "learning_rate_alpha": alpha_learning_rate,
            #     "weight_decay": 1e-4,
            #     "batch_size": 128,
            #     "reward_discount": 0.99,
            #     "auto_entropy": True,
            #     "gradient_clipping": False,
            #     "memory_size": 1_000_000,
            #     "tau": 0.0025,
            #     "seed": 192,
            #     "policy_structure": [('linear', hidden_layer_size),
            #                          ('relu', None)] * network_depth,
            #     "critic_structure": [('linear', hidden_layer_size),
            #                          ('relu', None)] * network_depth
            # },
            "algorithm": "ddpg",
            "agent_config": {
                "learning_rate_critic": learning_rate,
                "learning_rate_policy": learning_rate,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "reward_discount": 0.99,
                "reward_scale": 100,
                "automatic_entropy_regularization": True,
                "gradient_clipping": False,
                "memory_size": 1_000_000,
                "tau": 0.0025,
                "seed": 192,
                "policy_structure": [('linear', hidden_layer_size),
                                     ('relu', None)] * network_depth,
                "critic_structure": [('linear', hidden_layer_size),
                                     ('relu', None)] * network_depth
            },
            "number_envs": cpu_count(),
            "env_config": {
                "environment": "robot",
                "render": False,
                "task_config": {
                    "name": "reach",
                    "dof": 3,
                    "only_positive": False,
                    "sparse_reward": False,
                    "max_steps": 25
                },
                "robot_config": {
                    "name": "panda",
                    "dof": 3,
                    "sim_time": .1,
                    "scale": .1
                }
            }
        }

        trainer = Trainer(training_config, experiment_name)
