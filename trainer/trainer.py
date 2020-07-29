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
    @staticmethod
    def get_initial_state(random: bool, env_id=None):
        # initial_state = {
        #     'robot': self.env_orchestrator.observation_dict['state'][
        #         'robot'].sample(),
        #     'task': self.env_orchestrator.observation_dict['state'][
        #         'task'].sample()
        # }

        initial_state = None

        return initial_state

    @staticmethod
    def reward_function(done, goal, **kwargs):
        if goal["reached"]:
            reward = 1.
        elif done:
            reward = -1.
        else:
            reward = np.exp(-5 * np.linalg.norm(goal["achieved"] -
                                                goal["desired"])) - 1

        return reward

    @classmethod
    def trajectory2experiences(cls, trajectory):
        assert len(trajectory) % 2

        experiences = []

        for trajectory_step in range(len(trajectory) // 2):
            state, _ = trajectory[trajectory_step * 2]

            action = trajectory[trajectory_step * 2 + 1]

            next_state, goal = trajectory[trajectory_step * 2 + 2]

            done = trajectory_step == len(trajectory) // 2 - 1

            reward = cls.reward_function(state=state, action=action,
                                         next_state=next_state, done=done,
                                         goal=goal)

            state = np.concatenate([state["robot"], state["task"]], axis=-1)
            goal = goal["desired"]
            next_state = np.concatenate(
                [next_state["robot"], next_state["task"]], axis=-1)

            experience = {
                "state": state,
                "goal": goal,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            experiences.append(experience)

        return experiences

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
                                                            env_id)))
                else:
                    self.trajectories.pop(env_id, None)
                    self.trajectories[env_id].append(data)
            elif func == "step":
                state, goal, done = data
                self.trajectories[env_id].append((state, goal))

                if done:
                    experiences = self.trajectory2experiences(self.trajectories.pop(env_id))

                    reward = sum(
                        [experience["reward"] for experience in experiences])

                    self.agent.add_experiences(experiences)

                    self.writer.add_scalar(
                        f'{mode} reward episode', reward,
                        sum(self.steps.values()) + 1
                    )

                    results_episodes.append(goal["reached"])

                    requests.append(
                        (env_id, "reset",
                         self.get_initial_state(mode != "test", env_id)))

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

                    state = np.concatenate([state["robot"], state["task"]])
                    goal = goal["desired"]

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
                               self.env_orchestrator.action_space,
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
        env_responses = self.env_orchestrator.reset_all(
            partial(self.get_initial_state, random=True))

        self.steps = defaultdict(int)
        self.trajectories = defaultdict(list)

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
    learning_rates = [(0.0005, 0.0005)]
    hidden_layer_sizes = [32]
    network_depths = [10]
    entropy_regularization_learning_rates = [1e-3]
    taus = [0.0025]

    import itertools

    for params in itertools.product(learning_rates, hidden_layer_sizes,
                                    network_depths,
                                    entropy_regularization_learning_rates,
                                    taus):
        experiment_name = experiment_date + "/" + "_".join(
            list(map(str, params)))

        learning_rates, hidden_layer_size, network_depth, entropy_regularization_learning_rate, tau = params

        learning_rate_policy, learning_rate_critic = learning_rates

        training_config = {
            "total_timesteps": 20_000_000,
            "test_interval": 100_000,
            "number_tests": 100,
            # "base_experiment": {
            #     "experiment": "20200513-145010",
            #     "agent": 0,
            # },

            "algorithm": "sac",
            "agent_config": {
                "learning_rate_critic": learning_rate_critic,
                "learning_rate_policy": learning_rate_policy,
                "entropy_regularization": 1,
                "learning_rate_entropy_regularization": entropy_regularization_learning_rate,
                "weight_decay": 1e-4,
                "batch_size": 64,
                "reward_discount": 0.99,
                "reward_scale": 100,
                "automatic_entropy_regularization": True,
                "gradient_clipping": False,
                "memory_size": 1_000_000,
                "tau": tau,
                "policy_structure": [('linear', hidden_layer_size),
                                     ('relu', None)] * network_depth,
                "critic_structure": [('linear', hidden_layer_size),
                                     ('relu', None)] * network_depth
            },
            "number_envs": 12,
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
                    "scale": .1,
                    "use_gripper": True,
                    "mirror_finger_control": True,
                }
            }
        }

        trainer = Trainer(training_config, experiment_name)
