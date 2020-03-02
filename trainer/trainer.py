import json
import numpy as np
from agents.sac import AgentSAC
from environments.orchestrator import Orchestrator
from torch.utils.tensorboard.writer import SummaryWriter
from collections import defaultdict


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

    def get_agent(self, agent_config, state_dim, action_dim):
        algorithm = agent_config.pop("algorithm")

        if algorithm == "sac":
            agent = AgentSAC(agent_config, state_dim, action_dim)
        else:
            raise NotImplementedError(f"Unknown algorithm {algorithm}")

        return agent

    def __init__(self, training_config):

        assert training_config["base_pkg"] in ["stable-baselines"]

        # create results directories
        results_dir = training_config.pop("results_dir")
        models_dir = osp.join(results_dir, "models")
        log_dir = results_dir

        results_dir = osp.join("..", results_dir)
        models_dir = osp.join("..", models_dir)
        log_dir = osp.join("..", log_dir)

        if not osp.exists(results_dir):
            os.makedirs(results_dir)
        if not osp.exists(models_dir):
            os.makedirs(models_dir)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir)

        with open(osp.join(results_dir, 'config.json'), 'w') as f:
            json.dump(training_config, f)

        # get environment
        env_config = training_config.pop("env_config")

        nb_envs = env_config.pop("nb_envs", 1)
        env_orchestrator = Orchestrator(env_config, nb_envs)

        # get agents
        agent_config = training_config.pop("agent_config")

        # add action and state spaces to config
        agent = self.get_agent(agent_config,
                               env_orchestrator.observation_space,
                               env_orchestrator.action_space)

        # reset all
        env_responses = env_orchestrator.reset_all()

        steps = defaultdict(int)

        states = {}
        actions = {}
        episodic_reward = {}

        requests = []

        print(training_config)

        nb_tests = training_config["nb_tests"]
        test_interval = training_config["test_interval"]
        next_test_timestep = 0

        best_success_ratio = 0.5

        assert nb_tests >= nb_envs

        while sum(steps.values()) < training_config["total_timesteps"]:

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
                            state, reward, done, info = data

                            self.writer.add_scalar('test reward step',
                                                   reward,
                                                   sum(steps.values()) + 1
                                                   )

                            episodic_reward[env_id].append(reward)
                            previous_state = states.pop(env_id, None)
                            if previous_state is not None:
                                experience = (previous_state,
                                              actions.pop(env_id), reward,
                                              state, done)
                                agent.add_experience([experience])
                            if done:
                                episode_reward = sum(episodic_reward[env_id])

                                self.writer.add_scalar('test reward episode',
                                                       episode_reward,
                                                       sum(steps.values()) + 1
                                                       )

                                results_episodes.append(info["goal_reached"])
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
                        observations = [states[env_id] for env_id in
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
                    state, reward, done, info = data

                    self.writer.add_scalar('train reward step',
                                           reward,
                                           sum(steps.values()) + 1
                                           )

                    episodic_reward[env_id].append(reward)
                    previous_state = states.pop(env_id, None)
                    if previous_state is not None:
                        experience = (previous_state, actions.pop(env_id),
                                      reward, state, done)
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

                observations = [states[env_id] for env_id in required_predictions]
                observations = np.stack(observations)

                print(observations.shape)

                predictions = agent.random_action(observations)
                predictions = agent.predict(observations, deterministic=False)

                for env_id, prediction in zip(required_predictions,
                                              predictions):

                    actions[env_id] = prediction
                    requests.append((env_id, "step", prediction))

            env_responses = env_orchestrator.send_receive(requests)
            requests = []

            agent.learn()


if __name__ == "__main__":
    import os.path as osp
    import os
    import datetime

    results_dir = osp.join("results",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    training_config = {
        "base_pkg": "stable-baselines",
        "algorithm": "SAC",
        "test_interval": 10_000,
        "nb_tests": 100,
        "total_timesteps": 1_000_000,
        "results_dir": results_dir,
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
            "memory_size": 2500,
            "tau": 0.0025,
            "backup_interval": 100,
            "hidden_dim": 512,
            "seed": 192,
            "tensorboard_histogram_interval": 5
        },
        "env_config": {
            "nb_envs": 2,
            "base_pkg": "robot-task-rl",
            "render": False,
            "task_config": {"name": "reach",
                            "dof": 3,
                            "only_positive": False
                            },
            "robot_config": {
                "name": "pandas",
                "dof": 3
            }
        }
    }

    trainer = Trainer(training_config)
