from environments.orchestrator import Orchestrator
from agents.sac import AgentSAC
import numpy as np

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

        os.makedirs(results_dir)
        os.makedirs(models_dir)

        import json
        with open(osp.join(results_dir, 'config.json'), 'w') as f:
            json.dump(training_config, f)

        # get environment
        env_config = training_config.pop("env_config")

        nb_envs = env_config.pop("nb_envs", 1)
        env_orchestrator = Orchestrator(env_config, nb_envs)

        # get agents
        agent_config = training_config.pop("agent_config")

        # add action and state spaces to config
        state_dim = env_orchestrator.observation_space.shape
        action_dim = env_orchestrator.action_space.shape

        agent = self.get_agent(agent_config, state_dim, action_dim)

        # reset all
        env_responses = env_orchestrator.reset_all()

        step = 0

        states = {}
        actions = {}

        requests = []

        while step < training_config["total_timesteps"]:

            for env_id, response in env_responses:
                func, data = response

                if func == "reset":
                    states[env_id] = data
                    actions.pop(env_id, None)
                elif func == "step":
                    state, reward, done, info = data
                    previous_state = states.pop(env_id, None)
                    if previous_state is not None:
                        experience = (previous_state, actions.pop(env_id),
                                      reward, state, done)
                        # todo store experience
                    if done:
                        requests.append((env_id, "reset", None))
                    else:
                        states[env_id] = state
                    step += 1
                else:
                    raise NotImplementedError(f"Undefined behavior for {env_id}"
                                              f" | {response}")

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
