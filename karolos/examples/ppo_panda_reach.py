import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from karolos.experiment import Experiment

if __name__ == "__main__":
    experiment_name = "ppo_panda_reach"

    training_config = {
        "total_timesteps": 25_000_000,
        "test_interval": 500_000,
        # "number_tests": 96,
        # "number_processes": 96,
        "agent_config": {
            "algorithm": "ppo",
            "learning_rate_critic": 5e-3,
            "learning_rate_policy": 5e-3,
            "weight_decay": 5e-5,
            "batch_size": 40960,
            "reward_discount": 0.99,
            "clip_eps": 0.1,
            "gradient_steps": 8,
            "n_mini_batch": 20,
            "action_std_init": 0.9,
            "action_std_decay": 0.2,
            "action_std_decay_freq": 50_000,
            "min_action_std": 0.1,
            "value_loss_coeff": 1.0,
            "entropy_loss_coeff": 0.000,

            "policy_structure": [
                ("linear", 256),
                ("tanh", None),
                ("linear", 128),
                ("tanh", None),
                ("linear", 64),
                ("tanh", None),
                ("linear", 32),
                ("tanh", None),
            ],
            "critic_structure": [
                ("linear", 256),
                ("tanh", None),
                ("linear", 128),
                ("tanh", None),
                ("linear", 64),
                ("tanh", None),
                ("linear", 32),
                ("tanh", None),
            ]
        },
        "env_config": {
            "environment": "karolos",

            "task_config": {
                "name": "reach",
                "max_steps": 50,
            },
            "robot_config": {
                "name": "panda",
                "sim_time": .1,
                "scale": .1,
            }
        }
    }

    # Use a fixed goal for reach task
    def get_initial_state_custom(*args, **kwargs):
        initial_state_ppo = {
            "robot": None,  # random robot
            "task": [0.5, 0.25, 0.25],  # non-random task
        }

        return initial_state_ppo

    experiment = Experiment(training_config)
    experiment.get_initial_state = get_initial_state_custom
    experiment.run("../results", experiment_name=experiment_name)