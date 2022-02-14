import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from karolos.experiment import Experiment
from multiprocessing import cpu_count

if __name__ == "__main__":

    experiment_name = "sac_panda_reach"

    training_config = {
        "total_timesteps": 15_000_000,
        "test_interval": 500_000,
        "number_tests": 100,
        "number_processes": 2 * cpu_count(),
        "agent_config": {

            # SAC
            "algorithm": "sac",
            "learning_rate_critic": 0.0005,
            "learning_rate_policy": 0.0005,
            "entropy_regularization": 1,
            "learning_rate_entropy_regularization": 5e-5,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "reward_discount": 0.99,
            "reward_scale": 100,
            "automatic_entropy_regularization": True,
            "gradient_clipping": False,
            "tau": 0.0025,
            "policy_structure": [('linear', 128), ('tanh', None)] * 8,
            "critic_structure": [('linear', 128), ('tanh', None)] * 8,

            "replay_buffer": {
                "name": "priority",
                "buffer_size": int(1e6)
            }
        },
        "env_config": {
            "environment": "karolos",
            "task_config": {
                "name": "reach",
                "max_steps": 25,
            },
            "robot_config": {
                "name": "panda",
                "sim_time": .1,
                "scale": .1,
            }
        }
    }

    experiment = Experiment(training_config)
    experiment.run("results", experiment_name=experiment_name)
