import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from karolos.experiment import Experiment
from multiprocessing import cpu_count

if __name__ == "__main__":
    training_config = {
        "total_timesteps": 5_000_000,
        "test_interval": 500_000,
        "number_tests": 100,
        "number_processes": cpu_count(),
        "agent_config": {
            # SAC
            "name": "sac",
            "learning_rate_critic": 0.005,
            "learning_rate_policy": 0.005,
            "entropy_regularization": 1,
            "learning_rate_entropy_regularization": 5e-5,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "reward_discount": 0.99,
            "reward_scale": 100,
            "automatic_entropy_regularization": True,
            "gradient_clipping": False,
            "memory_size": 1_000_000,
            "tau": 0.0025,
            "policy_structure": [('linear', 128), ("tanh", None)] * 3,
            "critic_structure": [('linear', 128), ("tanh", None)] * 3,
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

    import copy

    # for _ in range(5):
    experiment = Experiment(copy.deepcopy(training_config))
    # experiment.run("results/sac_panda_reach")
    experiment.run(f"results/{os.path.basename(__file__).replace('.py', '')}")
