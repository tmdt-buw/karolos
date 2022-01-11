from karolos.experiment import Experiment

if __name__ == "__main__":

    activation = "tanh"  # relu, tanh, leaky_relu
    experiment_name = "ppo_panda"

    training_config = {
        "total_timesteps": 25_000_000,
        "test_interval": 500_000,
        # "number_tests": 96,
        # "number_processes": 96,
        "agent_config": {

            # SAC
            "algorithm": "sac",
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
            "policy_structure": [('linear', 128), (activation, None)] * 3,
            "critic_structure": [('linear', 128), (activation, None)] * 3,

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

    experiment = Experiment(training_config)
    experiment.run("../results", experiment_name=experiment_name)
