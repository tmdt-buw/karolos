import numpy as np
import tensorflow as tf


# def get_env_metaworld(env_config):
#     from metaworld.benchmarks import ML10
#     from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
#
#     task_name = env_config["task_name"]
#
#     if task_name in MEDIUM_MODE_ARGS_KWARGS['train'].keys():
#         def env_init():
#             return ML10.get_train_tasks(task_name)
#     elif task_name in MEDIUM_MODE_ARGS_KWARGS['test'].keys():
#         def env_init():
#             return ML10.get_test_tasks(task_name)
#     else:
#         possible_task_names = set(
#             MEDIUM_MODE_ARGS_KWARGS['train'].keys()) | set(
#             MEDIUM_MODE_ARGS_KWARGS['test'].keys())
#
#         raise AssertionError(
#             f"Unknown task {task_name} [Metaworld]. "
#             f"Possible tasks {list(possible_task_names)}")
#
#     return env_init


def get_env(env_config):
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


def train_stable_baselines(env, algorithm, test_interval, total_timesteps,
                           nb_tests, models_dir=None, log_dir=None):
    assert algorithm in ["SAC"]

    if algorithm == "SAC":
        from stable_baselines import SAC
        from stable_baselines.sac import LnMlpPolicy

        model = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log=log_dir)

        global next_test_timestep, best_success_rate
        next_test_timestep = 0
        best_success_rate = 0

        from functools import partial

        def _callback(locals_, globals_, test_interval, nb_tests):

            global next_test_timestep, best_success_rate

            current_timestep = locals_["step"]

            if current_timestep >= next_test_timestep:
                next_test_timestep = (
                                             current_timestep // test_interval + 1) * test_interval

                successes = []
                episodes_rewards = []

                for tt in range(nb_tests):
                    episode_rewards = []
                    obs = env.reset()
                    done = False
                    while not done:
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        episode_rewards.append(reward)

                    # todo: does sum make sense or should we use mean?
                    episodes_rewards.append(sum(episode_rewards))

                    successes.append(info["goal_reached"])

                success_rate = sum(successes) / len(successes)

                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='test_reward',
                                            simple_value=np.mean(
                                                episodes_rewards)),
                           tf.Summary.Value(tag='test_success_rate',
                                            simple_value=success_rate)])
                locals_['writer'].add_summary(summary, current_timestep)

                if success_rate >= best_success_rate and success_rate > 0.5 \
                        and models_dir is not None:
                    best_success_rate = success_rate
                    model.save(os.path.join(models_dir,
                                            f"{current_timestep}_{success_rate:.3f}.tf"))

            return True

        callback = partial(_callback, test_interval=test_interval,
                           nb_tests=nb_tests)

        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False,
                    callback=callback)


def run_training(training_config):
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
    env = get_env(env_config)()

    training_base_pkg = training_config.pop("base_pkg")

    if training_base_pkg == "stable-baselines":
        train_stable_baselines(env, **training_config,
                               log_dir=log_dir, models_dir=models_dir)


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
        "env_config": {
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

    run_training(training_config)
