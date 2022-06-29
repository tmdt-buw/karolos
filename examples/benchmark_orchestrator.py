import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

from karolos.environments.orchestrator import Orchestrator
from multiprocessing import cpu_count
import time
from tqdm import tqdm

# NOTE: Please install matplotlib, seaborn and pandas first. They are not included in the dependencies of Karolos.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_orchestrator(number_processes=1, number_threads=1, total_responses=1e4):
    with Orchestrator(env_config, number_processes,
                      number_threads) as orchestrator:

        durations = []
        for _ in tqdm(range(10)):
            env_responses = orchestrator.reset_all()
            t0 = time.time()

            responses = 0

            while responses < total_responses:
                requests = []
                for env_id, response in env_responses:
                    # func, data = response

                    requests.append((env_id, "reset", None))

                env_responses = orchestrator.send_receive(requests)
                responses += len(env_responses)

            duration = time.time() - t0
            durations.append(duration)

    return durations


if __name__ == "__main__":
    env_config = {
        "render": False,
        "task_config": {
            "name": "reach",
        },
        "robot_config": {
            "name": "panda",
        }

    }

    numbers_processes = range(1, cpu_count(), cpu_count() // 5)

    results = []

    for number_processes in tqdm(numbers_processes):
        durations = run_orchestrator(number_processes=number_processes)
        results.append(durations)

    df_cpu = pd.DataFrame(results, index=numbers_processes).T
    df_cpu.to_csv("benchmark_cpu.csv")

    numbers_threads = range(1, 6)

    results = []

    for number_threads in tqdm(numbers_threads):
        durations = run_orchestrator(number_processes=cpu_count(), number_threads=number_threads)
        results.append(durations)

    df_thread = pd.DataFrame(results, index=numbers_threads).T
    df_thread.to_csv("results_thread.csv")

    fig = plt.figure()
    ax = sns.boxplot(data=df_cpu, showfliers=False)
    ax.set_ylabel("Duration [s]")
    ax.set_xlabel("Number of CPUs")
    ax.set_yscale("log")

    fig = plt.figure()
    ax = sns.boxplot(data=df_thread, showfliers=False)
    ax.set_ylabel("Duration [s]")
    ax.set_xlabel("Number of Threads")
    # ax.set_yscale("log")
