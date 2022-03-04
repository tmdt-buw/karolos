from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1].resolve()))

from karolos.environments.orchestrator import Orchestrator

import time

if __name__ == "__main__":
    env_config = {

        "environment": "karolos",
        "render": False,
        "task_config": {
            "name": "reach",
        },
        "robot_config": {
            "name": "panda",
        }

    }

    for number_processes in [1, 2, 3]:
        for number_threads in [1, 5, 10]:
            with Orchestrator(env_config, number_processes,
                              number_threads) as orchestrator:

                env_responses = orchestrator.reset_all()

                t0 = time.time()
                total_responses = 0

                while total_responses < 1e2:
                    requests = []
                    for env_id, response in env_responses:
                        func, data = response

                        requests.append((env_id, "reset", None))

                    env_responses = orchestrator.send_receive(requests)
                    total_responses += len(env_responses)

                duration = time.time() - t0

                # print(number_processes, number_threads, duration,
                #       duration / (number_processes * number_threads))
