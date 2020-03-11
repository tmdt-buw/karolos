import multiprocessing as mp
import random
from environments import get_env

class Orchestrator(object):

    def __init__(self, env_config, nb_envs):

        self.pipes = {}
        self.action_space_ = None
        self.observation_space_ = None

        for ee in range(nb_envs):
            pipe_orchestrator, pipe_env = mp.Pipe()

            self.pipes[ee] = pipe_orchestrator

            p = mp.Process(target=self.run,
                           args=(env_config, pipe_env),
                           daemon=True)
            p.start()

    def run(self, env_config, pipe):

        env = get_env(env_config)()

        while True:

            func, params = pipe.recv()

            if func == "close":
                break
            if func == "ping":
                pipe.send(("ping", params))
            elif func == "reset":
                pipe.send(("reset", env.reset()))
            elif func == "step":
                pipe.send(("step", env.step(params)))
            elif func == "render":
                pipe.send(("render", env.render(params)))
            elif func == "action space":
                pipe.send(("action space", env.action_space))
            elif func == "observation space":
                pipe.send(("observation space", env.observation_space))
            else:
                raise NotImplementedError(func)

    def send_receive(self, actions=()):
        self.send(actions)
        return self.receive()


    def send(self, actions=()):

        for env_id, func, params in actions:

            if func == "step":
                if not self.action_space.contains(params):
                    print(params)

            # print("send", [func, params])

            self.pipes[env_id].send([func, params])

    def receive(self):
        responses = []

        for env_id, pipe in self.pipes.items():

            if pipe.poll():
                response = pipe.recv()

                # print("receive", response)

                responses.append((env_id, response))

        return responses

    def reset_all(self):
        """Resets all environment. Blocks until all environments are reset."""

        token = random.getrandbits(10)

        # send ping with token to flush the pipes
        self.send([(env_id, "ping", token) for env_id in self.pipes.keys()])
        self.send([(env_id, "reset", None) for env_id in self.pipes.keys()])

        required_env_ids = self.pipes.keys()

        responses = []

        # send reset commands
        for env_id, pipe in self.pipes.items():

            while True:
                response = pipe.recv()
                func, data = response

                if func == "ping" and data == token:
                    break

            # next response is reset
            response = pipe.recv()
            func, data = response

            assert func == "reset", func

            responses.append((env_id, response))

        return responses

    def reset(self, env_id):
        """Reset the environment and return new state
        """

        self.pipes[env_id].send(["reset", None])

        func, state = self.pipes[env_id].recv()

        assert func == "reset"

        return state

    @property
    def action_space(self):

        if self.action_space_ is None:
            self.pipes[0].send(["action space", None])
            func, self.action_space_ = self.pipes[0].recv()

            assert func == "action space", f"'{func}' istead of 'action space'"

        return self.action_space_

    @property
    def observation_space(self):
        if self.observation_space_ is None:
            self.pipes[0].send(["observation space", None])
            func, self.observation_space_ = self.pipes[0].recv()

            assert func == "observation space", f"'{func}' istead of " \
                                                f"'observation space'"

        return self.observation_space_

if __name__ == "__main__":
    import os.path as osp
    import os
    import datetime

    env_config = {
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

    import numpy as np
    import time

    nb_envs = 3

    orchestrator = Orchestrator(env_config, nb_envs)

    while True:
        result = orchestrator.send_receive([(ee, "reset", None) for ee in range(nb_envs)])

        print(len(result))
        # time.sleep(3)
