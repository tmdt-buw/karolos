import multiprocessing as mp
import random

import numpy as np

from environments import get_env


class Orchestrator(object):

    def __init__(self, env_config, number_envs):

        self.pipes = {}
        self.action_space_ = None
        self.observation_space_ = None
        self.observation_dict_ = None

        for ee in range(number_envs):
            pipe_orchestrator, pipe_env = mp.Pipe()

            self.pipes[ee] = pipe_orchestrator

            p = mp.Process(target=self.run,
                           args=(env_config, pipe_env),
                           daemon=True)
            p.start()

    def __del__(self):

        self.send([(env_id, "close", None) for env_id in self.pipes.keys()])

    def run(self, env_config, pipe):

        env = get_env(env_config)

        while True:

            func, params = pipe.recv()

            if func == "close":
                break
            if func == "ping":
                pipe.send(("ping", params))
            elif func == "reset":
                pipe.send(("reset", env.reset(params)))
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
            try:
                self.pipes[env_id].send([func, params])
            except TypeError:
                ...


    def receive(self):
        responses = []

        for env_id, pipe in self.pipes.items():
            if pipe.poll():
                response = pipe.recv()
                responses.append((env_id, response))

        return responses

    def reset_all(self, initial_state_generator=None):
        """ Resets all environment. Blocks until all environments are reset.
            If a desired_state is not possible, caller has to resubmit desired_state"""

        # send ping with token to flush the pipes
        token = random.getrandbits(10)
        self.send([(env_id, "ping", token) for env_id in self.pipes.keys()])

        if initial_state_generator is None:
            desired_states = [None] * len(self.pipes)
        else:
            desired_states = [initial_state_generator(env_id=env_id) for env_id
                              in self.pipes.keys()]

        assert len(desired_states) == len(self.pipes)

        self.send([(env_id, "reset", (desired_state, False)) for env_id, desired_state
                   in zip(self.pipes.keys(), desired_states)])

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

            assert func == "observation space", f"'{func}' istead of 'observation space'"

        return self.observation_space_


if __name__ == "__main__":

    env_config = {

        "base_pkg": "robot-task-rl",
        "render": False,
        "task_config": {"name": "reach",
                        "dof": 3,
                        "only_positive": False
                        },
        "robot_config": {
            "name": "panda",
            "dof": 3
        }

    }

    nb_envs = 3

    orchestrator = Orchestrator(env_config, nb_envs)

    env_responses = orchestrator.reset_all()

    desired_state = {
        'robot': orchestrator.observation_dict['state']['robot'].sample(),
        'task': np.ones_like(
            orchestrator.observation_dict['state']['task'].sample())
    }

    print(desired_state)

    while True:
        requests = []
        for env_id, response in env_responses:
            func, data = response

            # print(type(data))

            # print(env_id, func, data)
            requests.append((env_id, "reset", desired_state))

        env_responses = orchestrator.send_receive(requests)
        # result = orchestrator.send_receive(
        #     [(ee, "reset", None) for ee in range(nb_envs)])

        # print(len(result))
