import multiprocessing as mp
import random
import threading
from copy import deepcopy
from multiprocessing import Lock

from . import get_env


class Orchestrator:

    def __init__(self, env_config, number_processes, number_threads=1):

        self.pipes = {}
        self.locks = {}

        self.state_space_ = None
        self.goal_space_ = None
        self.action_space_ = None
        self.reward_function_ = None
        self.success_criterion_ = None

        self.number_processes = number_processes
        self.number_threads = number_threads

        for iprocess in range(number_processes):

            pipes_process = []
            locks_process = []

            for ithread in range(number_threads):
                env_id = iprocess * number_threads + ithread

                pipe_main, pipe_process = mp.Pipe()
                lock = Lock()

                self.pipes[env_id] = pipe_main
                self.locks[env_id] = lock

                pipes_process.append(pipe_process)
                locks_process.append(lock)

            p = mp.Process(target=self.run_process,
                           args=(env_config, pipes_process, locks_process),
                           daemon=True)
            p.start()

    def __len__(self):
        return len(self.pipes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        for env_id in list(self.pipes.keys()):
            pipe = self.pipes.pop(env_id)
            lock = self.locks.pop(env_id)

            pipe.send(("close", None))

            with lock:
                # wait until env lock is acquired, meaning that the env was shutdown
                pass

    def run_process(self, env_config, pipes, locks):

        if len(pipes) > 1:

            threads = []

            for pipe, lock in zip(pipes, locks):
                thread = threading.Thread(target=self.run,
                                          args=(deepcopy(env_config), pipe, lock),
                                          daemon=True)
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        else:
            self.run(env_config, pipes.pop(), locks.pop())

    def run(self, env_config, pipe, lock):

        with lock:

            env = get_env(env_config)

            while True:
                func, params = pipe.recv()

                if func == "close":
                    break
                if func == "ping":
                    pipe.send(("ping", params))
                elif func == "reset":
                    try:
                        pipe.send(("reset", env.reset(params)))
                    except AssertionError as e:
                        pipe.send(("reset", e))
                elif func == "step":
                    pipe.send(("step", env.step(params)))
                elif func == "render":
                    pipe.send(("render", env.render(params)))
                elif func == "state space":
                    pipe.send(("state space", env.state_space))
                elif func == "goal space":
                    pipe.send(("goal space", env.goal_space))
                elif func == "action space":
                    pipe.send(("action space", env.action_space))
                elif func == "reward function":
                    pipe.send(("reward function", env.reward_function))
                elif func == "success criterion":
                    pipe.send(("success criterion", env.success_criterion))
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

        self.send([(env_id, "reset", desired_state) for env_id, desired_state
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
    def state_space(self):
        if self.state_space_ is None:
            self.pipes[0].send(["state space", None])
            func, self.state_space_ = self.pipes[0].recv()

            assert func == "state space", f"'{func}' instead of 'state space'"

        return self.state_space_

    @property
    def goal_space(self):
        if self.goal_space_ is None:
            self.pipes[0].send(["goal space", None])
            func, self.goal_space_ = self.pipes[0].recv()

            assert func == "goal space", f"'{func}' instead of 'goal space'"

        return self.goal_space_

    @property
    def action_space(self):

        if self.action_space_ is None:
            self.pipes[0].send(["action space", None])
            func, self.action_space_ = self.pipes[0].recv()

            assert func == "action space", f"'{func}' istead of 'action space'"

        return self.action_space_

    @property
    def reward_function(self):
        if self.reward_function_ is None:
            self.pipes[0].send(["reward function", None])
            func, self.reward_function_ = self.pipes[0].recv()

            assert func == "reward function", f"'{func}' instead of 'reward function'"

        return self.reward_function_

    @property
    def success_criterion(self):
        if self.success_criterion_ is None:
            self.pipes[0].send(["success criterion", None])
            func, self.success_criterion_ = self.pipes[0].recv()

            assert func == "success criterion", f"'{func}' instead of 'success criterion'"

        return self.success_criterion_
