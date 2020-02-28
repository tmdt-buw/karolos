import multiprocessing as mp


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

        env = self.get_env(env_config)()

        while True:

            func, params = pipe.recv()

            if func == "close":
                break
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

    def get_env(self, env_config):
        base_pkg = env_config.pop("base_pkg")

        assert base_pkg in ["robot-task-rl"]

        if base_pkg == "robot-task-rl":
            from environments.environment_robot_task import Environment

            def env_init():
                env = Environment(**env_config)
                return env
        else:
            raise NotImplementedError(f"Unknown base package: {base_pkg}")

        return env_init

    def send_receive(self, actions=()):
        self.send(actions)
        return self.receive()


    def send(self, actions=()):

        for env_id, func, params in actions:

            print("send", [func, params])

            self.pipes[env_id].send([func, params])

    def receive(self):
        responses = []

        for env_id, pipe in self.pipes.items():

            if pipe.poll():
                response = pipe.recv()

                print("receive", response)

                responses.append((env_id, response))

        return responses

    def reset_all(self):

        responses = []

        # send reset commands
        for env_id, pipe in self.pipes.items():
            pipe.send(["reset", None])

            response = pipe.recv()

            func, state = response

            assert func == "reset"

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
