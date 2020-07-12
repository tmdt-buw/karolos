import multiprocessing as mp
import random
from environments import get_env
import pybullet_data as pd
import pybullet as p
import pybullet_utils.bullet_client as bc
import threading
from copy import deepcopy


class ThreadOrchestrator(object):

    def __init__(self, env_config,num_thread):

        self.pipes = {}
        self.action_space_ = None
        self.observation_space_ = None

        self.thread_list = []
        for i in range(num_thread):
            pipe_orchestrator, pipe_env = mp.Pipe()
            self.pipes[i] = pipe_orchestrator

            env_config["robot_config"]["offset"] = (0, 2 * i, 0)
            env_config["task_config"]["offset"] = (0 , 2 * i, 0)

            t = threading.Thread(target=self.run,
                                 name='Thread {}'.format(i),
                                 args=(env_config,pipe_env))
            self.thread_list.append(t)
            t.start()

    def run(self, env_config, pipe):

        env = get_env(deepcopy(env_config))()

        while True:

            func, params = pipe.recv()

            print("pipe.recv()* fun : {}  and params : {} ".format(func, params))

            if func == "close":
                break
            if func == "ping":
                pipe.send(("ping", params))
            elif func == "reset":
                pipe.send(("reset", env.reset()))
                print(("reset", env.reset()))
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

            print("env_id, func, params is**:",env_id, func, params)
            if func == "step":
                if not self.action_space.contains(params):
                    print(params)

            self.pipes[env_id].send([func, params])
    def receive(self):
        responses = []

        for env_id, pipe in self.pipes.items():
            print("self.pipes.items()")
            if pipe.poll():
                response = pipe.recv()

                responses.append((env_id, response))

        return responses

    def reset_all(self):
        """Resets all environment. Blocks until all environments are reset."""

        token = random.getrandbits(10)

        # send ping with token to flush the pipes
        self.send([(env_id, "ping", token) for env_id in self.pipes.keys()])
        self.send([(env_id, "reset", None) for env_id in self.pipes.keys()])

        #required_env_ids = self.pipes.keys()

        responses = []

        # send reset commands
        for env_id, pipe in self.pipes.items():
            while True:
                func, data = pipe.recv()

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
    import numpy as np
    connection_mode = p.GUI
    bullet_client = bc.BulletClient(connection_mode)
    bullet_client.setAdditionalSearchPath(pd.getDataPath())
    bullet_client.setGravity(0, 0, -9.8)
    time_step = 1. / 60.
    bullet_client.setTimeStep(time_step)
    bullet_client.setRealTimeSimulation(0)
    i=0
    env_config = {
        "base_pkg": "robot-task-rl",
        "render": True,
        "bullet_client": bullet_client,
        "task_config": {"name": "reach",
                        "dof": 3,
                        "offset": (0, i, 0),
                        "only_positive": False,
                        "sparse_reward": False,
                        "max_steps": 25
                        },
        "robot_config": {
            "name": "panda",
            "offset": (0, 0, 0),
            "dof": 3,
            "sim_time": .1,
            "scale": .1
        }
    }

    num_thread=3
    orchestrator = ThreadOrchestrator(env_config,num_thread)

    #print(orchestrator.action_space.shape)
    #print(orchestrator.send_receive(
        #[(0, "action space", None)]))

    responses = orchestrator.reset_all()
    requests = []
    # requests = [(ee, "step", np.random.random(7)) for ee in range(nb_envs)]
    num_execution = 200
    while num_execution > 0:

        for response in responses:

            env_id = response[0]

            requests.append((env_id, "reset", None))
            #print("requests type ",type(requests))
            #action = np.random.uniform(-1, 1, 7)

            #if np.random.random() > .9:
            #    requests.append((env_id, "reset", None))

            #else:
            #    requests.append((env_id, "step", action))

        responses = orchestrator.send_receive(requests)
        requests=[]
        num_execution -= len(responses)
        print("num_execution : ", num_execution)
    print(num_execution)
