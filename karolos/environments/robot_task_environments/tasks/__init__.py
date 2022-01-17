import numpy as np


class Task(object):

    def __init__(self,
                 bullet_client,
                 offset=(0, 0, 0),
                 max_steps=100,
                 parameter_distributions=None,
                 gravity=(0, 0, 0)):
        if parameter_distributions is None:
            parameter_distributions = {}

        self.bullet_client = bullet_client
        self.offset = offset
        self.parameter_distributions = parameter_distributions

        self.step_counter = 0
        self.bullet_client.setGravity(*gravity)

        self.max_steps = max_steps

    @staticmethod
    def success_criterion(goal_info):
        raise NotImplementedError()

    def reward_function(self, done, goal_info, **kwargs):
        raise NotImplementedError()

    # TODO should we pass robot into method? What if gravity not in self.parameter distirbutions?
    def reset(self):
        gravity_distribution = self.parameter_distributions.get("gravity", {})

        mean = gravity_distribution.get("mean", (0, 0, -9.81))
        std = gravity_distribution.get("std", (0, 0, 0))

        assert len(mean) == 3
        assert len(std) == 3

        gravity = np.random.normal(mean, std)

        self.bullet_client.setGravity(*gravity)

        self.step_counter = 0

    def step(self, observation_robot, robot):
        self.step_counter += 1

        observation_task, goal_info, done = self.get_status(observation_robot)

        done |= self.step_counter >= self.max_steps

        return observation_task, goal_info, done

    def get_status(self, observation_robot):
        raise NotImplementedError()


def get_task(task_config, bullet_client):
    task_name = task_config.pop("name")

    if task_name == 'reach':
        from .reach import Reach
        task = Reach(bullet_client, **task_config)
    elif task_name == 'pick_place':
        from .pick_place import Pick_Place
        task = Pick_Place(bullet_client, **task_config)
    else:
        raise ValueError()

    return task
