import warnings

import numpy as np


class Task(object):
    """
    Parent class for all tasks
    """
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
    def success_criterion(goal):
        """
        define the success criterion for each task
        :param goal_info:   a dictionary passed to experiment.py containing
                            info about a goal (reached/desired)
        :return: whether success_criterion is reached
        """
        raise NotImplementedError()

    def reward_function(self, goal, done, **kwargs):
        """
        define the reward function using goal_info
        rewards should be normalized
        :param done: Markov decision process - terminal boolean
        :param goal_info: goal_info
        :param kwargs:
        :return: the reward
        """
        raise NotImplementedError()

    def reset(self):
        """
        resets the task
        :return:
        """
        gravity_distribution = self.parameter_distributions.get("gravity", {})

        mean = gravity_distribution.get("mean", (0, 0, -9.81))
        std = gravity_distribution.get("std", (0, 0, 0))

        assert len(mean) == 3
        assert len(std) == 3

        gravity = np.random.normal(mean, std)

        self.bullet_client.setGravity(*gravity)

        self.step_counter = 0

    def step(self, state_robot=None, robot=None):
        """
        step in the task
        observation_robot is required to determine e.g. collisions
        :param observation_robot: robots observation
        :return: observation of the task, goal_info, terminal (done)
        """
        self.step_counter += 1

        state_task, goal, done, info = self.get_state(state_robot, robot)

        done |= self.step_counter >= self.max_steps

        return state_task, goal, done, info

    def get_state(self, state_robot=None, robot=None):
        """
        get status of task
        :param observation_robot:
        :return:
        """
        raise NotImplementedError()

    def get_expert_action(self, state_robot, robot):
        warnings.warn(
            'The task does not have an expert policy to query and thus cannot be used for Imitation Learning.')
        return None
