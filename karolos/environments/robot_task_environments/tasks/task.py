import numpy as np
from utils import unwind_dict_values


class Task(object):

    def __init__(self,
                 bullet_client,
                 offset=(0, 0, 0),
                 max_steps=100,
                 parameter_distributions=None,
                 gravity=(0,0,0)):

        if parameter_distributions is None:
            parameter_distributions = {}

        self.bullet_client = bullet_client
        self.offset = offset
        self.parameter_distributions = parameter_distributions

        self.step_counter = 0
        self.bullet_client.setGravity(*gravity)

        self.max_steps = max_steps

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

    # todo we dont need reset_object as it is already different for push/pick&place task
    # def reset_object(self, task_object, obstacles=None):
    #     if obstacles is None:
    #         obstacles = []
    #
    #     while True:
    #         new_target_position = np.random.uniform(-1, 1, 3)
    #
    #         if np.linalg.norm(new_target_position) < 0.8:
    #
    #             new_target_position += self.offset
    #             self.bullet_client.resetBasePositionAndOrientation(
    #                 task_object, new_target_position, [0, 0, 0, 1])
    #
    #             for obstacle in obstacles:
    #                 points_contact = self.bullet_client.getContactPoints(
    #                     obstacle, task_object)
    #
    #                 if len(points_contact):
    #                     continue
    #
    #             break

    def step(self, observation_robot):
        self.step_counter += 1

        observation_task = self.get_observation()
        achieved_goal, desired_goal, done = self.get_status(observation_robot)

        return observation_task, achieved_goal, desired_goal, done

    def convert_intervals(self, value, interval_origin, interval_target):

        value_mapped = value - interval_origin[0]
        value_mapped = value_mapped / (interval_origin[1] - interval_origin[0])
        value_mapped = value_mapped * (interval_target[1] - interval_target[0])
        value_mapped = value_mapped + interval_target[0]

        return value_mapped

    def reward_function(self, done, goal, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def success_criterion(goal):
        goal_achieved = unwind_dict_values(goal["achieved"])
        goal_desired = unwind_dict_values(goal["desired"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        return goal_distance < 0.01
