import numpy as np


class Task(object):

    def __init__(self,
                 bullet_client,
                 offset=(0, 0, 0),
                 max_steps=100,
                 parameter_distributions=None):

        if parameter_distributions is None:
            parameter_distributions = {}

        self.bullet_client = bullet_client
        self.offset = offset
        self.parameter_distributions = parameter_distributions

        self.step_counter = 0

        self.max_steps = max_steps

    def reset(self):
        if "gravity" in self.parameter_distributions:

            mean = self.parameter_distributions["gravity"].get("mean", (0, 0, -9.81))
            std = self.parameter_distributions["gravity"].get("std", (0, 0, 0))

            assert len(mean) == 3
            assert len(std) == 3

            gravity = np.random.normal(mean, std)

            self.bullet_client.setGravity(*gravity)

        self.step_counter = 0

    def reset_object(self, task_object, obstacles=None):
        if obstacles is None:
            obstacles = []

        while True:
            new_target_position = np.random.uniform(-1, 1, 3)

            if np.linalg.norm(new_target_position) < 0.8:

                new_target_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    task_object, new_target_position, [0, 0, 0, 1])

                for obstacle in obstacles:
                    points_contact = self.bullet_client.getContactPoints(
                        obstacle, task_object)

                    if len(points_contact):
                        continue

                break

    def step(self, robot=None):
        self.step_counter += 1

        observation = self.get_observation()

        return observation

    def convert_intervals(self, value, interval_origin, interval_target):

        value_mapped = value - interval_origin[0]
        value_mapped = value_mapped / (interval_origin[1] - interval_origin[0])
        value_mapped = value_mapped * (interval_target[1] - interval_target[0])
        value_mapped = value_mapped + interval_target[0]

        return value_mapped

    def get_observation(self):
        raise NotImplementedError()

    def randomize(self):
        gravity_z = np.random.normal(self.domain_randomization['gravity']['mean'],
                                     self.domain_randomization['gravity']['std'])
        self.bullet_client.setGravity(0, 0, gravity_z)

    def standard(self):
        self.bullet_client.setGravity(0, 0, self.domain_randomization['gravity']['mean'])
