import numpy as np


class Task(object):

    def __init__(self,
                 bullet_client,
                 gravity,
                 offset=(0, 0, 0),
                 dof=1,
                 only_positive=False):

        assert dof in [1, 2, 3]
        assert len(gravity) == 3

        self.bullet_client = bullet_client
        self.offset = offset
        self.dof = dof
        self.only_positive = only_positive

        bullet_client.setGravity(*gravity)

        self.step_counter = 0

    def reset(self):

        self.step_counter = 0

    def reset_object(self, task_object, obstacles=None):
        if obstacles is None:
            obstacles = []

        while True:
            new_target_position = np.zeros(3)

            for dd in range(self.dof):

                new_target_position[dd] = np.random.uniform(0, 1)

                if not self.only_positive:
                    new_target_position[dd] *= np.random.choice([1, -1])

            if np.linalg.norm(new_target_position) < 1:

                new_target_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    task_object, new_target_position, [0, 0, 0, 1])

                for obstacle in obstacles:
                    points_contact = self.bullet_client.getContactPoints(
                        obstacle, task_object)

                    if len(points_contact):
                        continue

                break

    def step(self):
        self.step_counter += 1

        success, observation = self.get_observation()

        return success, observation

    def convert_intervals(self, value, interval_origin, interval_target):

        value_mapped = value - interval_origin[0]
        value_mapped = value_mapped / (interval_origin[1] - interval_origin[0])
        value_mapped = value_mapped * (interval_target[1] - interval_target[0])
        value_mapped = value_mapped + interval_target[0]

        return value_mapped

    def get_observation(self):
        raise NotImplementedError()
