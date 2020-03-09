from environments.tasks.metatask import MetaTask
import numpy as np


class MoveBox(MetaTask):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 goal_box=(0.5, 0.5, 0.5),
                 dof=1,
                 only_positive=False):

        super().__init__(bullet_client=bullet_client,
                         object_path="objects/box.urdf",
                         gravity=[0, 0, -9.81],         # assuming 1kg box weight
                         offset=offset,
                         dof=dof,
                         only_positive=only_positive)

        self.goal_box = goal_box

    def calculate_reward(self, robot, success):

        goal_reached = False

        if not success:
            done, reward = True, -1.
        else:
            position_object = super().get_position_object()

            distance_object = np.linalg.norm(position_object - self.goal_box)

            if distance_object < 0.01:
                reward = 1.
                done = True
                goal_reached = True
            else:
                reward = np.exp(-distance_object * 3.5) * 2 - 1
                reward /= self.max_steps

                done = self.step_counter >= self.max_steps

                for position, limit in zip(position_object, self.limits_object):
                    done = done or (limit[0] >= position >= limit[1])
        reward = np.clip(reward, -1, 1)

        return done, reward, goal_reached