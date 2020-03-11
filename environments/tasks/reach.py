from environments.tasks.task import Task
import numpy as np


class Reach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 dof=1,
                 only_positive=False):

        super().__init__(bullet_client=bullet_client,
                         object_path="objects/sphere.urdf",
                         gravity=[0, 0, 0],
                         offset=offset,
                         dof=dof,
                         only_positive=only_positive)

    def calculate_reward(self, robot, success):

        goal_reached = False

        if not success:
            done, reward = True, -1.
        else:

            position_tcp = robot.get_position_tcp()
            position_object = super().get_position_object()

            distance_tcp_object = np.linalg.norm(position_tcp - position_object)

            if distance_tcp_object < 0.05:
                reward = 1.
                done = True
                goal_reached = True
            else:
                reward = np.exp(-distance_tcp_object * 3.5) * 2 - 1
                # reward = -distance_tcp_object
                reward /= self.max_steps

                done = self.step_counter >= self.max_steps

                for position, limit in zip(position_object, self.limits_object):
                    done = done or (limit[0] >= position >= limit[1])

        reward = np.clip(reward, -1, 1)

        return done, reward, goal_reached


if __name__ == "__main__":
    import pybullet as p
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    time_step = 1. / 60.
    p.setTimeStep(time_step)
    p.setRealTimeSimulation(0)

    task = Reach(p, dof=3)

    while True:
        p.stepSimulation()

        time.sleep(time_step)

        success = True

        obs = task.reset()

        print()
        print(obs)

        while success:
            success, obs = task.step()

            print(success, obs)

            p.stepSimulation()
