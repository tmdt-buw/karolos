from environments.tasks.task import Task
import numpy as np
from gym import spaces


class Reach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 dof=1, only_positive=False, sparse_reward=False, max_steps=100):

        super(Reach, self).__init__(bullet_client=bullet_client,
                                    gravity=[0, 0, 0],
                                    offset=offset,
                                    dof=dof,
                                    only_positive=only_positive,
                                    sparse_reward=sparse_reward)

        self.limits = np.array([
            (-1., 1.),
            (-1., 1.),
            (-0.2, 1.)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(3,))

        self.target = self.bullet_client.loadURDF("objects/sphere.urdf", useFixedBase=True)

        self.max_steps = max_steps

    def reset(self, robot=None, desired_state=None):

        super(Reach, self).reset()

        contact_points = True

        if desired_state:
            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot, self.target)
            else:
                contact_points = False

        while contact_points:

            target_position = np.random.uniform(size=3)

            if not self.only_positive:
                target_position *= np.random.choice([1, -1], 3)

            if np.linalg.norm(target_position) < 0.8:
                target_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    self.target, target_position, [0, 0, 0, 1])
                self.bullet_client.stepSimulation()
            else:
                continue

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot, self.target)
            else:
                contact_points = False

        return self.get_observation()

    def get_target(self):

        position_target, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)

        position_target = np.array(position_target)

        return position_target

    def get_observation(self):

        position_target = self.get_target()

        observation = []

        for position_dimension_target, limits in zip(position_target,
                                                            self.limits):
            observation_target = self.convert_intervals(
                position_dimension_target, limits, [-1, 1])
            observation.append(observation_target)

        observation = np.array(observation)

        observation = observation.clip(self.observation_space.low,
                                       self.observation_space.high)
        return observation

    def get_goals(self, robot, success=True):
        if success:
            achieved_goal = robot.get_position_tcp()
        else:
            achieved_goal = None
        desired_goal = self.get_target()

        return achieved_goal, desired_goal

    def compute_reward(self, achieved_goal, desired_goal):

        if achieved_goal is not None:
            distance_tcp_object = np.linalg.norm(achieved_goal - desired_goal)

            if distance_tcp_object < 0.05:
                reward = 1.
            else:
                if self.sparse_reward:
                    reward = -1.
                else:
                    reward = np.exp(-distance_tcp_object * 3.5) * 2 - 1
                    reward /= self.max_steps
        else:
            reward = -1.

        reward = np.clip(reward, -1, 1)

        return reward

    def compute_done(self, achieved_goal, desired_goal):

        if achieved_goal is not None:
            distance_tcp_object = np.linalg.norm(achieved_goal - desired_goal)

            goal_reached = distance_tcp_object < 0.05

            done = goal_reached or self.step_counter >= self.max_steps

        else:
            done = True
            goal_reached = False

        return done, goal_reached


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

        p.stepSimulation()
