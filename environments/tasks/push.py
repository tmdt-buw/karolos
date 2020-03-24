from environments.tasks.task import Task
import numpy as np
from gym import spaces


class Push(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0), dof=1,
                 only_positive=False, sparse_reward=False, max_steps=10):

        super(Push, self).__init__(bullet_client=bullet_client,
                                   gravity=[0, 0, -9.81],
                                   # assuming 1kg box weight
                                   offset=offset,
                                   dof=dof,
                                   only_positive=only_positive,
                                   sparse_reward=sparse_reward)

        self.limits = np.array([
            (-1., 1.),
            (-1., 1.),
            (0., 1.)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(6,))

        # add plane to place box on
        bullet_client.loadURDF("plane.urdf")

        self.target = self.bullet_client.loadURDF("objects/sphere.urdf", useFixedBase=True)
        self.object = self.bullet_client.loadURDF("objects/box.urdf")

        # self.camera_config = {
        #     "width": 200,
        #     "height": 200,
        #
        #     "viewMatrix": p.computeViewMatrix(
        #         cameraEyePosition=[0, 0, 2],
        #         cameraTargetPosition=[0, 0, 0],
        #         cameraUpVector=[0, 1, 0]),
        #
        #     "projectionMatrix": p.computeProjectionMatrixFOV(
        #         fov=60.0,
        #         aspect=1.0,
        #         nearVal=0.01,
        #         farVal=10),
        # }
        # width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        #     **self.camera_config)

        self.max_steps = max_steps

    def reset(self, robot=None):

        super(Push, self).reset()

        observation = None
        success = False

        while not success:
            self.reset_object(self.target, robot)
            self.reset_object(self.object, robot)
            success, observation = self.get_observation()

        return observation

    def get_target(self):

        position_target, _ = self.bullet_client.getBasePositionAndOrientation(
            self.target)

        position_target = np.array(position_target)

        return position_target

    def get_object(self):
        position_object, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        position_object = np.array(position_object)

        return position_object

    def get_observation(self):

        position_target = self.get_target()
        position_object = self.get_object()

        observation = []

        for position_dimension_target, limits in zip(position_target,
                                                     self.limits):
            observation_target = self.convert_intervals(
                position_dimension_target, limits, [-1, 1])
            observation.append(observation_target)

        for position_dimension_object, limits in zip(position_object,
                                                            self.limits):
            observation_object = self.convert_intervals(
                position_dimension_object, limits, [-1, 1])
            observation.append(observation_object)

        observation = np.array(observation)

        if not self.observation_space.contains(observation):
            observation = observation.clip(self.observation_space.low,
                                           self.observation_space.high)
            return False, observation
        else:
            return True, observation

    def get_goals(self, robot, success):
        if success:
            achieved_goal = self.get_object()
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
                    reward = -1
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

    task = Push(p, dof=3)

    while True:
        p.stepSimulation()

        time.sleep(time_step)

        success = True

        obs = task.reset()

        print()
        print(obs)

        p.stepSimulation()

        input()
