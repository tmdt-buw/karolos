import pybullet as p
import numpy as np
from gym import spaces


class Reach(object):

    def __init__(self, bullet_client, offset=(0, 0, 0), dof=1,
                 only_positive=False):

        assert dof in [1, 2, 3]

        self.limits_object = np.array([
            (-1., 1.),
            (-1., 1.),
            (-0.2, 1.)
        ])

        self.observation_space = spaces.Box(-1, 1, shape=(3,))

        self.bullet_client = bullet_client
        self.offset = offset
        self.dof = dof
        self.only_positive = only_positive

        bullet_client.setGravity(0, 0, 0)

        self.object = self.bullet_client.loadURDF("objects/box.urdf")

        self.max_steps = 5
        self.step_counter = 0

    def get_required_robot_dof(self):
        return self.dof

    def get_position_object(self):
        positions, _ = self.bullet_client.getBasePositionAndOrientation(
            self.object)

        return np.array(positions)

    def reset(self, robot=None):

        points_contact = None
        success = False
        observation = None
        new_object_position = None


        while points_contact is None or len(points_contact) or not success:

            if new_object_position is None or np.linalg.norm(new_object_position) > 1:
                new_object_position = np.zeros(3)

                for dd in range(self.dof):

                    new_object_position[dd] = np.random.uniform(0, 1)

                    if not self.only_positive:
                        new_object_position[dd] *= np.random.choice([1, -1])

            new_object_position += self.offset
            self.bullet_client.resetBasePositionAndOrientation(self.object,
                                                               new_object_position,
                                                               [0, 0, 0, 1])

            p.stepSimulation()
            if robot is not None:
                points_contact = self.bullet_client.getContactPoints(robot,
                                                                    self.object)
            else:
                points_contact = []

            success, observation = self.get_observation()

        self.step_counter = 0

        return observation

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
        state_object = self.get_position_object()

        observation = []

        for position_dimension_object, limits_object in zip(state_object,
                                                            self.limits_object):
            observation_object = self.convert_intervals(
                position_dimension_object, limits_object, [-1, 1])
            observation.append(observation_object)

        observation = np.array(observation)

        if not self.observation_space.contains(observation):
            observation = observation.clip(self.observation_space.low,
                                           self.observation_space.high)
            return False, observation
        else:
            return True, observation

    def calculate_reward(self, robot, success):

        if not success:
            done, reward = True, -10
        else:

            position_tcp = robot.get_position_tcp()

            position_object = self.get_position_object()

            distance_tcp_object = np.linalg.norm(position_tcp - position_object)

            if distance_tcp_object < 0.01:
                reward = 10
                done = True
            else:
                reward = 1 - 2 * (
                            np.sqrt(10 * distance_tcp_object) / np.sqrt(10))

                done = self.step_counter >= self.max_steps

                for position, limit in zip(position_object, self.limits_object):
                    done = done or (limit[0] >= position >= limit[1])

                reward = np.clip(reward, -1, 1)

        return done, reward


if __name__ == "__main__":
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
