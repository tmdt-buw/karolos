import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from gym import spaces
from numpy.random import RandomState

from karolos.agents.utils import unwind_dict_values
from karolos.environments.environments_robot_task.robots.robot import Robot

sys.path.append(str(Path(__file__).resolve().parent))
from task import Task


class TaskThrow(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=100, parameter_distributions=None):

        super(TaskThrow, self).__init__(bullet_client=bullet_client,
                                        parameter_distributions=parameter_distributions,
                                        offset=offset,
                                        max_steps=max_steps)

        self.distance = 2.5

        self.limits = np.array([
            (-self.distance, self.distance),
            (-self.distance, self.distance),
            (0., .8)
        ])

        self.state_space = spaces.Dict({})

        self.goal_space = spaces.Dict({
            "achieved": spaces.Box(-1, 1, shape=(3,)),
            "desired": spaces.Box(-1, 1, shape=(3,)),
        })

        self.target = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_SPHERE,
                                                                 radius=.5,
                                                                 rgbaColor=[0, 1, 1, 1],
                                                                 ),
        )

        self.tcp_object_constraint = None

        self.object = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_BOX, halfExtents=[.025] * 3),

            baseCollisionShapeIndex=bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=[.025] * 3),
            baseMass=1.,
        )

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    def __del__(self):
        self.bullet_client.removeBody(self.object)
        self.bullet_client.removeBody(self.target)

    @staticmethod
    def success_criterion(goal, **kwargs):
        goal_achieved = unwind_dict_values(goal["achieved"])
        goal_desired = unwind_dict_values(goal["desired"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        return goal_distance < 0.5

    def reward_function(self, goal, done, **kwargs):
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:
            goal_achieved = unwind_dict_values(goal["achieved"])
            goal_desired = unwind_dict_values(goal["desired"])

            goal_distance = np.linalg.norm(goal_achieved - goal_desired)

            reward = np.exp(-goal_distance) - 1
            reward /= 10

        return reward

    def reset(self, desired_state=None, robot=None, state_robot=None):

        super(TaskThrow, self).reset()

        # attach object to tcp
        if robot is not None:
            tcp_position, _ = robot.get_tcp_pose()
            self.bullet_client.resetBasePositionAndOrientation(self.object, tcp_position, [0, 0, 0, 1])

            robot.status_hand = Robot.STATUS_HAND.CLOSED

            if self.tcp_object_constraint is None:
                self.tcp_object_constraint = self.bullet_client.createConstraint(self.object, -1,
                                                                                 robot.model_id, robot.index_tcp,
                                                                                 p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                                                 parentFramePosition=[0, 0, 0],
                                                                                 childFramePosition=[0, 0, 0])

        contact_points = True

        if desired_state is not None:
            angle = np.arctan2(desired_state[1], desired_state[0])

            desired_state = np.array([
                self.distance * np.cos(angle),
                self.distance * np.sin(angle),
                0
            ])

            desired_state += self.offset

            self.bullet_client.resetBasePositionAndOrientation(self.target, desired_state, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.model_id, self.target)
            else:
                contact_points = False

        while contact_points:

            angle = np.random.uniform(0, 2 * np.pi)

            target_position = np.empty(3)

            target_position[0] = self.distance * np.cos(angle)
            target_position[1] = self.distance * np.sin(angle)
            target_position[-1] = 0

            target_position += self.offset
            self.bullet_client.resetBasePositionAndOrientation(self.target, target_position, [0, 0, 0, 1])

            if robot:
                contact_points = self.bullet_client.getContactPoints(robot.model_id, self.target)
            else:
                contact_points = False

        state, goal, done, info = self.get_state(state_robot, robot)

        return state, goal, info

    def step(self, state_robot=None, robot=None):
        if state_robot is not None and robot is not None:
            if state_robot["status_hand"] == 1:  # open
                if self.tcp_object_constraint is not None:
                    # release object
                    self.bullet_client.removeConstraint(self.tcp_object_constraint)
                    self.tcp_object_constraint = None

        return super(TaskThrow, self).step(state_robot, robot)

    def get_state(self, state_robot=None, robot=None):
        expert_action = None

        if state_robot is not None and robot is not None:
            expert_action = self.get_expert_action(state_robot, robot)

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        position_object = np.array(position_object)

        velocity_object, _ = self.bullet_client.getBaseVelocity(self.object)

        position_object_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_object_desired = np.array(position_object_desired)

        state = {}

        goal = {
            'achieved': {
                "object_position": position_object,
            },
            'desired': {
                "object_position": position_object_desired,
            },
        }

        done = self.step_counter >= self.max_steps

        info = {
            "expert_action": expert_action,
            "steps": self.step_counter
        }

        return state, goal, done, info

    def get_expert_action(self, state_robot, robot: Robot):
        return super(TaskThrow, self).get_expert_action(state_robot, robot)


if __name__ == "__main__":
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = TaskThrow(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
