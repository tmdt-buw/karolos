import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from gym import spaces
from numpy.random import RandomState

from karolos.agents.utils import unwind_dict_values

sys.path.append(str(Path(__file__).resolve().parent))
from task import Task


class TaskPickPlace(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=50, accuracy=0.05, parameter_distributions=None):

        super(TaskPickPlace, self).__init__(bullet_client=bullet_client,
                                            parameter_distributions=parameter_distributions,
                                            offset=offset, max_steps=max_steps,
                                            gravity=(0, 0, -9.8))

        self.accuracy = accuracy

        self.limits = np.array([
            (.2, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.state_space = spaces.Dict({
            # "position": spaces.Box(-1, 1, shape=(3,)),
            "object_gripped": spaces.Box(-1, 1, shape=(1,)),
        })

        self.goal_space = spaces.Dict({
            "achieved": spaces.Dict({
                "object_position": spaces.Box(-1, 1, shape=(3,), dtype=np.float64),
            }),
            "desired": spaces.Dict({
                "object_position": spaces.Box(-1, 1, shape=(3,), dtype=np.float64),
            })
        })

        self.plane = bullet_client.loadURDF("plane.urdf")

        self.target = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_SPHERE, radius=.03, rgbaColor=[0, 1, 1, 1]),
        )

        self.tcp_object_constraint = None

        self.object = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_BOX, halfExtents=[.025] * 3),

            baseCollisionShapeIndex=bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=[.025] * 3),
            baseMass=.1,
        )

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    def __del__(self):
        self.bullet_client.removeBody(self.object)
        self.bullet_client.removeBody(self.target)
        self.bullet_client.removeBody(self.plane)

    def success_criterion(self, goal, *args, **kwargs):
        # goal -> object + tcp
        goal_achieved = unwind_dict_values(goal["achieved"]["object_position"])
        # desired -> both should be at same desired position
        goal_desired = unwind_dict_values(goal["desired"]["object_position"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        # more error margin
        return goal_distance < self.accuracy

    def reward_function(self, goal, done, *args, **kwargs):
        # 0.2* dist(tcp, obj), 0.8*dist(obj,goal)
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:
            goal_achieved_object = unwind_dict_values(goal["achieved"]['object_position'])
            goal_achieved_tcp = unwind_dict_values(goal["achieved"]['tcp_position'])
            goal_desired_object = unwind_dict_values(goal["desired"]['object_position'])
            goal_desired_tcp = unwind_dict_values(goal["desired"]['tcp_position'])

            # 0.8 * dist(obj, goal_obj), how close obj is to obj_goal
            reward_object = np.exp(-1 * np.linalg.norm(goal_desired_object - goal_achieved_object)) - 1

            # 0.2 * dist(obj, tcp), how close tcp is to obj
            reward_tcp = np.exp(-1 * np.linalg.norm(goal_achieved_tcp - goal_desired_tcp)) - 1

            # scale s.t. reward in [-1,1]
            reward = .8 * reward_object + .2 * reward_tcp
            reward /= self.max_steps
        return reward

    def reset(self, desired_state=None, desired_goal=None, robot=None, state_robot=None, force=False):

        super(TaskPickPlace, self).reset()

        if self.tcp_object_constraint is not None:
            self.bullet_client.removeConstraint(self.tcp_object_constraint)
            self.tcp_object_constraint = None

        if desired_state is None:
            desired_goal = {}

        if desired_state is None:
            desired_state = {}

        if desired_state.get("object_gripped", 0) > 0:
            self.tcp_object_constraint = self.bullet_client.createConstraint(self.object, -1,
                                                                             robot.model_id, robot.index_tcp,
                                                                             p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                                             parentFramePosition=[0, 0, 0],
                                                                             childFramePosition=[0, 0, 0])
            self.bullet_client.changeConstraint(self.tcp_object_constraint, maxForce=1e5)

        def complete_state(state_dict, space):
            for key in space:
                if key not in state_dict:
                    state_dict[key] = space[key].sample()
                elif type(space[key]) == spaces.Dict:
                    state_dict[key] = complete_state(state_dict[key], space[key])
            return state_dict

        desired_goal = complete_state(desired_goal, self.goal_space)
        desired_state = complete_state(desired_state, self.state_space)

        contact_points = False

        # reset until state is valid
        while True:
            desired_state_object = desired_goal["achieved"]["object_position"]
            desired_state_target = desired_goal["desired"]["object_position"]

            desired_state_object = np.array([np.interp(value, [-1, 1], limits)
                                             for value, limits in
                                             zip(desired_state_object, self.limits)])
            desired_state_target = np.array([np.interp(value, [-1, 1], limits)
                                             for value, limits in
                                             zip(desired_state_target, self.limits)])

            distance_target = np.linalg.norm(desired_state_target)

            desired_state_object[-1] = .01  # put on floor
            # desired_state_target[-1] = .01  # put on floor

            desired_state_object += self.offset
            desired_state_target += self.offset

            self.bullet_client.resetBasePositionAndOrientation(self.object, desired_state_object, [0, 0, 0, 1])
            self.bullet_client.resetBasePositionAndOrientation(self.target, desired_state_target, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(robot.model_id, self.object)

            if force or (not contact_points and distance_target < .8):
                break
            else:
                # try reset again with random state
                desired_goal = self.goal_space.sample()

        state, goal, done, info = self.get_state(state_robot, robot)

        return state, goal, info

    def step(self, state_robot=None, robot=None):
        if state_robot is not None and robot is not None:
            if state_robot["hand"] == 1:  # open
                if self.tcp_object_constraint is not None:
                    self.bullet_client.removeConstraint(self.tcp_object_constraint)
                    self.tcp_object_constraint = None
            elif state_robot["hand"] == 0:  # closing
                assert self.tcp_object_constraint is None

                position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
                position_object = np.array(position_object) - self.offset

                # todo: make theshold a parameter
                tcp_position, _ = robot.get_tcp_pose()
                if np.linalg.norm(tcp_position - np.array(position_object)) < .03:
                    self.tcp_object_constraint = self.bullet_client.createConstraint(self.object, -1,
                                                                                     robot.model_id, robot.index_tcp,
                                                                                     p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                                                                     parentFramePosition=[0, 0, 0],
                                                                                     childFramePosition=[0, 0, 0])
                    self.bullet_client.changeConstraint(self.tcp_object_constraint, maxForce=1e5)

        return super(TaskPickPlace, self).step(state_robot, robot)

    def get_state(self, state_robot=None, robot=None):
        expert_action = None
        tcp_position = None

        if state_robot is not None and robot is not None:
            tcp_position, _ = robot.get_tcp_pose()
            tcp_position = np.array(
                [np.interp(value, limits, [-1, 1]) for value, limits in zip(tcp_position, self.limits)])

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        position_object = np.array(position_object) - self.offset
        position_object = np.array(
            [np.interp(value, limits, [-1, 1]) for value, limits in zip(position_object, self.limits)])

        position_object_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_object_desired = np.array(position_object_desired) - self.offset
        position_object_desired = np.array(
            [np.interp(value, limits, [-1, 1]) for value, limits in zip(position_object_desired, self.limits)])

        state = {
            "object_gripped": float(self.tcp_object_constraint is not None)
        }

        achieved_goal = {
            "object_position": position_object,
            "tcp_position": tcp_position
        }

        desired_goal = {
            "object_position": position_object_desired,
            "tcp_position": position_object_desired
        }

        goal = {
            'achieved': achieved_goal,
            'desired': desired_goal,
        }

        info = {
            "expert_action": expert_action,
            "steps": self.step_counter
        }

        done = False

        return state, goal, done, info


if __name__ == "__main__":
    import pybullet_data as pd
    import time

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")

    task = TaskPickPlace(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
