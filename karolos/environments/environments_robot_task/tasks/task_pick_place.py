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


class TaskPickPlace(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=50, parameter_distributions=None):

        super(TaskPickPlace, self).__init__(bullet_client=bullet_client,
                                            parameter_distributions=parameter_distributions,
                                            offset=offset, max_steps=max_steps,
                                            gravity=(0, 0, -9.8))

        self.limits = np.array([
            (.2, .8),
            (-.8, .8),
            (0., .0)
        ])

        self.state_space = spaces.Dict({
            "position": spaces.Box(-1, 1, shape=(3,)),
        })

        self.goal_space = spaces.Dict({
            "achieved": spaces.Box(-1, 1, shape=(3,)),
            "desired": spaces.Box(-1, 1, shape=(3,)),
        })

        bullet_client.loadURDF("plane.urdf")

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

    @staticmethod
    def success_criterion(goal, **kwargs):
        # goal -> object + tcp
        goal_achieved = unwind_dict_values(goal["achieved"]["object_position"])
        # desired -> both should be at same desired position
        goal_desired = unwind_dict_values(goal["desired"]["object_position"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        # more error margin
        return goal_distance < 0.05

    def reward_function(self, goal, done, **kwargs):
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
            reward_object = np.exp(-5 * np.linalg.norm(goal_desired_object - goal_achieved_object)) - 1

            # 0.2 * dist(obj, tcp), how close tcp is to obj
            reward_tcp = np.exp(-5 * np.linalg.norm(goal_achieved_tcp - goal_desired_tcp)) - 1

            # scale s.t. reward in [-1,1]
            reward = .8 * reward_object + .2 * reward_tcp
        return reward

    def reset(self, desired_state=None, robot=None, state_robot=None):

        super(TaskPickPlace, self).reset()

        if self.tcp_object_constraint is not None:
            self.bullet_client.removeConstraint(self.tcp_object_constraint)
            self.tcp_object_constraint = None

        if desired_state is not None:
            # gripped flag irrelevant for initial state
            desired_state_object, desired_state_target = np.split(
                np.array(desired_state), 2)

            desired_state_object = np.array([np.interp(value, [-1, 1], limits)
                                    for value, limits in
                                    zip(desired_state_object, self.limits)])
            desired_state_target = np.array([np.interp(value, [-1, 1], limits)
                                    for value, limits in
                                    zip(desired_state_target, self.limits)])

            assert np.linalg.norm(desired_state_object) < 0.8, \
                f"desired_state puts object out of reach. {desired_state_object}"

            assert np.linalg.norm(desired_state_target) < 0.8, \
                f"desired_state puts target out of reach. {desired_state_target}"

            desired_state_object[-1] = 0  # put on floor
            desired_state_target[-1] = max(.1, desired_state_target[-1])  # put on floor

            desired_state_object += self.offset
            desired_state_target += self.offset

            self.bullet_client.resetBasePositionAndOrientation(self.object, desired_state_object, [0, 0, 0, 1])
            self.bullet_client.resetBasePositionAndOrientation(self.target, desired_state_target, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.model_id, self.object)

                assert not contact_points, f"desired_state puts object and " \
                                           f"robot in collision"

        else:
            contact_points = True

            while contact_points:
                object_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])
                object_position[-1] = 0  # put object on floor and use gravity
                target_position = np.random.uniform(self.limits[:, 0],
                                                    self.limits[:, 1])
                target_position[-1] = max(.1, target_position[-1])  # put on floor

                if np.linalg.norm(object_position) < 0.8 and np.linalg.norm(target_position) < 0.8:
                    object_position += self.offset
                    target_position += self.offset

                    self.bullet_client.resetBasePositionAndOrientation(self.object, object_position, [0, 0, 0, 1])

                    self.bullet_client.resetBasePositionAndOrientation(self.target, target_position, [0, 0, 0, 1])
                    self.bullet_client.stepSimulation()
                else:
                    continue

                if robot:
                    contact_points = self.bullet_client.getContactPoints(
                        robot.model_id, self.object)
                else:
                    contact_points = False

        state, goal, done, info = self.get_status(state_robot, robot)

        return state, goal, info

    def step(self, state_robot=None, robot=None):
        if state_robot is not None and robot is not None:
            if state_robot["status_hand"] == 1:  # open
                if self.tcp_object_constraint is not None:
                    self.bullet_client.removeConstraint(self.tcp_object_constraint)
                    self.tcp_object_constraint = None
            elif state_robot["status_hand"] == 0:  # closing
                assert self.tcp_object_constraint is None

                position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)

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

    def get_status(self, state_robot=None, robot=None):
        expert_action = None
        tcp_position = None

        if state_robot is not None and robot is not None:
            expert_action = self.get_expert_action(state_robot, robot)
            tcp_position, _ = robot.get_tcp_pose()

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        position_object = np.array(position_object)

        position_object_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_object_desired = np.array(position_object_desired)

        state = {}

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

    def get_expert_action(self, state_robot, robot: Robot):
        LOWER_LIMITS = np.array([joint.limits[0] for joint in robot.joints])
        UPPER_LIMITS = np.array([joint.limits[1] for joint in robot.joints])
        JOINT_RANGES = np.array([ul - ll for ul, ll in zip(UPPER_LIMITS, LOWER_LIMITS)])
        REST_POSE = LOWER_LIMITS + .5 * JOINT_RANGES

        position_object, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        position_object = np.array(position_object)

        position_object_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_object_desired = np.array(position_object_desired)

        object_angle = np.arctan2(position_object[1], position_object[0])
        tcp_position, _ = robot.get_tcp_pose()

        current_positions = [np.interp(position, [-1, 1], joint.limits) for joint, position in
                             zip(robot.joints, state_robot['joint_positions'])]

        # current_positions = REST_POSE

        current_positions_arm = np.array(current_positions[:len(robot.joints_arm)])

        desired_pose_arm = REST_POSE[:len(robot.joints_arm)]
        desired_pose_arm[0] = object_angle

        action = np.zeros(robot.action_space.shape)

        if np.linalg.norm(position_object - tcp_position) > .01:
            # move to object with open gripper
            action[len(robot.joints_arm):] = 1

            goal_position = position_object.copy()

            if np.linalg.norm(position_object[:2] - tcp_position[:2]) > .02:
                # align over object
                goal_position[-1] += .1
        else:
            if self.tcp_object_constraint is not None:
                assert Robot.STATUS_HAND(state_robot["status_hand"][0]) is not Robot.STATUS_HAND.OPEN, state_robot["status_hand"]

                # object is gripped, keep gripper closed
                action[len(robot.joints_arm):] = -1

                if Robot.STATUS_HAND(state_robot["status_hand"][0]) is Robot.STATUS_HAND.CLOSED:
                    goal_position = position_object_desired.copy()
                    if np.linalg.norm(position_object[:2] - position_object_desired[:2]) > .02:
                        goal_position[-1] += .1
                else:
                    goal_position = None
            else:
                # object not gripped
                if Robot.STATUS_HAND(state_robot["status_hand"][0]) is Robot.STATUS_HAND.CLOSED:
                    # hand closed, but object not gripped --> open hand
                    action[len(robot.joints_arm):] = 1
                else:
                    # close gripper
                    action[len(robot.joints_arm):] = -1

                # do not move
                goal_position = None

        if goal_position is not None:
            ik_result = robot.calculate_inverse_kinematics(goal_position, [1, 0, 0, 0], initial_pose=current_positions)

            if ik_result is None:
                return None

            delta_poses_arm = ik_result[:len(robot.joints_arm)] - current_positions_arm

            action_arm = [delta_pose / (joint.limits[1] - joint.limits[0]) / robot.scale
                          for (_, joint), delta_pose in zip(robot.joints_arm.items(), delta_poses_arm)]

            action_arm = np.clip(action_arm, -1, 1)
            action[:len(robot.joints_arm)] = action_arm

        return action


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
