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


class TaskReach(Task):

    def __init__(self, bullet_client, offset=(0, 0, 0),
                 max_steps=100, parameter_distributions=None):

        super(TaskReach, self).__init__(bullet_client=bullet_client,
                                        parameter_distributions=parameter_distributions,
                                        offset=offset,
                                        max_steps=max_steps)

        self.limits = np.array([
            (-.8, .8),
            (-.8, .8),
            (0., .8)
        ])

        self.state_space = spaces.Box(-1, 1, shape=(3,))

        self.target = bullet_client.createMultiBody(
            baseVisualShapeIndex=bullet_client.createVisualShape(p.GEOM_SPHERE,
                                                                 radius=.03,
                                                                 rgbaColor=[0, 1, 1, 1],
                                                                 ),
            baseCollisionShapeIndex=bullet_client.createCollisionShape(p.GEOM_SPHERE,
                                                                       radius=.03,
                                                                       ),
        )

        self.random = RandomState(
            int.from_bytes(os.urandom(4), byteorder='little'))

    def __del__(self):
        self.bullet_client.removeBody(self.target)

    @staticmethod
    def success_criterion(goal_info):
        goal_achieved = unwind_dict_values(goal_info["achieved"])
        goal_desired = unwind_dict_values(goal_info["desired"])

        goal_distance = np.linalg.norm(goal_achieved - goal_desired)
        return goal_distance < 0.05

    def reward_function(self, goal_info, done, **kwargs):
        if self.success_criterion(goal_info):
            reward = 1.
        elif done:
            reward = -1.
        else:
            goal_achieved = unwind_dict_values(goal_info["achieved"])
            goal_desired = unwind_dict_values(goal_info["desired"])

            reward = np.exp(-1 * np.linalg.norm(goal_achieved - goal_desired)) - 1
            reward /= 10

        return reward

    def reset(self, robot=None, state_robot=None, desired_state=None):

        super(TaskReach, self).reset()

        contact_points = True

        if desired_state is not None:
            desired_state = [np.interp(value, [-1, 1], limits)
                             for value, limits in
                             zip(desired_state, self.limits)]

            assert np.linalg.norm(
                desired_state) < 0.85, \
                f"desired_state puts target out of reach. " \
                f"{np.linalg.norm(desired_state)}"

            self.bullet_client.resetBasePositionAndOrientation(
                self.target, desired_state, [0, 0, 0, 1])

            self.bullet_client.stepSimulation()

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.model_id, self.target)
            else:
                contact_points = False

        while contact_points:

            target_position = np.random.uniform(self.limits[:, 0],
                                                self.limits[:, 1])

            if np.linalg.norm(target_position) < 0.8:
                target_position += self.offset
                self.bullet_client.resetBasePositionAndOrientation(
                    self.target, target_position, [0, 0, 0, 1])
                self.bullet_client.stepSimulation()
            else:
                continue

            if robot:
                contact_points = self.bullet_client.getContactPoints(
                    robot.model_id, self.target)
            else:
                contact_points = False

        state, goal_info, done = self.get_status(state_robot, robot)

        return state, goal_info

    def get_status(self, state_robot=None, robot=None):
        expert_action = None
        tcp_position = None

        if state_robot is not None and robot is not None:
            expert_action = self.get_expert_prediction(state_robot, robot)
            tcp_position = robot.get_tcp_position()

        position_desired, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_desired = np.array(position_desired)

        state = {
            "goal": {"tcp_position": position_desired}
        }

        done = self.step_counter >= self.max_steps

        goal_info = {
            'achieved': {
                "tcp_position": tcp_position,
            },
            'desired': {
                "tcp_position": position_desired,
            },
            "expert_action": expert_action,
            "steps": self.step_counter
        }

        return state, goal_info, done

    def get_expert_prediction(self, state_robot, robot: Robot):
        position_target, _ = self.bullet_client.getBasePositionAndOrientation(self.target)
        position_target = np.array(position_target)

        current_positions = [np.interp(position, [-1, 1], joint.limits) for joint, position in
                             zip(robot.joints, state_robot['joint_positions'])]

        current_positions_arm = np.array(current_positions[:-2])

        action = np.zeros(robot.action_space.shape)

        # move to target
        goal_position = position_target.copy()

        ik_result = robot.calculate_inverse_kinematics(goal_position, [1, 0, 0, 0], initial_pose=current_positions)

        if ik_result is None:
            return None

        delta_poses_arm = ik_result[:-2] - current_positions_arm

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

    task = TaskReach(p)

    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

    while True:
        obs = task.reset()

        for _ in np.arange(1. / time_step):
            p.stepSimulation()

            time.sleep(time_step)
