from robot_task_environments.robots.Panda.panda import Panda
from robot_task_environments.robots.UR5.ur5 import UR5
import pybullet as p
import pybullet_utils.bullet_client as bc

bullet_client_panda = bc.BulletClient(p.DIRECT)
bullet_client_ur5 = bc.BulletClient(p.DIRECT)

panda = Panda(bullet_client_panda, sim_time=.1, scale=.1)
ur5 = UR5(bullet_client_ur5, sim_time=.1, scale=.02)

def generate_trajectory(robot):
    observations = []
    actions = []
    key_points = []

    observation = robot.reset()
    key_points.append(robot.get_key_points())

    observations.append(observation)

    for _ in range(100):
        action = panda.action_space.sample()
        observation = panda.step(action)

        actions.append(action)
        observations.append(observation)
        key_points.append(robot.get_key_points())

    return observations, actions, key_points

observations, actions, key_points = generate_trajectory(panda)

print(len(observations), len(actions))