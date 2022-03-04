import klampt
import numpy as np
import pybullet as p
from klampt import vis

from panda import Panda

p.connect(p.GUI)
# p.setAdditionalSearchPath(pd.getDataPath())

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=70,
                             cameraPitch=-27,
                             cameraTargetPosition=(0, 0, 0)
                             )
p.setRealTimeSimulation(0)

p.setGravity(0, 0, -9.81)

panda = Panda(p, sim_time=.1, scale=.1)

world = klampt.WorldModel()
world.loadElement("panda.urdf")
robot = world.robot(0)

dof_joint_ids = []

for jj in range(robot.numLinks()):
    if robot.getJointType(jj) != "weld":
        dof_joint_ids.append(jj)

vis.add("base", [0, 0, 0])
vis.add("world", world)  # shows the robot in the solved configuration
vis.show()  # this will pop up the visualization window until you close it

while True:
    # continue
    state = panda.get_state()

    joint_positions = []

    for joint, position_normed in zip(panda.joints, state["joint_positions"]):
        position = np.interp(position_normed, [-1, 1], joint.limits)
        joint_positions.append(position)

    print(joint_positions[0])
    print(state["tcp_position"])
    p.stepSimulation()

    for jj, position in zip(dof_joint_ids, joint_positions):
        robot.setDOFPosition(jj, position)

    vis.update()
