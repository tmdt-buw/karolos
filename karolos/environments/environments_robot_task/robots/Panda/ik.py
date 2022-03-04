import klampt
import numpy as np
import pybullet as p
from klampt import vis
from klampt.math import so3
from klampt.model import ik
import time
from panda import Panda

p.connect(p.DIRECT)
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

for tt in np.linspace(.1, .8, 50):
    print(tt)
    obj = ik.objective(robot.link(robot.numLinks() - 1),R=so3.identity(),t=[tt,0,0])
    # solver = ik.solver(obj)
    res = ik.solve_global(obj, activeDofs=dof_joint_ids)
    if not res: print("IK failure!")
    print(res)

    vis.update()
    time.sleep(.5)