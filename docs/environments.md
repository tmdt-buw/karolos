# Environments

The main focus of Karolos is on robotics which is why
a modular implementation of robots and tasks these ought
to perform is provided. 
Each robot and task have their own class and each task
and robot can be combined together to form an environment.
This happens in the file karolos/environments/robot_task_environments/environment_robot_task.py.
Implemented robots are currently an IIWA, a Pandas and an UR5 robot
using pybullet. 
Possible tasks for each of these robots are pick and place,
push and reach.
Pick and place tasks the robot with grasping an object and
placing it at designated coordinates.
Push tasks require the robot to move an object in two
dimensional space (i.e. on the ground plane) to designated coordinates.
The reach tasks is simple in the sense that a robot is only required
to move its manipulator to specific coordinates.
Coordinates where a "goal" is are part of the agents state, which enables
optimisation of the learning process by HER.

Benchmarking algorithms and the karolos architecture


