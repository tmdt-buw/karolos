# Tasks

## Reach
The goal of the agent is to move its manipulator or tool centre
point (TCP) to an area close to specified coordinates.

## Throw
This task places an object in the robot's gripper, which the robot ought to
throw such that the object (instead of the TCP) ends up
at specified coordinates outside of the robot's reach.

## Pick_Place
This task is the most challenging because a robot needs to
grasp an object and move it to certain coordinates.
Pick and place therefore requires skills that have to be acquired
by the reach task and the push task.
The reach task requires correct control of a robots kinematics
which is also required by the push task together with the
ability to control an objects position.