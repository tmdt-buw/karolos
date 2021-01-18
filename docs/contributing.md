# Contibuting

Contribute by implementing new functionalities, robots
or tasks. Test your new feature with some task/robot combination
(reach has the least amount of stress per unit of test ;-) ) 
and include a converging run with your merge request. 
If you implement a new task, also discuss your used
success_criterion and reward function, we would like
to avoid task specific reward shaping.


### Robots 

Added custom robots should use the same
functions as panda.py and ur5.py.
Pybullet allows for position, torque and velocity control of 
the robot. Torque and position control have been tested and work
on the current architecture.

```
robot.init              - read config and parameters; load robot urdf files;
                          define joints, links, observation and action space;

robot.reset             - reset the robot until no contact points with itself are left

robot.step              - execute action and return observation by get_observation()

robot.get_observation   - return tcp position, link velocities and posiitons. 
                          Strucutre is defined in robot.observation_space

robot.randomize         - Domain Randomization (DR) using config

robot.standardize       - Reset DR to standard case for testing

```

### Tasks

Custom tasks also have to reuse the same structure as
e.g. the reach task and furthermore inherit from the Base Task class.
Try to first achieve convergence using a non-specific success criterion 
and reward function (e.g. exponential decreasing distance measure) 
and gradually introduce more task-specific reward shaping.
If you need new objects like doors, tables or more, visit the urdf forums 
for available meshes.

```
task.init               - set gravity, cartesian offset, DR, limits and observation/goal space
                          load urdf files (-objects) for the task

task.reset              - reset the objects/targets until no contact points with robot

task.get_observation    - 
```