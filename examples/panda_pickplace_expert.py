import numpy as np
import pybullet as p

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from karolos.environments import get_env

if __name__ == "__main__":

    env_config = {
        "environment": "karolos",
        # "render": True,
        "task_config": {
            "name": "pick_place",
            "max_steps": 50
        },
        "robot_config": {
            "name": "panda",
            "scale": .1,
            "sim_time": 10
        }
    }

    env = get_env(env_config)

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )
    time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]


    for _ in range(100):
        state, goal_info = env.reset({"robot": np.zeros_like(env.state_space["robot"]["joint_positions"].sample())})
        # state, goal_info = env.reset()

        done = False
        while not done:
            action = goal_info["expert_action"]

            state, goal_info, done = env.step(action)

            done |= env.success_criterion(goal_info)

        print("Success:", env.success_criterion(goal_info), goal_info["steps"])