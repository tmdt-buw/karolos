import gym


class GymWrapper(gym.Env):

    def __init__(self, env, task_config, render, **kwargs):
        self.env = env
        self.step_counter = 0
        self.max_steps = task_config["max_steps"]

        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)

        self.step_counter += 1

        done |= self.step_counter >= self.max_steps

        return next_state, reward, done

    def reset(self):
        state = self.env.reset()

        return state

    def render(self, mode='human'):
        ...
