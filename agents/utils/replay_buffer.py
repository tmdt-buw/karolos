import random
from collections import deque, namedtuple

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, reward_function):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])

        self.reward_function = reward_function

    def add(self, state, action, next_state, done, goal):
        """Add a new experience to memory."""

        reward = self.reward_function(state=state, action=action,
                                      next_state=next_state, done=done,
                                      goal=goal)

        experience = {
            "state": np.concatenate([state["robot"], state["task"]], axis=-1),
            "goal": goal["desired"],
            "action": action,
            "reward": reward,
            "next_state": np.concatenate(
                [next_state["robot"], next_state["task"]], axis=-1),
            "done": done
        }

        self.memory.append(experience)

        return reward

    def sample(self, n_samples):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.choices(self.memory, k=n_samples)

        states = np.stack([experience["state"] for experience in experiences])
        goals = np.stack([experience["goal"] for experience in experiences])
        actions = np.stack(
            [experience["action"] for experience in experiences])
        rewards = np.array(
            [experience["reward"] for experience in experiences])
        next_states = np.stack(
            [experience["next_state"] for experience in experiences])
        dones = np.array([experience["done"] for experience in experiences])

        return states, goals, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        import time

        buffer = ReplayBuffer(100)

        for i in range(500):
            sample = [i] * 5
            buffer.add(sample)

        for _ in range(1):
            samples = buffer.sample(1000)

            states, actions, rewards, next_states, terminals = samples

            plt.figure()
            plt.hist(states)

        sample_times = []

        for exp in range(5):
            st = []
            for _ in range(1000):
                t0 = time.time()
                samples = buffer.sample(10 ** exp)
                st.append(time.time() - t0)
            sample_times.append(st)

        plt.figure()
        plt.boxplot(sample_times)

        plt.show()
