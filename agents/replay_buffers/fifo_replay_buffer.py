import random
from collections import deque

import numpy as np
from . import ReplayBuffer


class FifoReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=None):
        super(FifoReplayBuffer, self).__init__(buffer_size=buffer_size)

        self.memory = deque(maxlen=self.buffer_size)

    def add(self, experience, priority=None):
        """Add a new experience to memory."""

        experience = [experience[key] for key in self.experience_keys]

        self.memory.append(experience)

    def sample(self, n_samples):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.choices(self.memory, k=n_samples)

        experiences = \
            map(np.stack, zip(*experiences))

        return experiences, None

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    buffer = FifoReplayBuffer(100)

    for i in range(500):
        sample = {key: i for key in ["state", "goal", "action", "reward",
                                     "next_state", "done"]}
        buffer.add(sample)

    samples, _ = buffer.sample(1000)
    states, goals, actions, rewards, next_states, dones = samples

    plt.figure()
    plt.hist(states, bins=100)

    sample_times = []

    for exp in range(4):
        st = []
        for _ in range(1000):
            t0 = time.time()
            samples = buffer.sample(10 ** exp)
            st.append(time.time() - t0)
        sample_times.append(st)

    plt.figure()
    plt.boxplot(sample_times)

    plt.show()
