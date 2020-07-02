from collections import deque, namedtuple
import numpy as np
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])

    def add(self, experience):
        """Add a new experience to memory."""
        e = self.experience(*experience)
        self.memory.append(e)

    def sample(self, n_samples):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.choices(self.memory, k=n_samples)

        experiences = map(np.stack, zip(*experiences))

        return experiences

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