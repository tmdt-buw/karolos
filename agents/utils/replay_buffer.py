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

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, n_samples):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.choices(self.memory, k=n_samples)

        states, actions, rewards, next_states, terminals = map(np.stack, zip(
            *experiences))

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
