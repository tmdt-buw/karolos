from pathlib import Path
import random
import sys
from collections import deque

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from replay_buffer import ReplayBuffer


class ReplayBufferUniform(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=None):
        super(ReplayBufferUniform, self).__init__(buffer_size=buffer_size)

        self.memory = deque(maxlen=self.buffer_size)

    def add(self, experience, error=None):
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
