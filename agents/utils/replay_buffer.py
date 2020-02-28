from collections import  deque, namedtuple
import random
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, terminals = map(np.stack, zip(*experiences))

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
