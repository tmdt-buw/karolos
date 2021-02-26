import numpy as np
from . import ReplayBuffer

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, sample, priority):
        idx = self.write + self.capacity - 1

        self.data[self.write] = sample
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, index, priority):
        change = priority - self.tree[index]

        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_size=None, e=0.01, a=0.6):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size=buffer_size,
                                                      uses_priority=True)

        self.tree = SumTree(self.buffer_size)

        self.e = e
        self.a = a

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, experience, priority=100):
        experience = [experience[key] for key in self.experience_keys]

        p = self._getPriority(priority)
        self.tree.add(experience, p)

    def sample(self, n_samples):
        segment = self.tree.total() / n_samples

        s = np.random.uniform(np.arange(n_samples), np.arange(n_samples) + 1)
        s *= segment

        indices = []
        experiences = []

        for s_ in s:
            index, _, experience = self.tree.get(s_)

            indices.append(index)
            experiences.append(experience)

        indices = np.stack(indices)
        experiences = map(np.stack, zip(*experiences))

        return experiences, indices

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    buffer = PrioritizedReplayBuffer(1000)

    for i in range(500):
        sample = {key: i for key in ["state", "goal", "action", "reward",
                                     "next_state", "done"]}
        buffer.add(sample, 100)

    samples, _ = buffer.sample(1000)
    states, goals, actions, rewards, next_states, dones = samples

    plt.figure()
    plt.hist(states, bins=100)

    sample_times = []

    for exp in range(4):
        st = []
        for _ in range(1000):
            t0 = time.time()
            indices, samples = buffer.sample(10 ** exp)
            st.append(time.time() - t0)
        sample_times.append(st)

    plt.figure()
    plt.boxplot(sample_times)

    plt.show()
