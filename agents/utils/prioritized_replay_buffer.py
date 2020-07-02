
#-------------------- MEMORY --------------------------
import random

from agents.utils.SumTree import SumTree
from collections import namedtuple
import numpy as np

class ReplayBuffer:   # stored as ( s, a, r, s_, t ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, buffer_size):
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, experience, priority=100):
        e = self.experience(*experience)
        p = self._getPriority(priority)
        self.tree.add(e, p)

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
        try:
            experiences = map(np.stack, zip(*experiences))
        except TypeError:
            ...
            raise

        return indices, experiences

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    buffer = ReplayBuffer(100)

    for i in range(100):
        sample = [i] * 5
        buffer.add(sample, i ** 2)

    for _ in range(1):
        indices, samples = buffer.sample(1000)

        states, actions, rewards, next_states, terminals = samples

        plt.figure()
        plt.hist(states)

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

    sample_times = []

    for exp in range(5):
        st = []

        indices, samples = buffer.sample(10 ** exp)
        errors = np.ones_like(indices)

        for _ in range(100):
            t0 = time.time()
            indices, samples = buffer.sample(10 ** exp)
            # buffer.update(indices, errors)
            for index, error in zip(indices, errors):
                buffer.update(index, error)
            st.append(time.time() - t0)
        sample_times.append(st)

    plt.figure()
    plt.boxplot(sample_times)

    plt.show()

    exit()

    for i in range(100):
        buffer.add(1 / (i + .1), i)

    result = buffer.sample(10000)

    print(result)

    ixs = [x[0] for x in result]
    data = [x[1] for x in result]

    print(data)

    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    vals, counts = zip(*Counter(data).items())

    plt.scatter(vals, counts)

    for xx in sorted(np.unique(ixs)):
        buffer.update(xx, 1 / xx)

    vals, counts = zip(*Counter(data).items())

    plt.figure()
    plt.scatter(vals, counts)

    plt.show()