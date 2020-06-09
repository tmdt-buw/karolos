
#-------------------- MEMORY --------------------------
import random

from agents.utils.SumTree import SumTree


class Memory:   # stored as ( s, a, r, s_, t ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        error = abs(error)
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def add_all(self, errors, s, a, r, s_, t):
      for i in range(len(s)):
        self.add(errors[i], (s[i], a[i], r[i], s_[i], t[i]))

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


if __name__ == '__main__':

    mem = Memory(100)

    for i in range(100):
        mem.add(1/(i+.1), i)

    result = mem.sample(10000)

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
        mem.update(xx, 1/xx)

    vals, counts = zip(*Counter(data).items())

    plt.figure()
    plt.scatter(vals, counts)

    plt.show()