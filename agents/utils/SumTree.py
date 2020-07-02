import numpy


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

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

        return (idx, self.tree[idx], self.data[dataIdx])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from collections import Counter

    import numpy as np

    tree = SumTree(1000)

    for i in range(10000):
        tree.add(1 / ((i % 1000) + 100), i)

    n = 10000

    segment = tree.total() / n

    result = []
    idxs = set()

    for i in range(n):
        a = segment * i
        b = segment * (i + 1)

        s = a
        (idx, p, data) = tree.get(s)

        result.append(data)
        idxs.add(idx)

        print(data)

    vals, counts = zip(*Counter(result).items())

    plt.scatter(vals, counts)

    for xx in sorted(list(idxs)):
        tree.update(xx, 1)

    vals, counts = zip(*Counter(result).items())

    plt.figure()
    plt.scatter(vals, counts)

    plt.show()
