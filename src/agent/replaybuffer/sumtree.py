#my_project\src\agent\replaybuffer\sumtree.py
import numpy as np

class SumTree:
    """
    Binary SumTree for Prioritized Experience Replay
    -----------------------------------------------
    - Fast O(log N) add / update / sample
    - Array-based (cache friendly)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1

        self.tree = np.zeros(self.tree_size, dtype=np.float32)
        self.data_pointer = 0
        self.size = 0

    # --------------------------------------------------
    @property
    def total_priority(self):
        return self.tree[0]

    # --------------------------------------------------
    def add(self, priority: float):
        """
        Add priority at current data pointer
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return tree_idx

    # --------------------------------------------------
    def update(self, tree_idx: int, priority: float):
        """
        Update priority and propagate the change
        """
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # propagate change
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    # --------------------------------------------------
    def get_leaf(self, value: float):
        """
        Traverse tree to find leaf for cumulative sum = value
        """
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= self.tree_size:
                leaf_idx = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], data_idx
