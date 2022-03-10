class ReplayBuffer:

    def __init__(self, buffer_size=None, uses_priority=False):
        self.experience_keys = ["state", "goal", "action", "reward", "next_state", "next_goal", "done"]

        if buffer_size is None:
            buffer_size = int(1e6)

        self.buffer_size = buffer_size
        self.uses_priority = uses_priority

    def add(self, experience, priority=None):
        raise NotImplementedError

    def sample(self, number_samples):
        raise NotImplementedError

    def update(self, indices, error):
        pass
