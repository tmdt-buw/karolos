def get_replay_buffer(config):

    buffer_name = config.pop("name")

    if buffer_name == "fifo":
        from .fifo_replay_buffer import FifoReplayBuffer
        buffer = FifoReplayBuffer(**config)
    elif buffer_name == "priority":
        from .prioritized_replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(**config)
    elif buffer_name == "OnPolBuffer":
        from .OnPolBuffer import OnPolBuffer
        buffer = OnPolBuffer(**config)
    else:
        raise NotImplementedError(f"Unknown replay buffer {buffer_name}")

    return buffer


class ReplayBuffer:
    experience_keys = ["state", "action", "reward", "next_state", "done"]

    def __init__(self, buffer_size=None, uses_priority=False):
        if buffer_size is None:
            buffer_size = int(1e6)

        self.buffer_size = buffer_size
        self.uses_priority = uses_priority

    def add(self, experience, env_id=None, priority=None):
        raise NotImplementedError

    def sample(self, n_samples):
        raise NotImplementedError

    def update(self, indices, error):
        pass
