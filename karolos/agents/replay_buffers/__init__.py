def get_replay_buffer(config):

    replay_buffer_name = config.pop("name")

    if replay_buffer_name == "uniform":
        from .replay_buffer_uniform import ReplayBufferUniform
        buffer = ReplayBufferUniform(**config)
    elif replay_buffer_name == "priority":
        from .replay_buffer_priority import ReplayBufferPriority
        buffer = ReplayBufferPriority(**config)
    elif replay_buffer_name == "OnPolBuffer":
        from .OnPolBuffer_WIP import OnPolBuffer
        import warnings
        warnings.warn("OnPolBuffer is WIP")
        buffer = OnPolBuffer(**config)
    else:
        raise NotImplementedError(f"Unknown replay buffer {replay_buffer_name}")

    return buffer
