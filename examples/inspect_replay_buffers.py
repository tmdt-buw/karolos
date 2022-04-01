# todo derive suitable tests
import time

import matplotlib.pyplot as plt

from karolos.agents.replay_buffers import get_replay_buffer

replay_buffers = ["uniform", "priority"]

for replay_buffer_name in replay_buffers:
    print(replay_buffer_name)

    replay_buffer = get_replay_buffer({
        "name": replay_buffer_name,
        "buffer_size": 50
    })

    replay_buffer.experience_keys = ["state", "action", "reward", "next_state", "done"]

    for i in range(150):
        sample = {key: i for key in ["state", "action", "reward", "next_state", "done"]}
        replay_buffer.add(sample, i)

    # test general functionality / sample distribution
    samples, indices = replay_buffer.sample(10_000)
    states, actions, rewards, next_states, dones = samples

    plt.figure()
    plt.hist(states, bins=100)
    plt.xlabel("sample")
    plt.ylabel("amount sampled")
    plt.title(replay_buffer_name)
    plt.show()

    # test inversion of priorities
    if replay_buffer.uses_priority:
        errors = 1 / states

        for idx, error in zip(indices, errors):
            replay_buffer.update(idx, error)

        samples, indices = replay_buffer.sample(10_000)
        states, actions, rewards, next_states, dones = samples

        plt.figure()
        plt.hist(states, bins=100)
        plt.xlabel("sample")
        plt.ylabel("amount sampled")
        plt.title(f"{replay_buffer_name} | priorities inverted")
        plt.show()

    # test sampling behavior with bigger sample sizes
    sample_times = []

    for exp in range(4):
        number_samples = 10 ** exp
        st = []
        for _ in range(1000):
            t0 = time.time()
            samples, indices = replay_buffer.sample(number_samples)

            if replay_buffer.uses_priority:
                states, actions, rewards, next_states, dones = samples
                errors = states

                for idx, error in zip(indices, states):
                    replay_buffer.update(idx, error)
            st.append((time.time() - t0) / number_samples)
        sample_times.append(st)

    plt.figure()
    plt.boxplot(sample_times)
    plt.xlabel("number of samples [10^x]")
    plt.ylabel("sample time per sample")
    plt.title(replay_buffer_name)
    plt.show()

    # test clearing buffer

    replay_buffer.clear()
    assert len(replay_buffer) == 0