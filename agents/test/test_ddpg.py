import torch

from agents.ddpg import Policy, Critic


def test_policy():
    policy_structure = [('linear', 64), ('relu', None), ('dropout', 0.2),
                        ('linear', 32)]

    policy = Policy([[100]], [10], policy_structure)
    dummy = policy(torch.zeros((1, 100)))
    assert dummy.shape == (1, 10)

    dummy = policy(torch.zeros((100, 100)))
    assert dummy.shape == (100, 10)

    # test multiple state components
    policy = Policy([[100], [50]], [10], policy_structure)
    dummy = policy(torch.zeros((1, 100)), torch.zeros((1, 50)))
    assert dummy.shape == (1, 10)

    dummy = policy(torch.zeros((100, 100)), torch.zeros((100, 50)))
    assert dummy.shape == (100, 10)


def test_critic():
    critic_structure = [('linear', 64), ('relu', None), ('dropout', 0.2),
                        ('linear', 32)]

    critic = Critic([[100]], [10], critic_structure)
    dummy = critic(torch.zeros((1, 100)), torch.zeros((1, 10)))
    assert dummy.shape == (1, 1)

    dummy = critic(torch.zeros((100, 100)), torch.zeros((100, 10)))
    assert dummy.shape == (100, 1)

    # test multiple state components
    critic = Critic([[100], [50]], [10], critic_structure)
    dummy = critic(torch.zeros((1, 100)), torch.zeros((1, 50)),
                   torch.zeros((1, 10)))
    assert dummy.shape == (1, 1)

    dummy = critic(torch.zeros((100, 100)), torch.zeros((100, 50)),
                   torch.zeros((100, 10)))
    assert dummy.shape == (100, 1)