import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np


# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # initialize,
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)
        # print(self)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim, init_w=3e-3):
        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        super(SoftQNetwork, self).__init__()
        self.l1 = nn.Linear(in_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # init
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)
        # print(self)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim, device, log_std_min=-20,
                 log_std_max=2, init_w=3e-3):
        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        super(PolicyNet, self).__init__()
        # device is initialized by agents Class

        self.device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        # init
        # print(self)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, eps=1e-06):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = Normal(mean, std).log_prob(
            mean + std * z.to(self.device)) - torch.log(
            1. - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        mean, log_std = self.forward(state)

        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        # action [-1,1]
        # when testing, we can only use the mean (see spinningup doc)
        action = torch.tanh(mean + std * z)
        action = action.detach().cpu().numpy()[0]
        return action


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    pol = PolicyNet(4, 2, device, 32)
    state = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    for i in range(20):
        a = pol.get_action(state)
        print(a)
