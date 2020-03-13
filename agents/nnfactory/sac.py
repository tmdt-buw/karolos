import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np


class ValueNet(nn.Module):
    # not used
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # initialize,
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim,
                 num_layers_linear_hidden, init_w=3e-3):
        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        super(SoftQNetwork, self).__init__()

        layers = [nn.Linear(in_dim + action_dim, hidden_dim)]

        for l in range(num_layers_linear_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))

        # init
        layers[-1].weight.data.uniform_(-init_w, init_w)
        layers[-1].bias.data.uniform_(-init_w, init_w)

        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        out = self.net(x)

        return out


class PolicyNet(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim, device, num_layers_linear_hidden,
             log_std_min=-20, log_std_max=2, init_w=3e-3):

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        super(PolicyNet, self).__init__()
        # device is initialized by agents Class

        self.device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = [nn.Linear(in_dim, hidden_dim)]

        for l in range(num_layers_linear_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = self.net(state)

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
    use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)
    t = SoftQNetwork([4], [3], 512, 3).to(device)
    print(t.forward(torch.FloatTensor([[1,1,1,1]]).to(device),
                    torch.FloatTensor([[1,1,1]]).to(device)))
    t2 = PolicyNet([4], [3], 512, device, 3)
    print(t2.forward(torch.FloatTensor([1,1,1,1]).to(device)))