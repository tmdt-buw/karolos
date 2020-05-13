import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Clamp(torch.nn.Module):

    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.sigmoid(x) * (self.max - self.min) + self.min


def init_xavier_uniform(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)


class Critic(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim,
                 num_layers_linear_hidden):
        super(Critic, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        self.operators = nn.ModuleList([
            Flatten(),
            nn.Linear(in_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(.1)
        ])

        for l in range(num_layers_linear_hidden - 1):
            self.operators.append(nn.Linear(hidden_dim, hidden_dim))
            self.operators.append(nn.ReLU()),
            self.operators.append(nn.Dropout(.1))

        self.operators.append(nn.Linear(hidden_dim, 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for operator in self.operators:
            x = operator(x)
        return x


class Policy(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim,
                 num_layers_linear_hidden, log_std_min=-20,
                 log_std_max=2):
        super(Policy, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        self.in_dim = np.product(in_dim)
        self.action_dim = np.product(action_dim)
        self.hidden_dim = hidden_dim

        # device is initialized by agents Class
        self.operators = nn.ModuleList([
            Flatten(),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(.1)
        ])

        for l in range(num_layers_linear_hidden - 1):
            self.operators.append(nn.Linear(hidden_dim, hidden_dim))
            self.operators.append(nn.ReLU())
            self.operators.append(nn.Dropout(.1))

        self.operators.append(nn.Linear(self.hidden_dim, 2 * self.action_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, state, deterministic=True):
        x = state
        for operator in self.operators:
            x = operator(x)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.ones_like(log_std)
        else:
            std = log_std.exp()

            normal = Normal(0, 1)
            z = normal.sample()
            action = torch.tanh(mean + std * z)

            action_bound_compensation = torch.log(1. - action.pow(2) + 1e-6)
            log_prob = Normal(mean, std).log_prob(mean + std * z)
            log_prob.sub_(action_bound_compensation)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def get_weights(self):

        weights = []

        for operator in self.operators:
            if type(operator) == nn.Linear:
                weights.append(operator.weight)

        return weights

    def get_activations(self, state):
        x = state

        activations = []

        for operator in self.operators:
            x = operator(x)

            if type(operator) == nn.ReLU:
                activations.append(x)

        return activations


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)
    # critic = Critic([4], [3], 512, 3).to(device)
    # print(critic(torch.FloatTensor([[1,1,1,1]]).to(device),
    #                 torch.FloatTensor([[1,1,1]]).to(device)))
    policy = Policy([4], [3], 512, 3).to(device)
    print(policy(torch.FloatTensor([[1, 1, 1, 1]]).to(device)))
    print(policy.operators)
