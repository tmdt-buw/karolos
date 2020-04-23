import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

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
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv2d:
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
            nn.ReLU()
        ])

        for l in range(num_layers_linear_hidden - 1):
            self.operators.append(nn.Linear(hidden_dim, hidden_dim))
            self.operators.append(nn.ReLU())

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
        self.hidden_dim = np.product(hidden_dim)
        self.action_dim = np.product(action_dim)

        # device is initialized by agents Class
        self.operators = nn.ModuleList([
            Flatten(),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU()
        ])

        for l in range(num_layers_linear_hidden - 1):
            self.operators.append(nn.Linear(hidden_dim, hidden_dim))
            self.operators.append(nn.ReLU())
            # self.operators.append(nn.Dropout(0.2))

        self.operators.append(nn.Linear(self.hidden_dim, 2 * self.action_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, state, deterministic=True):
        x = state
        for operator in self.operators:
            x = operator(x)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        if deterministic:
            action = mean.tanh()
            log_prob = torch.zeros_like(log_std)
        else:
            # todo: is clamp really necessary?
            log_std = self.std_clamp(log_std)
            std = log_std.exp()
            covariance = torch.diag_embed(std)
            m = MultivariateNormal(mean, covariance)
            action_base = m.sample()
            log_prob = m.log_prob(action_base)
            log_prob.unsqueeze_(-1)

            action = action_base.tanh()

            # According to "Soft Actor-Critic" (Haarnoja et. al) Appendix C
            action_bound_compensation = torch.log(1. - action.tanh().pow(2) + 1e-6)
            action_bound_compensation = action_bound_compensation.sum(dim=-1, keepdim=True)
            log_prob.sub_(action_bound_compensation)

        return action, log_prob


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
