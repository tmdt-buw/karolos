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
            action = mean.tanh()
            log_prob = torch.zeros_like(log_std)
        else:
            log_std = self.std_clamp(log_std)
            std = log_std.exp()

            # Draw from Normal so that log_prob is normalized
            normal = MultivariateNormal(torch.zeros_like(mean),
                                        torch.diag_embed(torch.ones_like(std)))
            z = normal.sample()

            log_prob = normal.log_prob(z)
            log_prob.unsqueeze_(-1)

            action_base = mean + std * z
            action = torch.tanh(action_base)

            action_bound_compensation = torch.log(
                1. - action.pow(2) + np.finfo(float).eps).sum(dim=1,
                                                              keepdim=True)

            log_prob.sub_(action_bound_compensation)

        return action, log_prob

    def get_activations(self, state):
        x = state

        activations = []

        for operator in self.operators:
            x = operator(x)

            if type(operator) == nn.ReLU:
                activations.append(x)

        return activations


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # print(device)
    # critic = Critic([4], [3], 512, 3).to(device)
    # print(critic(torch.FloatTensor([[1,1,1,1]]).to(device),
    #                 torch.FloatTensor([[1,1,1]]).to(device)))
    policy = Policy([4], [2], 32, 8).to(device)
    # print(policy(torch.FloatTensor([[1, 1, 1, 1]]).to(device)))
    # print(policy.operators)

    actions = []
    log_probs = []

    for ii in range(1_000):
        action, log_prob = policy(torch.FloatTensor([[1, 1, 1, 1]]).to(device),
                                  False)

        # print(action.shape, log_prob)
        actions.append(action[0].cpu().detach().numpy())
        log_probs.append(log_prob[0].exp().cpu().detach().numpy())

    actions = np.array(actions)
    log_probs = np.array(log_probs)

    plt.figure()
    plt.hist(actions[:, 0], alpha=0.5)

    plt.figure()
    plt.hist(log_probs[:, 0], alpha=0.5)

    xy = np.vstack([actions[:, 0], actions[:, 1]])
    z = gaussian_kde(xy)(xy)

    plt.figure()
    plt.scatter(actions[:, 0], actions[:, 1], c=z, s=100, edgecolor=None)
    plt.show()
