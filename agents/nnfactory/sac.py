import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, in_dim, action_dim, network_structure):
        super(Critic, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        self.operators = nn.ModuleList([
            Flatten()
        ])

        current_layer_size = in_dim + action_dim

        for layer, params in network_structure:
            if layer == 'linear':
                self.operators.append(nn.Linear(current_layer_size, params))
                current_layer_size = params
            elif layer == 'relu':
                assert params is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == 'selu':
                assert params is None, 'No argument for SeLU please'
                self.operators.append(nn.SELU())
            elif layer == 'tanh':
                assert params is None, 'No argument for Tanh please'
                self.operators.append(nn.Tanh())
            elif layer == 'gelu':
                assert params is None, 'No argument for GreLU please'
                self.operators.append(nn.GELU())
            elif layer == 'dropout':
                self.operators.append(nn.Dropout(params))
            else:
                raise NotImplementedError(f'{layer} not known')

        self.operators.append(nn.Linear(current_layer_size, 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for operator in self.operators:
            x = operator(x)
        return x


class Policy(nn.Module):
    def __init__(self, in_dim, action_dim, network_structure, log_std_min=-20,
                 log_std_max=2):
        super(Policy, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        self.operators = nn.ModuleList([
            Flatten()
        ])

        current_layer_size = in_dim

        for layer, params in network_structure:
            if layer == 'linear':
                self.operators.append(nn.Linear(current_layer_size, params))
                current_layer_size = params
            elif layer == 'relu':
                assert params is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == 'selu':
                assert params is None, 'No argument for SeLU please'
                self.operators.append(nn.SELU())
            elif layer == 'tanh':
                assert params is None, 'No argument for Tanh please'
                self.operators.append(nn.Tanh())
            elif layer == 'gelu':
                assert params is None, 'No argument for GreLU please'
                self.operators.append(nn.GELU())
            elif layer == 'dropout':
                self.operators.append(nn.Dropout(params))
            else:
                raise NotImplementedError(f'{layer} not known')

        self.operators.append(nn.Linear(current_layer_size, 2 * action_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, state, deterministic=True):
        x = state
        for operator in self.operators:
            x = operator(x)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        log_std = self.std_clamp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.ones_like(log_std)
        else:
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.rsample()

            action = torch.tanh(z)

            log_prob = normal.log_prob(z)
            log_prob -= torch.log(1. - action.pow(2) + 1e-6)

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

    pol_struct = [('linear', 64), ('relu', None), ('dropout', 0.2),
                  ('linear', 32)]

    policy = Policy([21], [7], pol_struct).to(device)

    print(policy.operators)

    val_struct = [('linear', 32), ('relu', None), ('dropout', 0.2),
                  ('linear', 32)]

    val = Critic([21], [7], val_struct).to(device)

    print(val.operators)
