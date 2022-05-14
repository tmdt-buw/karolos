import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Clamp(torch.nn.Module):

    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


def init_xavier_uniform(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)


class NeuralNetwork(nn.Module):
    def __init__(self, in_dims, network_structure):
        super(NeuralNetwork, self).__init__()

        if type(in_dims) is int:
            in_dim = in_dims
        else:
            in_dim = int(np.sum([np.product(dim) for dim in in_dims]))

        assert type(in_dim) == int

        self.operators = nn.ModuleList([
            nn.Flatten()
        ])

        current_layer_size = in_dim

        for layer, params in network_structure:
            if layer == 'linear':
                self.operators.append(nn.Linear(current_layer_size, params))
                current_layer_size = params
            elif layer == 'relu':
                assert params is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == "leaky_relu":
                assert params is None, 'No argument for ReLU please'
                self.operators.append(nn.LeakyReLU())
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
            elif layer == 'batchnorm':
                self.operators.append(nn.BatchNorm1d(current_layer_size))
            else:
                raise NotImplementedError(f'{layer} not known')

    def forward(self, *args, **kwargs):
        x = torch.cat(args, dim=-1)

        for operator in self.operators:
            x = operator(x)
        return x

    def get_weights(self):

        weights = []

        for operator in self.operators:
            if type(operator) == nn.Linear:
                weights.append(operator.weight)

        return weights

    def get_activations(self, x):
        activations = []

        for operator in self.operators:
            x = operator(x)

            if type(operator) == nn.ReLU:
                activations.append(x)

        return activations


class Actor(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure, log_std_min=-20, log_std_max=2):
        in_dim = int(np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim)) * 2

        super(Actor, self).__init__(in_dim, network_structure)

        dummy = super(Actor, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, *state_args, deterministic=True):
        x = super(Actor, self).forward(*state_args)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        log_std = self.std_clamp(log_std)
        std = log_std.exp()

        normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))

        if deterministic:
            action_base = mean
        else:
            action_base = normal.rsample()

        log_prob = normal.log_prob(action_base)
        log_prob.unsqueeze_(-1)

        action = torch.tanh(action_base)

        action_bound_compensation = (
                2 * (np.log(2) - action_base - torch.nn.functional.softplus(-2 * action_base))).sum(dim=1,
                                                                                                    keepdim=True)

        log_prob.sub_(action_bound_compensation)

        return action, log_prob

    def evaluate(self, *state_args, action):
        x = super(Actor, self).forward(*state_args)
        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        log_std = self.std_clamp(log_std)
        std = log_std.exp()

        normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))

        action_base = torch.atanh(action)

        log_prob = normal.log_prob(action_base)
        log_prob.unsqueeze_(-1)

        action_bound_compensation = (2 * (np.log(2) - action_base - torch.nn.functional.softplus(-2 * action_base))) \
            .sum(dim=1, keepdim=True)
        log_prob.sub_(action_bound_compensation)

        entropy = normal.entropy()

        return log_prob, entropy


class Critic(NeuralNetwork):
    def __init__(self, in_dims, network_structure):
        in_dim = int(np.sum([np.product(arg) for arg in in_dims]))

        super(Critic, self).__init__(in_dim, network_structure)

        dummy = super(Critic, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *args):
        return super(Critic, self).forward(*args)
