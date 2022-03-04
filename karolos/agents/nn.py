import numpy as np
import torch
import torch.nn as nn


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


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, network_structure):
        super(NeuralNetwork, self).__init__()

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


class Critic(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(np.sum([np.product(arg) for arg in state_dims]) + np.product(action_dim))

        super(Critic, self).__init__(in_dim, network_structure)

        dummy = super(Critic, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *args):
        return super(Critic, self).forward(*args)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    network_structure = [('linear', 64), ('relu', None), ('dropout', 0.2), ('linear', 32)]

    neural_network = NeuralNetwork(21, network_structure).to(device)

    print(neural_network)

    x = torch.rand((1, 21)).to(device)
    y = neural_network(x)

    print(y.shape)
