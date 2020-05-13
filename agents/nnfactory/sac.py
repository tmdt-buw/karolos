import torch
import torch.nn as nn
import numpy as np


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
    def __init__(self, in_dim, action_dim, value_structure):
        super(Critic, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        # first position should be a layer, linear layer for now

        assert value_structure[0][1] is not None
        prev_object = value_structure.pop(0)

        self.operators = nn.ModuleList([
            Flatten(),
            nn.Linear(in_dim + action_dim, prev_object[1]),
        ])

        for layer, argument in value_structure[:-1]:
            if layer == 'linear':
                self.operators.append(nn.Linear(prev_object[1], argument))
                prev_object = (layer, argument)
            elif layer == 'relu':
                assert argument is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == 'dropout':
                self.operators.append(nn.Dropout(argument))
            else:
                raise NotImplementedError(f'{layer} not known')

        self.operators.append(nn.Linear(prev_object[1], 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for operator in self.operators:
            x = operator(x)
        return x


class Policy(nn.Module):
    def __init__(self, in_dim, action_dim, policy_structure, log_std_min=-20,
                 log_std_max=2):
        super(Policy, self).__init__()

        assert len(in_dim) == 1
        assert len(action_dim) == 1

        in_dim = np.product(in_dim)
        action_dim = np.product(action_dim)

        # first position should be a layer, linear layer for now
        assert policy_structure[0][1] is not None
        prev_object = policy_structure.pop(0)

        self.operators = nn.ModuleList([
            Flatten(),
            nn.Linear(in_dim, prev_object[1]),
        ])

        for layer, argument in policy_structure[:-1]:
            if layer == 'linear':
                self.operators.append(nn.Linear(prev_object[1], argument))
                prev_object = (layer, argument)
            elif layer == 'relu':
                assert argument is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == 'dropout':
                self.operators.append(nn.Dropout(argument))
            else:
                raise NotImplementedError(f'{layer} not known')

        self.operators.append(nn.Linear(prev_object[1], 2 * action_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, state):
        x = state
        for operator in self.operators:
            x = operator(x)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        log_std = self.std_clamp(log_std)

        return mean, log_std


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    pol_struct = [('linear', 64), ('relu', None), ('dropout', 0.2), ('linear', 32)]

    policy = Policy([21], [7], pol_struct).to(device)

    print(policy.operators)

    val_struct = [('linear', 32), ('relu', None), ('dropout', 0.2), ('linear', 32)]

    val = Critic([21], [7], val_struct).to(device)

    print(val.operators)