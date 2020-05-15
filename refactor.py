import torch
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1e-06

old_actions = []
actions = []
old_log_prob = []
log_probs = []

mean = torch.Tensor([[0, 0], [1, 1], [2, 2]])
std = torch.Tensor([[.1, 1e-7], [.1, .1], [.01, .01]])

for ii in range(1_000):
    normal = MultivariateNormal(torch.zeros_like(mean),
                                torch.diag_embed(torch.ones_like(std)))
    z = normal.sample()

    log_prob = normal.log_prob(z)
    log_prob.unsqueeze_(-1)

    action_base = mean + std * z
    action = torch.tanh(action_base)

    action_bound_compensation = torch.log(
        1. - action.pow(2) + np.finfo(float).eps).sum(dim=1, keepdim=True)

    log_prob.sub_(action_bound_compensation)
    # log_prob = log_prob.exp().sqrt()

    actions.append(action[0].numpy())
    log_probs.append(log_prob[0].numpy())

actions = np.array(actions)
log_probs = np.array(log_probs)

plt.figure()
plt.hist(actions[:, 0], alpha=0.5)  # , range=[-1,1])

plt.figure()
plt.hist(log_probs[:, 0], alpha=0.5)  # , range=[-1,1])

from scipy.stats import gaussian_kde

plt.figure()
# xy = np.vstack([actions[:, 0], actions[:, 1]])
# z = gaussian_kde(xy)(xy)
sc = plt.scatter(actions[:, 0], actions[:, 1], c=log_probs, s=100, edgecolor='')
plt.colorbar(sc)
plt.show()
