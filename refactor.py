import torch
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1e-06


def old_code(mean, std):
    normal = Normal(0, 1)
    z = normal.sample()

    action = torch.tanh(mean + std * z)

    # print(torch.log(1. - action.pow(2) + 1e-6).sum(dim=1, keepdim=True))

    action_bound_compensation = torch.log(1. - action.pow(2) + np.finfo(float).eps).sum(dim=1, keepdim=True)

    log_prob = Normal(mean, std).log_prob(mean + std * z)
    log_prob -= action_bound_compensation

    # print(action.shape, log_prob.shape)

    return action, log_prob.exp()


def new_code(mean, std):
    covariance = torch.diag_embed(std.pow(2))
    m = MultivariateNormal(mean, covariance)
    action_base = m.sample()
    action = action_base.tanh()

    log_prob = m.log_prob(action_base)
    log_prob.unsqueeze_(-1)

    # According to "Soft Actor-Critic" (Haarnoja et. al) Appendix C
    action_bound_compensation = torch.log(1. - action.pow(2) + np.finfo(float).eps)
    action_bound_compensation = action_bound_compensation.sum(dim=-1,
                                                              keepdim=True)
    # log_prob = log_prob - action_bound_compensation

    return action, torch.zeros_like(log_prob.exp())


old_actions = []
new_actions = []
old_log_prob = []
new_log_prob = []

mean = torch.Tensor([[0, 0], [1, 1], [2, 2]])
std = torch.diag_embed(torch.Tensor([[1., 1.], [.1, .1], [10, 10]]))

#print(mean.shape, std.shape)

m = MultivariateNormal(mean, std)
action_base = m.sample()
# action_base = action_base.reshape(mean.shape)
log_prob = m.log_prob(action_base)
action = action_base.tanh()

# According to "Soft Actor-Critic" (Haarnoja et. al) Appendix C
action_bound_compensation = torch.log(1. - action.tanh().pow(2) + 1e-6)
#print(action_bound_compensation.shape)
action_bound_compensation = action_bound_compensation.sum(dim=-1)
#print(action_bound_compensation.shape)
log_prob -= action_bound_compensation

mean = torch.Tensor([[0, 0], [1, 1], [2, 2]])
std = torch.Tensor([[.1, .0001], [.1, .1], [.01, .01]])

#mean = torch.Tensor([[0, 0]])
#std = torch.Tensor([[.1]])

print(mean.shape, std.shape)

for ii in range(1_000):
    action, log_prob = old_code(mean, std)
    # print(action)
    old_actions.append(action[0].numpy())
    old_log_prob.append(log_prob[0].numpy())

    action, log_prob = new_code(mean, std)
    # print(action)
    # print(log_prob)
    new_actions.append(action[0].numpy())
    new_log_prob.append(log_prob[0].numpy())

old_actions = np.array(old_actions)
new_actions = np.array(new_actions)
old_log_prob = np.array(old_log_prob)
new_log_prob = np.array(new_log_prob)
# print(len(old_actions))
# print(len(new_actions))
#
plt.figure()
plt.hist(old_actions[:,0], alpha=0.5)  # , range=[-1,1])
plt.hist(new_actions[:,0], alpha=0.5)  # , range=[-1,1])

plt.figure()
plt.hist(old_log_prob[:,0], alpha=0.5)  # , range=[-1,1])
plt.hist(new_log_prob[:,0], alpha=0.5)  # , range=[-1,1])


from scipy.stats import gaussian_kde
xy = np.vstack([new_actions[:,0],new_actions[:,1]])
z = gaussian_kde(xy)(xy)


plt.figure()
plt.scatter(old_actions[:,0],old_actions[:,1])
plt.figure()
plt.scatter(new_actions[:,0],new_actions[:,1], c=z, s=100, edgecolor='')
plt.show()
