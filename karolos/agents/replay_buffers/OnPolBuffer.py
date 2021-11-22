from collections import OrderedDict
import torch


class OnPolBuffer(object):

    def __init__(self, capacity_per_env: int, state_shape, action_shape, device,
                 # $experience_keys for different on_pol_algos
                 number_envs: int, gae: bool = True, mc: bool = False,
                 gamma_gae: float = 0.99, lambda_gae: float = 0.95):

        self.device = device
        self.cap_p_env = capacity_per_env

        self.env_buffer = OrderedDict({
            'states': torch.zeros((number_envs, self.cap_p_env, *state_shape.shape),
                                  dtype=torch.float32, device=self.device),
            'actions': torch.zeros((number_envs, self.cap_p_env, *action_shape.shape),
                                                  dtype=torch.float32, device=self.device),
            'rewards': torch.zeros((number_envs, self.cap_p_env),
                                 dtype=torch.float32, device=self.device),
            'terminals': torch.zeros((number_envs, self.cap_p_env),
                                   dtype=torch.bool, device=self.device),
            'ac_log_probs': torch.zeros((number_envs, self.cap_p_env),
                                  dtype=torch.float32, device=self.device)})

        self.no_envs = number_envs
        self._idx = {e: 0 for e in range(self.no_envs)}
        self.fulls = []

        self.mc = mc
        self.gae = gae

        if self.mc:
            assert not gae, 'Either gae or mc have to be true'
            self._adv_func = self._mc
        elif self.gae:
            assert not mc, 'Either gae or mc have to be true'
            self._adv_func = self._gae
            self.env_buffer['values'] = torch.zeros((number_envs, self.cap_p_env),
                                                    dtype=torch.float32, device=self.device)
            self.gamma = gamma_gae
            self.lam = lambda_gae
        else:
            raise NotImplementedError('Either gae or mc have to be true')

    def add(self, env_id: int, experience: dict):
        assert experience.keys() == self.env_buffer.keys()
        if self._idx[env_id] >= self.cap_p_env:
            raise IndexError

        for k, v in experience.items():
            self.env_buffer[k][env_id, self._idx[env_id]] = v

        self._idx[env_id] += 1
        if self._idx[env_id] == (self.cap_p_env - 1):
            self.fulls.append(env_id)

    def clear(self):
        for k, v in self.env_buffer.items():
            self.env_buffer[k] = torch.zeros_like(self.env_buffer[k], device=self.device)
        self.fulls = []
        self._idx = {e: 0 for e in range(self.no_envs)}

    def sample(self, policy=None):
        # sample in order
        adv = self._adv_func(policy)
        adv = adv.detach()
        samples = []
        for k, v in self.env_buffer.items():
            if k in ['rewards', 'terminals']:
                continue
            samples.append(
                self.env_buffer[k].reshape(self.env_buffer[k].shape[0] * self.env_buffer[k].shape[1],
                                           *list(self.env_buffer[k].shape[2:]))
            )
        samples.append(
            adv.reshape(adv.shape[0] * adv.shape[1], *list(adv.shape[2:]))
        )
        return samples

    def full_ids(self):
        # if env_id in mem.is_full(): continue
        return self.fulls

    def is_full(self):
        return len(self.fulls) == self.no_envs

    def _gae(self, critic):
        assert critic is not None
        adv = torch.zeros_like(self.env_buffer['rewards'],
                              dtype=torch.float32, device=self.device)

        # this is gae version if not-complete episodes in memory
        last_adv = 0
        last_val = critic(torch.stack([e[-1] for e in self.env_buffer['states']])).squeeze()
        # last_val = last_val.cpu().data.numpy()

        for tstep in reversed(range(self.cap_p_env)):
            mask = ~ self.env_buffer['terminals'][:, tstep]
            last_val = last_val * mask
            last_adv = last_adv * mask
            delta = self.env_buffer['rewards'][:, tstep] + self.gamma * last_val - \
                    self.env_buffer['values'][:, tstep]
            last_adv = delta + self.gamma * self.lam * last_adv
            adv[:, tstep] = last_adv
            last_val = self.env_buffer['values'][:, tstep]
        return adv

        # this would be the version if completed episodes in memory (last state terminated)
        # advantage = 0
        # masks = 1.0 - self.env_buffer['terminals']
        # for tstep in reversed(range(self.no_envs + 1)):
        #     delta = self.env_buffer["rewards"][:, tstep] + self.gamma * \
        #             self.env_buffer['values'][:, tstep] * masks[:, tstep] - \
        #             self.env_buffer['values'][:, tstep]
        #     advantage = delta + self.gamma * self.lam * masks[:, tstep] * advantage
        #     adv[:, tstep] = advantage
        # return adv

    def _mc(self, model_not_used=None):
        raise NotImplementedError