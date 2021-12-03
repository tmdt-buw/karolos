from collections import OrderedDict
import torch


class OnPolBuffer(object):

    def __init__(self, capacity: int, state_shape, action_shape, device,
                 # $experience_keys for different on_pol_algos
                 gae: bool = True, mc: bool = False,
                 gamma_gae: float = 0.99, lambda_gae: float = 0.95):

        self.device = device
        self.capacity = capacity

        self.env_buffer = OrderedDict({
            'states': torch.zeros((capacity, state_shape),
                                  dtype=torch.float32, device=self.device),
            'actions': torch.zeros((capacity, action_shape),
                                   dtype=torch.float32, device=self.device),
            'rewards': torch.zeros(capacity,
                                   dtype=torch.float32, device=self.device),
            'terminals': torch.zeros(capacity,
                                     dtype=torch.bool, device=self.device),
            'ac_log_probs': torch.zeros(capacity,
                                        dtype=torch.float32, device=self.device)})

        self._idx = 0

        self.mc = mc
        self.gae = gae

        if self.mc:
            assert not gae, 'Either gae or mc have to be true'
            self._adv_func = self._mc
            raise NotImplementedError
        elif self.gae:
            assert not mc, 'Either gae or mc have to be true'
            self._adv_func = self._gae
            self.env_buffer['values'] = torch.zeros(capacity,
                                                    dtype=torch.float32, device=self.device)
            self.gamma = gamma_gae
            self.lam = lambda_gae
        else:
            raise NotImplementedError('Either gae or mc have to be true')

    def is_full(self):
        return self._idx >= self.capacity

    def add(self, experience: dict):
        assert experience.keys() == self.env_buffer.keys()

        if self._idx >= self.capacity:
            raise IndexError

        for k, v in experience.items():
            self.env_buffer[k][self._idx] = v

        self._idx += 1

    def clear(self):
        for k, v in self.env_buffer.items():
            self.env_buffer[k] = torch.zeros_like(self.env_buffer[k], device=self.device)
        self._idx = 0

    def sample(self, policy=None):
        # sample in order
        adv = self._adv_func(policy)
        adv = adv.detach()
        samples = []
        samples.append(self.env_buffer["states"])
        samples.append(self.env_buffer["actions"])
        samples.append(self.env_buffer["ac_log_probs"])
        samples.append(self.env_buffer["values"])
        samples.append(adv)
        return samples

    def _gae(self, critic):
        assert critic is not None
        adv = torch.zeros_like(self.env_buffer['rewards'],
                               dtype=torch.float32, device=self.device)

        # this is gae version if not-complete episodes in memory
        last_adv = 0
        last_val = critic(self.env_buffer["states"][-1].unsqueeze(0)).squeeze()
        # last_val = last_val.cpu().data.numpy()

        for tstep in reversed(range(self.capacity)):
            mask = ~ self.env_buffer['terminals'][tstep]
            last_val = last_val * mask
            last_adv = last_adv * mask
            delta = self.env_buffer['rewards'][tstep] + self.gamma * last_val - \
                    self.env_buffer['values'][tstep]
            last_adv = delta + self.gamma * self.lam * last_adv
            adv[tstep] = last_adv
            last_val = self.env_buffer['values'][tstep]
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