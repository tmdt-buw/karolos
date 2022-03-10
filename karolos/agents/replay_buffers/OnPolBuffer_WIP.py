from collections import OrderedDict, defaultdict
from operator import itemgetter
import torch
import time


class OnPolBuffer2:

    def __init__(self, number_envs: int, capacity: int, device,
                 gae: bool = True, mc: bool = False,
                 gamma_gae: float = 0.99, lambda_gae: float = 0.95):

        torch.set_num_threads(3)

        self.number_envs = number_envs
        self.device = device
        self.capacity = capacity  # on-pol batch size

        # self.env_buffer keeps dynamic python list of pytorch tensors
        self.env_buffer = OrderedDict()
        self._idx = {}  # keeps track of transitions saved for each env
        self._total_samples = 0  # keeps track of total samples, avoid sum(self._idx.values())
        for i in range(self.number_envs):
            self.env_buffer[i] = {
                'states': [],
                'actions': [],
                'rewards': [],
                'terminals': [],
                'ac_log_probs': []
            }
            self._idx[i] = 0


        self.mc = mc
        self.gae = gae

        if self.mc:
            assert not gae, 'Either gae or mc have to be true'
            self._adv_func = self._mc
            raise NotImplementedError
        elif self.gae:
            assert not mc, 'Either gae or mc have to be true'
            self._adv_func = self._gae
            for i in range(number_envs):
                self.env_buffer[i]['values'] = []
            self.gamma = gamma_gae
            self.lam = lambda_gae
        else:
            raise NotImplementedError('Either gae or mc have to be true')

    def is_full(self):
        if self._total_samples > self.capacity:
            raise ValueError
        return self._total_samples == self.capacity

    def add(self, experience: dict, env_id: int):
        assert experience.keys() == self.env_buffer[0].keys()

        if self.is_full():
            return

        for k, v in experience.items():
            self.env_buffer[env_id][k].append(v)

        self._total_samples += 1
        self._idx[env_id] += 1

    def clear(self):
        if self.gae:
            for i in range(self.number_envs):
                self.env_buffer[i] = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'ac_log_probs': [],
                    'values': []
                }
        else:
            for i in range(self.number_envs):
                self.env_buffer[i] = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'ac_log_probs': []
                }

    def sample(self, critic=None):
        self.env_buffer = {
            k_o: {
                k_i: torch.stack(v_i) for k_i, v_i in v_o.items()
            }
            for k_o, v_o in self.env_buffer.items()
        }

        # self.env_buffer['states'] = torch.stack([buf['states'] for _, buf in self.env_buffer.items()][0])
        # self.env_buffer['actions'] = torch.stack(self.env_buffer[k]['actions'] for k in self.env_buffer.keys() -
        #                                           self.env_buffer[0].keys())
        # self.env_buffer['ac_log_probs'] = torch.stack([self.env_buffer[k]['ac_log_probs'] for k in self.env_buffer.keys() -
        #                                           self.env_buffer[0].keys()][0])
        # self.env_buffer['rewards'] = torch.stack([self.env_buffer[k]['rewards'] for k in self.env_buffer.keys() -
        #                                           self.env_buffer[0].keys()][0])
        # self.env_buffer['terminals'] = torch.stack([self.env_buffer[k]['terminals'] for k in self.env_buffer.keys() -
        #                                           self.env_buffer[0].keys()][0])
        samples = []
        if self.gae:
            # self.env_buffer['values'] = torch.stack([self.env_buffer[k]['values'] for k in self.env_buffer.keys() -
            #                                       self.env_buffer[0].keys()][0])
            self._adv_func(critic)


        return samples

    def _gae(self, critic):
        assert critic is not None
        start = time.time()
        # sort env_id: transitions
        self._idx = {k: v for k, v in sorted(self._idx.items(), key=lambda i: i[1], reverse=True)}
        last_adv = {}
        # sort env_ids by number of transitions
        torch_blocks = defaultdict(list)
        for k, v in self._idx.items():
            torch_blocks[v].append(k)
            last_adv[k] = 0  # last advantage of collected transition
        # get last value, this operation requires python >= 3.7 because of dicts being implemented as ordered dicts
        last_val = critic(torch.stack([e['states'][-1] for e in self.env_buffer.values()]))
        last_val = {k:v for k,v in zip(self.env_buffer.keys(), last_val)}

        done_idx = []
        torch_blocks = list(torch_blocks.items())
        for i, (traj_length, idxs) in enumerate(torch_blocks):
            for idx in idxs:
                if idx not in done_idx:
                    self.env_buffer[idx]['adv'] = torch.zeros_like(self.env_buffer[idx[0]]['rewards'])
            #self.env_buffer[idxs]['adv'] = torch.zeros_like(self.env_buffer[idxs[0]]['rewards'])  # init adv



            done_idx.extend(idxs)


        for i, (number_transitions, ids) in enumerate(torch_blocks.items()):
            if i == len(torch_blocks):  # skip range from index 0
                break
            for tstep in reversed(range(list(torch_blocks.keys())[i+1], number_transitions)):
                pass
        for env_id, item in self.env_buffer.items():
            adv = torch.zeros(len(item['rewards']), dtype=torch.float32, device=self.device)
            last_adv = 0
            last_val = critic(item['states'][-1][None,:]).squeeze()
            for tstep in reversed(range(len(item['states']))):
                done = ~ item['terminals'][tstep]
                last_val *= done
                last_adv *= done
                delta = item['rewards'][tstep] + self.gamma * last_val - item['values'][tstep]
                last_adv = delta + self.gamma * self.lam * last_adv
                adv[tstep] = last_adv
                last_val = item['values'][tstep]
            self.env_buffer[env_id]['adv'] = adv.detach()
        print(f"gae time {time.time()-start}")


        # adv = torch.zeros_like(self.env_buffer['rewards'],
        #                        dtype=torch.float32, device=self.device)
        #
        # # this is gae version if not-complete episodes in memory
        # last_adv = 0
        # last_val = critic(torch.stack([e[-1] for e in self.env_buffer['states']])).squeeze()
        # # last_val = last_val.cpu().data.numpy()
        #
        # for tstep in reversed(range(self.cap_p_env)):
        #     mask = ~ self.env_buffer['terminals'][:, tstep]
        #     last_val = last_val * mask
        #     last_adv = last_adv * mask
        #     delta = self.env_buffer['rewards'][:, tstep] + self.gamma * last_val - \
        #             self.env_buffer['values'][:, tstep]
        #     last_adv = delta + self.gamma * self.lam * last_adv
        #     adv[:, tstep] = last_adv
        #     last_val = self.env_buffer['values'][:, tstep]
        # return adv

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


class OnPolBuffer(object):

    def __init__(self, size: int, state_shape, action_shape, device,
                 # $experience_keys for different on_pol_algos
                 number_envs: int, gae: bool = True, mc: bool = False,
                 gamma_gae: float = 0.99, lambda_gae: float = 0.95):

        assert size % number_envs == 0, \
            f"Choose multiple of number envs as batch_size ({size}, {number_envs})"

        self.device = device
        self.cap_p_env = size // number_envs

        self.env_buffer = OrderedDict({
            'states': torch.zeros((number_envs, self.cap_p_env, state_shape),
                                  dtype=torch.float32, device=self.device),
            'actions': torch.zeros((number_envs, self.cap_p_env, action_shape),
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

    def add(self, experience: dict, env_id: int):
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
