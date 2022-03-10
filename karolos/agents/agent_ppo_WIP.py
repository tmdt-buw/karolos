# https://github.com/nikhilbarhate99/PPO-PyTorch/issues/38
#

import copy
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from karolos.agents.agent import OnPolAgent
from karolos.agents.nn import NeuralNetwork, init_xavier_uniform, Clamp


# https://intellabs.github.io/coach/components/agents/policy_optimization/ppo.html


class PolicyPPO(NeuralNetwork):
    def __init__(
            self,
            state_dims,
            action_dim,
            network_structure,
            action_stddev_init,
            device,
    ):
        out_dim = int(np.product(action_dim)) * 2
        in_dim = int(np.sum([np.product(state_dim) for state_dim in state_dims]))

        network_structure = copy.deepcopy(network_structure)
        super(PolicyPPO, self).__init__(in_dim, network_structure)

        dummy = super(PolicyPPO, self).forward(torch.zeros((1, in_dim)))

        self.action_dim = action_dim
        self.operators.append(nn.Linear(dummy.shape[1], out_dim))
        self.device = device
        self.operators.apply(init_xavier_uniform)

        self.action_std = action_stddev_init
        self.mean_head = nn.Linear(out_dim // 2, out_dim // 2)
        self.std_head = nn.Linear(out_dim // 2, out_dim // 2)
        # self.action_var = torch.full(self.action_dim, self.action_std_log,
        #                              requires_grad=True, dtype=torch.float)\
        #     .to(self.device)

        self.action_var = torch.full(self.action_dim,
                                     self.action_std * self.action_std).to(self.device)

        # self.action_var_learn = torch.nn.Parameter(torch.ones(1, action_dim[0]) * 0)

        self.std_clamp = Clamp(-20, 2)
        self.std_activation = torch.nn.Softplus()

    def forward(self, *state_args, deterministic=False):
        x = super(PolicyPPO, self).forward(*state_args)
        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)
        mean = self.mean_head(mean)
        log_std = self.std_head(log_std)

        # std = log_std.exp()
        # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        if deterministic:
            # action = torch.tanh(mean)
            action = mean
            # log_prob = torch.zeros_like(torch.diag(self.action_var).unsqueeze(dim=0))
            log_prob = torch.zeros_like(action)
        else:
            # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # normal = MultivariateNormal(mean, covariance_matrix=cov_mat)

            # action_std = self.action_var.expand_as(mean).exp()
            # normal = Normal(mean, action_std)

            # log_std = self.std_clamp(log_std)
            # log_std = self.std_activation(log_std)
            # std = log_std.exp()

            std = self.std_activation(log_std)

            normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))

            action = normal.rsample()
            log_prob = normal.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state_args, action_args):
        x = super(PolicyPPO, self).forward(state_args)
        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)
        mean = self.mean_head(mean)
        log_std = self.std_head(log_std)

        # log_std = self.std_clamp(log_std)
        # log_std = self.std_activation(log_std)
        # std = log_std.exp()

        std = self.std_activation(log_std)

        normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))

        # mean = super(PolicyPPO, self).forward(state_args)
        # action_var = self.action_var.expand_as(mean)
        # cov_mat = torch.diag_embed(action_var).to(self.device)
        # normal = MultivariateNormal(mean, cov_mat)

        # action_std = self.action_var.expand_as(mean).exp()
        # normal = Normal(mean, action_std)

        action_logprob = normal.log_prob(action_args)
        entropy = normal.entropy()

        return action_logprob, entropy

    def set_action_std(self, action_std):
        return
        self.action_std = action_std
        self.action_var = torch.full(
            self.action_dim, self.action_std * self.action_std
        ).to(self.device)


class CriticPPO(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        # if isinstance(state_dims, dict):
        #     raise NotImplementedError
        # if len(state_dims) == 1:
        #     in_dim = int(np.sum([np.product(arg) for arg in state_dims]))
        # elif (
        #     len(state_dims) == 3
        # ):  # 2d state input, 1d action input -> check net structure keys for this
        #     in_dim = {0: state_dims}
        # else:
        #     raise NotImplementedError(f"Unknown dims: state:{state_dims}")

        in_dim = int(np.sum([np.product(arg) for arg in state_dims]) + np.product(action_dim))

        network_structure = copy.deepcopy(network_structure)

        super(CriticPPO, self).__init__(
            in_dim=in_dim, network_structure=network_structure)

        dummy = super(CriticPPO, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *args):
        args = torch.cat(args, dim=-1)
        return super(CriticPPO, self).forward(args)


class AgentPPO(OnPolAgent):
    def __init__(self, config, state_space, goal_space, action_space,
                 reward_function, experiment_dir='.'):

        super(AgentPPO, self).__init__(
            config, state_space, goal_space, action_space,
            reward_function, experiment_dir
        )

        self.is_transfer_learning = False
        self.is_on_policy = True

        self.learning_rate_critic = config.get("learning_rate_critic", 1e-4)
        self.learning_rate_policy = config.get("learning_rate_policy", 5e-5)

        self.weight_decay = config.get("weight_decay", 5e-5)
        self.batch_size = config.get("batch_size", 96000)
        self.gradient_steps = config.get("gradient_steps", 40)
        self.n_mini_batch = config.get("n_mini_batch", 20)  # crash if not in config
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert self.batch_size % self.n_mini_batch == 0
        self.reward_discount = config.get("reward_discount", 0.99)
        self.clip_eps = config.get("clip_eps", 0.15)
        self.grad_clip = config.get("gradient_clipping", False)
        self.grad_norm = config.get("gradient_normalization", True)  # good for ppo
        self.action_std_init = config.get("action_std_init", 0.9)
        self.action_std_decay_rate = config.get("action_std_decay", 0.03)
        self.action_std_decay_freq = config.get("action_std_decay_freq", 500_000)
        self.next_decay_step = self.action_std_decay_freq
        self.min_action_std = config.get("min_action_std", 0.1)

        self.loss_value_coeff = config.get("value_loss_coeff", 0.5)
        self.loss_entropy_coeff = config.get("entropy_loss_coeff", 0.01)

        self.adam_epsilon = config.get("adam_epsilon", 1e-5)

        # self.reward_scale = config.get('reward_scale', 10.)

        policy_structure = config.get('policy_structure', [])
        critic_structure = config.get('critic_structure', [])

        self.critic = CriticPPO((self.state_dim, self.goal_dim), self.action_dim, critic_structure).to(self.device)
        self.test_policy = None
        self.policy = PolicyPPO((self.state_dim, self.goal_dim), self.action_dim, policy_structure,
            action_stddev_init=self.action_std_init, device=self.device
        ).to(self.device)
        # self.policy_old = PolicyPPO(self.state_dim, self.action_dim,
        #                        self.policy_structure,
        #                        device=self.device).to(self.device)
        # self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer_policy = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate_policy,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon
        )
        self.loss = nn.MSELoss()

        self.learn_steps = 0
        self.log_step = 1

    def _decay_action_std(self, step):
        if step >= self.next_decay_step:
            action_std = self.policy.action_std - self.action_std_decay_rate
            action_std = np.round(action_std, 4)
            if action_std <= self.min_action_std:
                action_std = self.min_action_std
            self.policy.set_action_std(action_std)
            self.next_decay_step += self.action_std_decay_freq

    def learn(self):
        self._decay_action_std(self.learning_step)

        if not self.replay_buffer.is_full():
            return

        experience = self.replay_buffer.sample(self.critic)

        if self.learn_steps % self.log_step == 0:
            losses = []
            v_losses = []
            rets = []
            a_logprobs = []
            ratios_log = []
            entropies = []

        for i in range(self.gradient_steps):
            # print(i)
            idxs = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = idxs[start:end]
                samples = [
                    e[mb_idx] for e in experience
                ]  # states, actions, ac_log_probs, values, gae_advantages

                samples_ret = samples[3] + samples[4]

                # or if samples_ret.std() close to 0 -> samples_ret.std() = 1
                # normalize advantages
                samples_ret_norm = (samples_ret - samples_ret.mean()) / (
                        samples_ret.std() + 1e-8
                )
                log_pis, pi_entropies = self.policy.evaluate_actions(
                    samples[0], samples[1]
                )
                vals = self.critic(samples[0])
                ratios = torch.exp(log_pis - samples[2])

                clipped_ratios = ratios.clamp(
                    min=1.0 - self.clip_eps, max=1.0 + self.clip_eps
                )
                pol_reward = torch.min(
                    ratios * samples_ret_norm, clipped_ratios * samples_ret_norm
                ).mean()

                # value function clipping
                clipped_vals = samples[3] + (vals - samples[3]).clamp(
                    min=-self.clip_eps, max=self.clip_eps
                )
                val_loss = torch.max(
                    (vals - samples_ret) ** 2, (clipped_vals - samples_ret) ** 2
                )
                val_loss = val_loss.mean()

                loss = -(pol_reward - self.loss_value_coeff * val_loss +
                         self.loss_entropy_coeff * pi_entropies.mean())

                self.optimizer_policy.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0)
                    nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=1.0)
                if self.grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=0.5, norm_type=2
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=0.5, norm_type=2
                    )

                self.optimizer_policy.step()
                self.optimizer_critic.step()

                if self.learn_steps % self.log_step == 0:
                    v_losses.append(val_loss)
                    losses.append(loss)
                    rets.append(samples_ret.mean())
                    a_logprobs.append(log_pis.mean())
                    ratios_log.append(ratios.mean())
                    entropies.append(pi_entropies.mean())

        if self.writer and (self.learn_steps % self.log_step == 0):
            self.writer.add_scalar("loss", np.mean([l.item() for l in losses]), step)
            self.writer.add_scalar(
                "value loss", np.mean([l.item() for l in v_losses]), step)
            self.writer.add_scalar(
                "advantages", np.mean([r.item() for r in rets]), step
            )
            self.writer.add_scalar(
                "action_logprob", np.mean([al.item() for al in a_logprobs]), step
            )

            self.writer.add_histogram(
                "action_logprob_hist", self.policy.action_var, step
            )
            self.writer.add_scalar(
                "ratio", np.mean([rl.item() for rl in ratios_log]), step
            )
            self.writer.add_scalar(
                "entropy", np.mean([e.item() for e in entropies]), step
            )

        self.replay_buffer.clear()
        self.learn_steps += 1

    def get_state_infos(self, states, goals, deterministic):
        self.policy.eval()
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)

        with torch.no_grad():
            actions, logprob = self.policy(states, goals, deterministic=deterministic)
            values = self.critic(states, goals, actions).detach()
            logprob = logprob.detach().cpu()

        return [{"ac_log_probs": lgp, "values": v, "actions": ac} for lgp, v, ac in zip(logprob, values, actions)]

    def predict(self, states, goals, deterministic):
        self.policy.eval()
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action, logprob = self.policy(states, goals, deterministic=deterministic)

        action_cpu = action.detach().cpu().numpy()
        action_clip = action_cpu.clip(self.action_space.low, self.action_space.high)

        return action_clip

    def save(self, path):
        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), osp.join(path, "policy.pt"))
        torch.save(self.critic.state_dict(), osp.join(path, "critic.pt"))
        # torch.save(self.policy_old.state_dict(), osp.join(path, "policy_old.pt"))

        torch.save(
            self.optimizer_critic.state_dict(), osp.join(path, "optimizer_critic.pt")
        )
        torch.save(
            self.optimizer_policy.state_dict(), osp.join(path, "optimizer_policy.pt")
        )

    def load(self, path):

        self.policy.load_state_dict(
            torch.load(osp.join(path, "policy.pt"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(osp.join(path, "critic.pt"), map_location=self.device)
        )
        self.optimizer_policy.load_state_dict(
            torch.load(osp.join(path, "optimizer_policy.pt"), map_location=self.device)
        )
        self.optimizer_critic.load_state_dict(
            torch.load(osp.join(path, "optimizer_critic.pt"), map_location=self.device)
        )


if __name__ == "__main__":
    from karolos.environments.environment_wrappers.gym_wrapper import GymWrapper


    # LunarLanderContinuous-v2
    # MountainCarContinuous-v0
    def test_ppo_gym(name="LunarLanderContinuous-v2"):
        render = False
        epochs = 15000
        activation = "tanh"
        config = {
            "learning_rate_critic": 1e-4,
            "learning_rate_policy": 1e-4,
            "weight_decay": 5e-5,
            "batch_size": 4096,
            "reward_discount": 0.99,
            "clip_eps": 0.1,
            "gradient_steps": 8,
            "n_mini_batch": 4,
            "action_std_init": 0.9,
            "action_std_decay": 0.2,
            "action_std_decay_freq": 50_000,
            "min_action_std": 0.1,
            "value_loss_coeff": 0.5,
            "entropy_loss_coeff": 0.01,

            "policy_structure": [
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
            ],
            "critic_structure": [
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
                ("linear", 128),
                (activation, None),
            ]

        }

        # config["exp_per_cpu"] = config["batch_size"]  # one process

        env = GymWrapper(name=name, max_steps=5000)
        agent = AgentPPO(config, env.observation_space, env.action_space,
                         reward_function=None)
        total_step = 0
        score = 0
        for i_epoch in range(epochs):

            if score > 0:
                render = False
            score = 0
            state = env.reset()
            state = state[0]['state']
            if render: env.render()
            done = False
            step = 0
            while not done:
                action, agent_info = agent.predict([state], deterministic=False)
                next_state, reward, done = env.step(action[0])

                next_state = next_state["state"].flatten()
                reward = reward["achieved"]["reward"]

                # agent.memory.add(env_id=0, experience={
                experience = [{
                    "states": torch.tensor(state),
                    "actions": agent_info[0]['actions'],  # ppo trains on un-clipped actions
                    "rewards": torch.tensor(reward),
                    "ac_log_probs": agent_info[0]['ac_log_probs'],
                    "terminals": torch.tensor(done),
                    "values": agent_info[0]['values'],
                }]
                agent.add_experiences(experience)

                if render: env.render()
                # if agent.store_transition(trans):
                agent.train(step=total_step)
                step += 1
                total_step += 1
                score += reward
                state = next_state

            # if i_epoch % 10 == 0:
            print("Epoch {},\tscore: {:.2f},\tstep: {},\tstddev: {},\ttotal_step: {}".format(i_epoch,
                                                                                             score, step,
                                                                                             agent.policy.action_std,
                                                                                             total_step))


    test_ppo_gym()
