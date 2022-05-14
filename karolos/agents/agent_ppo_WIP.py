# https://github.com/nikhilbarhate99/PPO-PyTorch/issues/38
#

import copy
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

sys.path.append(str(Path(__file__).resolve().parent))

from agent import Agent
from nn import NeuralNetwork, init_xavier_uniform, Clamp, Critic, Actor


# https://intellabs.github.io/coach/components/agents/actor_optimization/ppo.html


class AgentPPO(Agent):
    def __init__(self, config, state_space, goal_space, action_space,
                 reward_function, experiment_dir='.'):

        super(AgentPPO, self).__init__(
            config, state_space, goal_space, action_space,
            reward_function, experiment_dir
        )

        self.replay_buffer.experience_keys += ["logprob", "value", "advantage"]

        self.learning_rate_critic = config.get("learning_rate_critic", 1e-4)
        self.learning_rate_actor = config.get("learning_rate_actor", 5e-5)

        self.weight_decay = config.get("weight_decay", 5e-5)
        self.batch_size = config.get("batch_size", 64)
        self.gradient_steps = config.get("gradient_steps", 40)
        self.minibatch_size = config.get("minibatch_size", self.batch_size)  # crash if not in config
        self.reward_discount = config.get("reward_discount", 0.99)
        self.generalized_advantage_discount = config.get("gae_discount", 0.95)
        self.clip_eps = config.get("clip_eps", 0.15)
        # self.action_std_init = config.get("action_std_init", 0.9)
        # self.action_std_decay_rate = config.get("action_std_decay", 0.03)
        # self.action_std_decay_freq = config.get("action_std_decay_freq", 500_000)

        self.loss_value_coeff = config.get("value_loss_coeff", 0.5)
        self.loss_entropy_coeff = config.get("entropy_loss_coeff", 0.01)

        # self.adam_epsilon = config.get("adam_epsilon", 1e-5)

        actor_structure = config.get('actor_structure', [])
        critic_structure = config.get('critic_structure', [])

        self.actor = Actor((self.state_dim, self.goal_dim), self.action_dim, actor_structure).to(self.device)
        self.critic = Critic((self.state_dim, self.goal_dim), critic_structure).to(self.device)

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate_actor,
            # weight_decay=self.weight_decay,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic,
            # weight_decay=self.weight_decay,
        )
        self.loss = nn.MSELoss()

        self.learn_steps = 0
        self.log_step = 1

    # def add_experience(self, experience, action_meta=None):
    #     experience["logprob"], experience["value"], experience["advantage"] = action_meta
    #     super(AgentPPO, self).add_experience(experience)

    def add_experiences(self, experiences):
        assert not any([e["done"] for e in experiences[:-1]])
        assert experiences[-1]["done"]

        # handle advantage of last experience
        logprob, value = experiences[-1].pop("action_meta")
        advantage = experiences[-1]["reward"] - value
        experiences[-1]["logprob"] = logprob
        experiences[-1]["value"] = value
        experiences[-1]["advantage"] = advantage

        for ii in reversed(range(len(experiences) - 1)):
            next_value = experiences[ii + 1]["value"]
            next_advantage = experiences[ii + 1]["advantage"]
            logprob, value = experiences[ii].pop("action_meta")

            delta = experiences[ii]["reward"] + self.reward_discount * next_value - value
            advantage = delta + self.reward_discount * self.generalized_advantage_discount * next_advantage

            experiences[ii]["logprob"] = logprob
            experiences[ii]["value"] = value
            experiences[ii]["advantage"] = advantage

        super(AgentPPO, self).add_experiences(experiences)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        experiences, _ = self.replay_buffer.sample(self.batch_size, remove=True)
        # self.replay_buffer.clear()

        states, goals, actions, rewards, next_states, next_goals, dones, logprobs, values, advantages = experiences

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # next_goals = torch.FloatTensor(next_goals).to(self.device)
        # dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
        logprobs = torch.FloatTensor(logprobs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        advantages_mean = advantages.mean()
        advantages_std = advantages.std() + 1e-8

        for i in range(self.gradient_steps):
            # shuffle training data
            indices = torch.randperm(self.batch_size)
            minibatches = [indices[start: start + self.minibatch_size] for start in
                           range(0, self.batch_size, self.minibatch_size)]

            for minibatch_indices in minibatches:
                states_ = states[minibatch_indices]
                goals_ = goals[minibatch_indices]
                actions_ = actions[minibatch_indices]
                old_logprobs_ = logprobs[minibatch_indices]
                advantages_ = advantages[minibatch_indices]
                values_ = values[minibatch_indices]

                new_logprobs, entropy = self.actor.evaluate(states_, goals_, action=actions_)
                new_values = self.critic(states_, goals_)

                prob_ratios = (new_logprobs - old_logprobs_).exp()
                prob_ratios = prob_ratios.reshape(-1, 1)

                advantages_normed = advantages_ - advantages_mean
                advantages_normed /= advantages_std

                weighted_probs = advantages_normed * prob_ratios
                weighted_clipped_probs = advantages_normed * torch.clamp(prob_ratios, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantages_ + values_
                critic_loss = (returns - new_values) ** 2
                critic_loss = critic_loss.mean()

                entropy_loss = -entropy.mean()

                loss = actor_loss + self.loss_value_coeff * critic_loss + self.loss_entropy_coeff * entropy_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()

    def predict(self, states, goals, deterministic):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)

        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()

            actions, logprobs = self.actor(states, goals, deterministic=deterministic)
            values = self.critic(states, goals)

        actions = actions.detach().cpu().numpy()
        logprobs = logprobs.detach().cpu().numpy()
        values = values.detach().cpu().numpy()

        actions = actions.clip(self.action_space.low, self.action_space.high)

        return actions, logprobs, values

    def save(self, path):
        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), osp.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), osp.join(path, "critic.pt"))
        # torch.save(self.actor_old.state_dict(), osp.join(path, "actor_old.pt"))

        torch.save(
            self.optimizer_critic.state_dict(), osp.join(path, "optimizer_critic.pt")
        )
        torch.save(
            self.optimizer_actor.state_dict(), osp.join(path, "optimizer_actor.pt")
        )

    def load(self, path):

        self.actor.load_state_dict(
            torch.load(osp.join(path, "actor.pt"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(osp.join(path, "critic.pt"), map_location=self.device)
        )
        self.optimizer_actor.load_state_dict(
            torch.load(osp.join(path, "optimizer_actor.pt"), map_location=self.device)
        )
        self.optimizer_critic.load_state_dict(
            torch.load(osp.join(path, "optimizer_critic.pt"), map_location=self.device)
        )


if __name__ == "__main__":
    from karolos.environments.environment_gym_wrapper import GymWrapper


    # LunarLanderContinuous-v2
    # MountainCarContinuous-v0
    def ppo_gym(name="LunarLanderContinuous-v2"):
        render = False
        epochs = 15000
        activation = "tanh"
        config = {
            "learning_rate_critic": 1e-4,
            "learning_rate_actor": 1e-4,
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

            "actor_structure": [
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
        agent = AgentPPO(config, env.state_space, env.goal_space, env.action_space,
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

            trajectory = []

            while not done:
                action, *action_meta = agent.predict([state], [], deterministic=False)
                next_state, reward, done = env.step(action[0])

                next_state = next_state["state"].flatten()
                reward = reward["achieved"]["reward"]

                # agent.memory.add(env_id=0, experience={
                experience = [{
                    "state": state,
                    "action": agent_info[0]['actions'],  # ppo trains on un-clipped actions
                    "reward": torch.tensor(reward),
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
                                                                                             agent.actor.action_std,
                                                                                             total_step))


    ppo_gym()
