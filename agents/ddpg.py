"""
"""

import copy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.nnfactory.ac import Actor, Critic
from agents.utils.replay_buffer import ReplayBuffer
from agents.utils.noise_emerging_gaussian import emerging_gaussian

#random.seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgentDDPG(object):

    def __init__(self, config):

        # print("Init agents with cuda: {} device: {}".format(torch.cuda.is_available(),
        #                                                     torch.cuda.get_device_name(
        #                                                     torch.cuda.current_device())))

        self.critic_lr = config['critic_lr']
        self.actor_lr = config['actor_lr']
        self.actor_init = config["actor_init"]
        self.critic_init = config["critic_init"]
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.weight_decay = config["weight_decay"]
        self.memory_size = config['memory_size']
        self.tau = config['tau']
        self.root_path = config['root_path']
        self.run_name = config["run_name"]
        self.backup_interval = config["backup_interval"]
        self.action_dim = config["action_dim"]
        self.state_dim = config["state_dim"]
        torch.manual_seed(config['seed'])

        # Initialize Neural Networks from nnfactory/ac.py
        self.critic = Critic(self.action_dim, self.state_dim, init=self.critic_init).to(device)
        self.actor = Actor(self.action_dim, self.state_dim, init=self.actor_init).to(device)

        # target network
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.opt_critic = torch.optim.AdamW(lr=self.critic_lr, params=self.critic.parameters(), weight_decay=self.weight_decay)
        self.opt_actor = torch.optim.AdamW(lr=self.actor_lr, params=self.actor.parameters(), weight_decay=self.weight_decay)

        # If the run number is > 0, assume we continue training from previous checkpoint
        run_no = int(self.run_name.split('_')[1])

        if run_no > 0:
            previous = 'run_{}'.format(run_no - 1)
            # load previous models (state dictionaries) and optimizers
            try:
                print('Loading agents from {}/{}'.format(self.root_path, previous))
                self.critic.load_state_dict(torch.load('{}/{}/Critic.pt'.format(self.root_path, previous)))
                self.actor.load_state_dict(torch.load('{}/{}/Actor.pt'.format(self.root_path, previous)))
                self.critic_target.load_state_dict(torch.load('{}/{}/TCritic.pt'.format(self.root_path, previous)))
                self.actor_target.load_state_dict(torch.load('{}/{}/TActor.pt'.format(self.root_path, previous)))
                self.opt_critic.load_state_dict(torch.load('{}/{}/OptCritic.pt'.format(self.root_path, previous)))
                self.opt_actor.load_state_dict(torch.load('{}/{}/OptActor.pt'.format(self.root_path, previous)))
            except FileNotFoundError:
                print('Could not locate savefiles in {}/{}'.format(self.root_path, previous))
                print(' \nUsing new agents/Critic and Optimizers! \n')
                # equalize weights for targets, bec new networks
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            # init new model, because we are at start of new experiment

            # target networks should have same weights in the beginning (only if we are not! restoring from a savepoint!)
            # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            print("Creating new agents")
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.eval()
            self.critic_target.eval()

        # memory
        self.memory = ReplayBuffer(buffer_size=self.memory_size, batch_size=self.batch_size, seed=0)

        # noise
        self.noise = emerging_gaussian

        # train flag for initial random runs
        self.train = True

        # tensorboard
        self.no_step = 0
        actor_dummy = torch.tensor([0.5 for _ in range(self.state_dim)]).to(device)
        #critic_dummy_action = torch.tensor([0.5 for _ in range(self.action_dim)]).to(device)
        #critic_dummy_state = torch.tensor([0.5 for _ in range(self.state_dim)]).to(device)

        #res = self.critic.forward(critic_dummy_state, critic_dummy_action)

        self.tb = SummaryWriter('{}/{}'.format(self.root_path, self.run_name))
        self.tb.add_graph(self.actor, (actor_dummy, ))
        #self.tb.add_graph(self.critic, input_to_model=([critic_dummy_state, critic_dummy_action]))
        self.tb.flush()

    def learn(self):
        # https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301

        if self.train:
            states, actions, rewards, next_states, terminals = self.memory.sample()
            print("training")
            # print("Got memory:")
            # print('states', type(states),states)
            # print('actions',type(actions),actions)
            # print('rewards',type(rewards),rewards)
            # print('next_states',type(next_states),next_states)
            # print('terminals',type(terminals), terminals)

            proposed_action = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, proposed_action)

            target_Q = rewards + ((1-terminals) * self.gamma * target_Q).detach()
            current_Q = self.critic(states, actions)

            self.tb.add_scalar('Critic_target/TargetQ_mean', target_Q.mean(), self.no_step)
            self.tb.add_scalar('Critic/CurrentQ_mean', current_Q.mean(), self.no_step)

            # print("current_q: \n{}".format(current_Q))
            # print("target_q: \n{}".format(target_Q))
            critic_loss = F.mse_loss(current_Q, target_Q)

            self.opt_critic.zero_grad() # erase old gradients
            critic_loss.backward()      # calculate new gradients using critic_loss
            self.opt_critic.poll()      # let the optimizer step once using the previously computed gradients

            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.poll()

            # tensorboard
            for tag, param in self.critic.named_parameters():
                #print("hist crit {}".format(tag))
                self.tb.add_histogram('Critic/param_{}'.format(tag), param.data.cpu().numpy(), self.no_step)
                self.tb.add_histogram('Critic/grad_{}'.format(tag), param.grad.data.cpu().numpy(), self.no_step)

            for tag, param in self.actor.named_parameters():
                #print("hist act {}".format(tag))
                self.tb.add_histogram('Actor/param_{}'.format(tag), param.data.cpu().numpy(), self.no_step)
                self.tb.add_histogram('Actor/grad_{}'.format(tag), param.grad.data.cpu().numpy(), self.no_step)

            self.tb.add_scalar('Critic/loss', critic_loss, self.no_step)
            self.tb.add_scalar('Actor/loss', actor_loss, self.no_step)
            self.tb.flush()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def step(self, s, a, r, s_, t, tensorboard=None):
        if tensorboard is not None:
            if tensorboard['test']:
                # only add the result of our test run to tensorboard (currently reward is 0 if goal not reached and [-1, 1] if terminal state reached)
                if tensorboard['reward'] != 0:
                    self.tb.add_scalar('Test/reward', tensorboard['reward'], self.no_step)

                self.tb.add_scalar('Test/exploration_prob', tensorboard['noise'], self.no_step)
                self.tb.add_scalars('Test/actions',{'noise_height': a[0], 'noise_pause': a[1],
                                                     'height': tensorboard['orig_action'][0],
                                                     'pause': tensorboard['orig_action'][1]}, self.no_step)
            else:
                # we dont need to see many 0 on our graph
                if tensorboard['reward'] != 0:
                    self.tb.add_scalar('Step/rewards', tensorboard['reward'], self.no_step)
                self.tb.add_scalar('Step/exploration_prob', tensorboard['noise'], self.no_step)
                self.tb.add_scalars('Step/actions', {'noise_height': a[0], 'noise_pause': a[1],
                                                     'height': tensorboard['orig_action'][0],
                                                     'pause': tensorboard['orig_action'][1]}, self.no_step)

        self.no_step += 1
        self.memory.add(s, a, r, s_, t)
        if len(self.memory) > self.batch_size:
            self.learn()

        # save the model every n steps
        if self.no_step % self.backup_interval == 0:
            # set evaluation mode (disable dropout, batchnorm)
            # https://pytorch.org/docs/stable/nn.html?highlight=eval#torch.nn.Module.eval
            self.actor.eval()
            self.critic.eval()
            self.actor_target.eval()
            self.critic_target.eval()

            torch.save(self.actor.state_dict(), '{}/{}/Actor.pt'.format(self.root_path, self.run_name))
            torch.save(self.critic.state_dict(), '{}/{}/Critic.pt'.format(self.root_path, self.run_name))
            torch.save(self.actor_target.state_dict(), '{}/{}/TActor.pt'.format(self.root_path, self.run_name))
            torch.save(self.critic_target.state_dict(), '{}/{}/TCritic.pt'.format(self.root_path, self.run_name))
            torch.save(self.opt_actor.state_dict(), '{}/{}/OptActor.pt'.format(self.root_path, self.run_name))
            torch.save(self.opt_critic.state_dict(), '{}/{}/OptCritic.pt'.format(self.root_path, self.run_name))

            self.actor.train()
            self.critic.train()
            self.actor_target.train()
            self.critic_target.train()

    def act(self, state: list, exploration_prob=0):
        # no_grad() ?
        # actor.eval(), actor.train() needed if using dropout or batchnorm
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval


        state = torch.tensor(state, dtype=torch.float32).to(device)

        self.actor.eval()
        action = self.actor(state)
        self.actor.train()

        action_orig = copy.deepcopy(list(action.cpu().data.numpy().flatten()))
        action = list(action.cpu().data.numpy().flatten())
        action[0] = self.noise(action_orig[0], exploration_prob, 0, 1)
        action[1] = self.noise(action_orig[1], exploration_prob, 0, 1)

        print('a:', action, 'a_orig:', action_orig)


        return action, action_orig


if __name__ == '__main__':
    ac = AgentAC('./agent_config.json')
