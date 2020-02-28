# TODO Batchnormalization

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# good seeds:
# 115, random.seed(89)


#random.seed(90)
# not for ReLU

def xavier_uniform(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Kaiming for ReLU
def kaiming_uniform(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_( m.weight, nonlinearity='relu',)
        m.bias.data.fill_(0.01)

def kaiming_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal(m.weight, nonlinearity='relu', mode='fan_in')
        #m.bias.data.fill_(0.01)

class Actor(nn.Module):

    def __init__(self, action_dim, state_dim, init='default'):
        super(Actor, self).__init__()

        self.p_net = nn.Sequential(
            nn.Linear(state_dim, 14),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(14, 14),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(14, action_dim),
            nn.Sigmoid()
        )

        print("Initializing Actor with {}".format(init))
        if init=='xavier_uniform':
            fn = xavier_uniform
            self.apply(fn)
        elif init=='kaiming_uniform':
            fn = kaiming_uniform
            self.apply(fn)
        elif init=='kaiming_normal':
            fn = kaiming_normal
            self.apply(fn)
        elif init=='default':
            pass
        else:
            print("Could not recognize initialization {}. Choosing default".format(init))
            pass

        print(self)

    def forward(self, xs):
        xs = self.p_net(xs)
        return xs


class Critic(nn.Module):

    def __init__(self, action_dim, state_dim, init='kaiming_uniform'):
        super(Critic, self).__init__()
        self.sc = nn.Sequential(
            nn.Linear(state_dim, 14),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(14, 14),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(14, 14),
            nn.PReLU(),
            nn.Dropout(0.2)
        )

        self.ac = nn.Sequential(
            nn.Linear(action_dim, 14),
            nn.PReLU(),
            nn.Dropout(0.2))
        self.Q = nn.Linear(28, 1)

        print("Initializing Critic with {}".format(init))
        if init == 'xavier_uniform':
            fn = xavier_uniform
            self.apply(fn)
        elif init=='kaiming_uniform':
            fn = kaiming_uniform
            self.apply(fn)
        elif init=='kaiming_normal':
            fn = kaiming_normal
            self.apply(fn)
        elif init == 'default':
            pass
        else:
            print("Could not recognize initialization {}. Choosing default".format(init))
            pass

        print(self)

    def forward(self, xs, xa):

        xs = self.sc(xs)

        xa = self.ac(xa)

        q = torch.cat((xa, xs), -1)

        q = self.Q(q)

        return q

if __name__ == '__main__':
    import random

    def find_seed():
        done = False
        while not done:

            sed = int(random.random() * 10000)
            torch.manual_seed(sed)

            ac = Actor(2, 4)
            c = Critic(2, 4)
            s = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
            out = ac(s)
            heights = []
            pause= []
            cs = []

            for i in range(1000):
                # torch.manual_seed(i)
                # random.seed(i)
                # ac = Actor(2, 4, init='kaiming_uniform')
                # s = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
                # out = ac(s)
                # if (out[0] != 0.5) and (out[1] != 0.5):
                s = torch.rand((1, 4), dtype=torch.float32)
                #print(s)
                out = ac(s)
                heights.append(out[0][0].data.numpy())
                pause.append(out[0][1].data.numpy())
                c_out = c(s, out)
                cs.append(c_out[0][0].item())
                #print(s, out[0], c_out[0][0].data.numpy())

            print("seed: {}".format(sed))

            print(" Height {} - {} avg: {}\n Pause {} - {} avg: {}\n c {} - {}".format(str(min(heights)), str(max(heights)), str(sum(heights)/len(heights)),
                                                                                       str(min(pause)), str(max(pause)), str(sum(pause)/len(pause)),
                                                                                       str(min(cs)), str(max(cs))))
            if (sum(pause)/len(pause) > 0.1):
                print("success")
                done=True

    def test_seed(seed):
        torch.manual_seed(seed)

        ac = Actor(2, 4)
        c = Critic(2, 4)
        s = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
        out = ac(s)
        heights = []
        pause = []
        cs = []

        for i in range(1000):
            # torch.manual_seed(i)
            # random.seed(i)
            # ac = Actor(2, 4, init='kaiming_uniform')
            # s = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
            # out = ac(s)
            # if (out[0] != 0.5) and (out[1] != 0.5):
            s = torch.rand((1, 4), dtype=torch.float32)
            # print(s)
            out = ac(s)
            heights.append(out[0][0].data.numpy())
            pause.append(out[0][1].data.numpy())
            c_out = c(s, out)
            cs.append(c_out[0][0].item())
            # print(s, out[0], c_out[0][0].data.numpy())

        print("seed: {}".format(seed))

        print(" Height {} - {} avg: {}\n Pause {} - {} avg: {}\n c {} - {}".format(str(min(heights)), str(max(heights)),
                                                                               str(sum(heights) / len(heights)),
                                                                               str(min(pause)), str(max(pause)),
                                                                               str(sum(pause) / len(pause)), str(min(cs)), str(max(cs))))
        test_seed(193)

    # ac = Actor(2, 4)
    # c = Critic(2, 4)
    # print(ac)
    # print(c(torch.tensor([1, 1, 1, 0], dtype=torch.float32), torch.tensor([0.23, 0.78], dtype=torch.float32)))