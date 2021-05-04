import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from .models import FeatureExtractor, ValueNetwork, SoftQNetwork, PolicyNetwork
from .replay_buffers import BasicBuffer
from .checkpointer import Checkpointer

class SACAgent:
    def __init__(self,
                 action_range, action_dim, gamma, tau, v_lr, q_lr, pi_lr, buffer_maxlen=int(1e6),
                 image_size=(256,256,3), kernel_size=(3,3), conv_channels=4,
                 logFile='logs/losses.txt'):
        #TODO: Known issue when cuda is enabled, robot can't de-pickle stuff from server
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.action_range = action_range
        self.action_dim = action_dim

        self.image_size = tuple(image_size)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        # Logging
        self.logFile = open(logFile, 'w')

        # Network initialization
        self.fe = FeatureExtractor(image_size[2], conv_channels, kernel_size).to(self.device)
        self.in_dim = self.fe.get_output_size(self.image_size)
        self.in_dim = np.prod(self.in_dim)

        self.v_net = ValueNetwork(self.in_dim, 1).to(self.device)
        self.target_v_net = ValueNetwork(self.in_dim, 1).to(self.device)

        self.q_net1 = SoftQNetwork(self.in_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.in_dim, self.action_dim).to(self.device)

        self.pi_net = PolicyNetwork(self.in_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param)

        # Optimizer initialization
        self.v_optimizer  = optim.Adam(self.v_net.parameters(),  lr=v_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=pi_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def update(self, batch_size):
        if len(self.replay_buffer) <= batch_size:
            print('Replay buffer not large enough to sample, returning models...')
            return self.fe.state_dict(), self.pi_net.state_dict(), False, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = np.stack(states)
        next_states = np.stack(next_states)

        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        # Process images
        features = self.fe(states)
        next_features = self.fe(next_states)
        features = torch.reshape(features, (batch_size, self.in_dim))
        next_features = torch.reshape(next_features, (batch_size, self.in_dim))

        next_actions, next_log_pi = self.pi_net.sample(next_features)
        next_q1 = self.q_net1(next_features, next_actions)
        next_q2 = self.q_net2(next_features, next_actions)
        next_v = self.target_v_net(next_features)

        next_v_target = torch.min(next_q1, next_q2) - next_log_pi
        curr_v = self.v_net.forward(features)
        v_loss = F.mse_loss(curr_v, next_v_target.detach())
        self.logFile.write(' v_loss: %s\n' % v_loss.item())

        # Q loss
        curr_q1 = self.q_net1.forward(features, actions)
        curr_q2 = self.q_net2.forward(features, actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_v
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())
        self.logFile.write('q1_loss: %s\n' % q1_loss.item())
        self.logFile.write('q2_loss: %s\n' % q2_loss.item())

        # update v_net and q_nets
        self.v_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)
        self.v_optimizer.step()

        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_optimizer.step()

        losses = None
        if self.update_step % self.delay_step == 0:
            new_actions, log_pi = self.pi_net.sample(features)
            min_q = torch.min(
                self.q_net1.forward(features, new_actions),
                self.q_net2.forward(features, new_actions)
            )
            pi_loss = (log_pi - min_q).mean()
            self.logFile.write('pi_loss: %s\n\n' % pi_loss.item())
            losses = { 'v_loss': v_loss.item(), 
                       'q_loss': min(q1_loss.item(),q2_loss.item()),
                       'pi_loss': pi_loss.item() }

            self.pi_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            self.pi_optimizer.step()

            for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        else:
            self.logFile.write('\n\n')

        self.update_step += 1
        return self.fe.state_dict(), self.pi_net.state_dict(), True, losses
