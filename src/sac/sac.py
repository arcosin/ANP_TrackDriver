import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from .models import FeatureExtractor, ValueNetwork, SoftQNetwork, PolicyNetwork
from .replay_buffers import BasicBuffer

class SACAgent:
    def __init__(self,
                 action_range, action_dim, gamma, tau, v_lr, q_lr, pi_lr, buffer_maxlen=int(1e6),
                 image_size=(256,256,3), kernel_size=(3,3), conv_channels=4,
                 checkpoint=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_range = action_range
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        if checkpoint == None: checkpoint = {}
        saved_fe = checkpoint.get('fe', None)

        saved_v_net  = checkpoint.get('v_net', None)
        saved_tv_net = checkpoint.get('tv_net', None)
        saved_q1_net = checkpoint.get('q1_net', None)
        saved_q2_net = checkpoint.get('q2_net', None)
        saved_pi_net = checkpoint.get('pi_net', None)

        saved_v_opt  = checkpoint.get('v_opt', None)
        saved_q1_opt = checkpoint.get('q1_opt', None)
        saved_q2_opt = checkpoint.get('q2_opt', None)
        saved_pi_opt = checkpoint.get('pi_opt', None)

        # Network initialization
        self.fe = FeatureExtractor(image_size[2], conv_channels, kernel_size, saved_fe).to(self.device)
        self.in_dim = self.fe.get_output_size(image_size)
        self.in_dim = np.prod(self.in_dim)

        self.v_net = ValueNetwork(self.in_dim, 1, saved_v_net=saved_v_net).to(self.device)
        self.target_v_net = ValueNetwork(self.in_dim, 1, saved_v_net=saved_tv_net).to(self.device)

        self.q_net1 = SoftQNetwork(self.in_dim, self.action_dim, saved_q_net=saved_q1_net).to(self.device)
        self.q_net2 = SoftQNetwork(self.in_dim, self.action_dim, saved_q_net=saved_q2_net).to(self.device)

        self.pi_net = PolicyNetwork(self.in_dim, self.action_dim, saved_pi_net=saved_pi_net).to(self.device)

        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param)

        # Optimizer initialization
        self.v_optimizer  = optim.Adam(self.v_net.parameters(),  lr=v_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=pi_lr)

        if saved_v_opt:  self.v_optimizer.load_state_dict(saved_v_opt)
        if saved_q1_opt: self.q1_optimizer.load_state_dict(saved_q1_opt)
        if saved_q2_opt: self.q2_optimizer.load_state_dict(saved_q2_opt)
        if saved_pi_opt: self.pi_optimizer.load_state_dict(saved_pi_opt)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        input = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        features = self.fe(input)
        features = features.view(-1, self.in_dim)

        mean, log_std = self.policy_net.forward(features)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action):
        action_range = self.action_range
        return action * ((action_range[1] - action_range[0]) / 2 + (action_range[1] + action_range[0]) / 2)

    def update(self, batch_size):
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

        # Q loss
        curr_q1 = self.q_net1.forward(features, actions)
        curr_q2 = self.q_net2.forward(features, actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_v
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

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

        if self.update_step % self.delay_step == 0:
            new_actions, log_pi = self.pi_net.sample(features)
            min_q = torch.min(
                self.q_net1.forward(features, new_actions),
                self.q_net2.forward(features, new_actions)
            )
            pi_loss = (log_pi - min_q).mean()

            self.pi_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            self.pi_optimizer.step()

            for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.update_step += 1
        return self.fe.state_dict(), self.pi_net.state_dict()
