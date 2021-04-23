import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from math import floor

import sys
from os import path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#from .models import FeatureExtractor, PolicyNetwork
sys.path.append(path.join(path.dirname(__file__), '..'))
from sac import FeatureExtractor, PolicyNetwork

class Agent():
    def __init__(self, image_size, num_actions, conv_channels, kernel_size, action_range, saved_fe_weights=None, saved_pi_weights=None):
        # Image_size a tuple where 0 = height, 1 = width, 2 = num_channels
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        
        self.image_size = image_size
        self.action_range = action_range

        print(f"Agent: Image input shape (ndarray) = {image_size}")

        self.fe = FeatureExtractor(image_size[2], conv_channels, kernel_size, saved_fe_weights)
        
        out_size_info = self.fe.get_output_size(image_size)
        print(f"Agent: Feature Extractor out shape = {out_size_info}")
        
        self.num_linear_inputs = out_size_info[0] * out_size_info[1] * out_size_info[2]
        print(f"Agent: Policy Network input shape = {(1, self.num_linear_inputs)}")

        self.pi = PolicyNetwork(self.num_linear_inputs, num_actions, saved_pi_net = saved_pi_weights)

        self.fe.eval()
        self.pi.eval()

    def get_action(self, state):
        if state.shape != self.image_size:
            print(f"Invalid size, expected shape {self.image_size}, got {state.shape}")
            return None

        # Assume channel is the last dimension, so we permute
        # Unsqueeze for batch size arg
        # Final shape of inp is (1, height, width, channels)
        inp = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)

        features = self.fe(inp)
        features = features.view(-1, self.num_linear_inputs)

        action_tensor, _ = self.pi.sample(features)

        action = action_tensor.detach().squeeze(0).numpy()
        #log_pi = log_pi_tensor.detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action):
        scaled_action = []
        for idx, a in enumerate(action):
            action_range = self.action_range[idx]
            a = a * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0            
            scaled_action.append(a)
        return scaled_action

    def load_weights(self, fe_weights, pi_weights):
        # Expects dictionaries
        self.fe.load_state_dict(fe_weights)
        self.pi.load_state_dict(pi_weights)

        self.fe.eval()
        self.pi.eval()

    def get_weights(self):
        # Both of these are dictionaries
        return self.fe.state_dict(), self.pi.state_dict()
