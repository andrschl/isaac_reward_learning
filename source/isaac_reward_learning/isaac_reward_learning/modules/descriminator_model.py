from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class DiscriminatorModel(nn.Module):

    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims=[256, 256, 256],
        is_linear=False,
        discriminator_features=None,
        num_features=None,
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "DiscriminatorModelc.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        input_dim = num_obs + num_actions

        # Define discriminator model
        if is_linear:
            self.init_linear_model(num_features)
        else:
            self.init_nn_model(input_dim, hidden_dims, activation)
      
        print(f"Discriminator MLP: {self.discriminator}")

    def init_linear_model(self, num_features):
        if self.features is None or num_features is None:
            raise ValueError("If is_linear=True, feature_map and num_features must not be None")
        self.discriminator = nn.Sequential(nn.Linear(num_features, 1, bias=False))

    def init_nn_model(self, input_dim, hidden_dims, activation):
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        discriminator_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[layer_index], 1))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        self.feature_map = lambda observations, actions: torch.cat([observations, actions], dim=-1)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_discrimination(self, observations, actions):
        with torch.no_grad():
            features = self.feature_map(observations, actions)
        return torch.squeeze(self.discriminator(features), dim=-1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
