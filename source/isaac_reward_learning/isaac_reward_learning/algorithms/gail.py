from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from isaac_reward_learning.modules import DiscriminatorModel
from isaac_reward_learning.storage import ExpertRolloutStorage, ImitatorRolloutStorage

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

class GAIL:
    rl_alg: PPO
    discriminator: DiscriminatorModel

    def __init__(
        self,
        rl_alg,
        discriminator,
        expert_data_path="",
        max_rollout_length=128,
        batch_size=256,
        num_learning_epochs=1,
        weight_decay=1e-6,
        max_grad_norm=1.0,
        discriminator_loss_coef=1.0,
        device="cpu",
    ):
        self.device = device

        # RL components
        self.rl_alg = rl_alg

        # IRL components
        self.discriminator = discriminator.to(self.device)
        self.expert_data_path = expert_data_path
        self.expert_storage = None  # initialized later
        self.imitator_storage = None  # initialized later
        self.transition = RolloutStorage.Transition()
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=rl_alg.learning_rate, weight_decay=weight_decay)
        self.discriminator_criterion = nn.BCEWLoss()

        # IRL parameters
        self.batch_size = batch_size
        self.num_learning_epochs = num_learning_epochs
        self.discriminator_loss_coef = discriminator_loss_coef
        self.max_grad_norm = max_grad_norm
        self.max_rollout_length = max_rollout_length

    def init_expert_storage(self, expert_data_path, obs_shape, actions_shape, **kwargs):
        self.expert_storage = ExpertRolloutStorage(
            expert_data_path, obs_shape, actions_shape, self.rl_alg.gamma, 
            max_traj_length=self.max_rollout_length, 
            device=self.device, **kwargs
        )

    def init_imitator_storage(self, num_envs, obs_shape, action_shape, **kwargs):
        self.imitator_storage = ImitatorRolloutStorage(
            num_envs, obs_shape, action_shape, self.rl_alg.gamma, 
            max_traj_length=self.max_rollout_length,
            device=self.device, **kwargs
        )

    def eval_mode(self):
        self.rl_alg.actor_critic.eval()
        self.discriminator.eval()

    def train_mode(self):
        self.rl_alg.actor_critic.train()
        self.discriminator.train()

    def act(self, obs):
        # Compute the actions and values
        self.transition.actions = self.rl_alg.actor_critic.act(obs).detach()
        self.transition.observations = obs
        return self.transition.actions
        

    def process_env_step(self, dones):
        self.transition.dones = dones
        self.imitator_storage.add_transition(self.transition)

    def discriminatot_update(self):

        mean_discriminator_loss = 0
        expert_generator = self.expert_storage.mini_batch_generator(self.batch_size, 10**100)    # 10**100 is a large number of epochs
        imitator_generator = self.imitator_storage.mini_batch_generator(self.batch_size, self.num_learning_epochs)
        
        for i, (
            obs_batch,
            actions_batch,
            time_ids,
            num_imitator_samples
        ) in enumerate(imitator_generator):
           
            # GAIL loss
            expert_obs_batch, expert_actions_batch, expert_time_ids, num_expert_samples = next(expert_generator)
            
            g_o = self.discriminator(obs_batch, actions_batch)
            e_o = self.discriminator(expert_obs_batch, expert_actions_batch)
            
            g_o_labels = torch.ones_like(g_o)
            e_o_labels = torch.zeros_like(e_o)
            discriminator_loss = self.discriminator_criterion(g_o, g_o_labels) + self.discriminator_criterion(e_o, e_o_labels)

            # Gradient step
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_gradient_norm = nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()

            mean_discriminator_loss += discriminator_loss.item()

        num_updates = i + 1
        mean_discriminator_loss /= num_updates

        return mean_discriminator_loss, discriminator_gradient_norm
