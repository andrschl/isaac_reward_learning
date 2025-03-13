from __future__ import annotations

import torch
import h5py
from abc import ABC, abstractmethod
from einops import rearrange

from isaac_reward_learning.modules import RewardModel

class RolloutStorage(ABC):
    class Trajectory:
        def __init__(self):
            # When used for expert demos these will be torch.Tensors;
            self.observations = None
            self.actions = None
            self.dones = None
            self.time_ids = None
            self.traj_id = 0

        def clear(self):
            self.__init__()

        def __len__(self):
            if self.observations is None:
                return 0
            if isinstance(self.observations, torch.Tensor):
                return self.observations.size(0)
            return len(self.observations)
        
    class Transition():
        def __init__(self):
            self.observations = None
            self.actions = None
            self.dones = None

    def __init__(self, obs_shape, actions_shape, gamma, max_traj_length=128, subsampling=1, device="cpu"):
        self.device = device
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape
        self.gamma = gamma
        self.max_traj_length = max_traj_length
        self.subsampling = subsampling

        # Pre-allocate tensors for all demonstrations.
        self.traj_ids = None
        self.time_ids = None
        self.observations = None
        self.actions = None
        self.dones = None

    def count_trajectories(self) -> int:
        unique_ids = torch.unique(self.traj_ids)
        return unique_ids[unique_ids != -1].shape[0]    # Don't count -1 which represents padding

    def get_expected_reward(self, reward: RewardModel, obs_batch=None, actions_batch=None, time_ids=None, num_trajs_in_batch=1):
        """
        Compute expected returns.
        """
        rewards = reward.get_reward(obs_batch, actions_batch)  # shape: (B, T)

        discounts = self.gamma ** time_ids
        discounted_rewards = rewards * discounts
        return discounted_rewards.sum() / num_trajs_in_batch

    @abstractmethod
    def mini_batch_generator(self, mini_batch_size, num_epochs=8):
        """
        Generate mini-batches for training.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


class ExpertRolloutStorage(RolloutStorage):
    """
    Rollout storage for expert demos with a neural network reward model. This class loads demos from a HDF5 file
    and stores them in tensors of shape (T, ...), where T is the total number of observations across all demos (num_envs=1).
    """
    def __init__(self, expert_data_path, obs_shape, actions_shape, gamma,
                 max_num_demos=int(1e4), max_traj_length=128, subsampling=1, device="cpu"):
        super().__init__(obs_shape, actions_shape, gamma, max_traj_length, subsampling, device=device)

        self.expert_data_path = expert_data_path

        # Open the HDF5 file and determine total number of trajectories and steps.
        with h5py.File(self.expert_data_path, 'r') as f:
            self.num_demos = min(len(f['data'].keys()), max_num_demos)
            total_length = 0
            for i, demo_key in enumerate(f['data'].keys()):
                if i >= self.num_demos:
                    break
                demo_group = f['data'][demo_key]
                total_length += min(len(demo_group['actions']), max_traj_length)
            self.num_obs = total_length

            print(f"Number of trajectories: {self.num_demos}")
            print(f"Total length of the expert buffer: {total_length}")

            # Pre-allocate tensors for all demonstrations. Initialize empty ids with -1.
            self.traj_ids = - torch.ones(self.num_obs, device=self.device)
            self.time_ids = - torch.ones(self.num_obs, device=self.device)
            self.observations = torch.zeros(self.num_obs, *self.obs_shape, device=self.device)
            self.actions = torch.zeros(self.num_obs, *self.actions_shape, device=self.device)
            self.dones = torch.zeros(self.num_obs, 1, device=self.device).byte()

            # Iterate over demos in the file.
            self._traj_count = 0
            self._step = 0
            for i, demo_key in enumerate(f['data'].keys()):
                if i >= self.num_demos:
                    break
                demo_group = f['data'][demo_key]
                trajectory = self.Trajectory()
                trajectory.actions = torch.tensor(demo_group['actions'][:max_traj_length])
                trajectory.dones = torch.tensor(demo_group['dones'][:max_traj_length])
                trajectory.observations = torch.tensor(demo_group['obs']['full_obs'][:max_traj_length])
                trajectory.traj_id = i
                trajectory.time_ids = torch.arange(len(trajectory), device=self.device)               
                self.add_trajectory(trajectory)

    def add_trajectory(self, trajectory: RolloutStorage.Trajectory):
        """
        Add a single trajectory to the preallocated buffer.
        """
        if self._step + len(trajectory) > self.num_obs:
            raise AssertionError("Rollout buffer overflow")
        self.traj_ids[self._step:self._step+len(trajectory)] = trajectory.traj_id
        self.time_ids[self._step:self._step+len(trajectory)] = trajectory.time_ids
        self.observations[self._step:self._step+len(trajectory)] = trajectory.observations
        self.actions[self._step:self._step+len(trajectory)] = trajectory.actions
        self.dones[self._step:self._step+len(trajectory)] = trajectory.dones.unsqueeze(1)
        self._traj_count += 1
        self._step += len(trajectory)

    def mini_batch_generator(self, mini_batch_size, num_epochs=1):
        """
        Generate mini-batches from the expert buffer. This code identifies the start/end
        indices of each trajectory from the dones flag and then returns padded mini-batches.
        """

        num_trajectories = self.count_trajectories()
        if num_trajectories == 0:
            raise AssertionError("No trajectories stored in the buffer")
        num_mini_batches = (num_trajectories + mini_batch_size - 1) // mini_batch_size

        for epoch in range(num_epochs):
            indices = torch.randperm(num_trajectories, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_ids = indices[start:end]
                mask = torch.isin(self.traj_ids, batch_ids) & (self.time_ids % self.subsampling == 0)
                num_samples_in_batch = len(batch_ids)

                yield self.observations[mask, ...], self.actions[mask, ...], self.time_ids[mask], num_samples_in_batch

class ImitatorRolloutStorage(RolloutStorage):
    def __init__(self, num_envs, obs_shape, actions_shape, gamma, max_traj_length=128, subsampling=1, device="cpu"):
        super().__init__(obs_shape, actions_shape, gamma, max_traj_length, subsampling, device=device)

        # Pre-allocate tensors for all demonstrations. Initialize empty ids with -1.
        self.traj_ids = - torch.ones(self.max_traj_length, num_envs, device=self.device)
        self.time_ids = - torch.ones(self.max_traj_length, num_envs, device=self.device)
        self.observations = torch.zeros(self.max_traj_length, num_envs, *self.obs_shape, device=self.device)
        self.actions = torch.zeros(self.max_traj_length, num_envs, *self.actions_shape, device=self.device)
        self.dones = torch.zeros(self.max_traj_length, num_envs, device=self.device).byte()

        self._traj_ids = torch.arange(num_envs, device=self.device)
        self._traj_count = num_envs
        self._time_ids = torch.zeros(num_envs, device=self.device)
        self._step = 0

    def clear(self):
        num_envs = len(self.traj_ids[0])
        self._traj_ids = torch.arange(num_envs, device=self.device)
        self._traj_count = num_envs
        self._time_ids = torch.zeros(num_envs, device=self.device)
        self._step = 0

    def add_transition(self, transition: RolloutStorage.Transition):
        """
        Add a single transition to the current trajectory.
        """
        if self._step >= self.max_traj_length:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self._step] = transition.observations
        self.actions[self._step] = transition.actions
        self.dones[self._step] = transition.dones
        self.traj_ids[self._step] = self._traj_ids
        self.time_ids[self._step] = self._time_ids

        # Update trajectory and time ids.
        self._time_ids += 1
        for i in torch.nonzero(transition.dones, as_tuple=True)[0].tolist():
            self._traj_ids[i] = self._traj_count
            self._traj_count += 1
            self._time_ids[i] = 0
        self._step += 1

    def mini_batch_generator(self, mini_batch_size, num_epochs=1):

        num_trajectories = self.count_trajectories()
        if num_trajectories == 0:
            raise AssertionError("No trajectories stored in the buffer")
        num_mini_batches = (num_trajectories + mini_batch_size - 1) // mini_batch_size

        for epoch in range(num_epochs):
            indices = torch.randperm(num_trajectories, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_ids = indices[start:end]
                mask = torch.isin(self.traj_ids, batch_ids) & (self.time_ids % self.subsampling == 0)
                num_samples_in_batch = len(batch_ids)

                yield self.observations[mask, ...], self.actions[mask, ...], self.time_ids[mask], num_samples_in_batch