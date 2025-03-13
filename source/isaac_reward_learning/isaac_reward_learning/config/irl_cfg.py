from dataclasses import MISSING
from collections.abc import Callable
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class RewardCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "NeuralNetwork"
    """The policy class name. Default is NeuralNetwork."""

    reward_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the reward network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    reward_is_linear: bool = MISSING
    """Whether the reward function is linear."""

    reward_features: Callable = MISSING
    """The features used for the reward function if linear."""

    num_reward_features: int = MISSING
    """The number of features used for the reward function if linear."""


@configclass
class IrlAlgorithmCfg:
    """Configuration for the IRL algorithm."""

    expert_data_path: str = ""
    """The path to the expert data."""

    max_rollout_length: int = MISSING
    """The rollout length for the reward update."""

    batch_size: int = MISSING
    """The batch size for the reward update."""

    num_learning_epochs: int = 1
    """The number of learning epochs per reward update."""

    weight_decay: float = 1e-6
    """The weight decay for the reward update."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    reward_loss_coef: float = MISSING
    """The coefficient for the reward loss."""

@configclass
class IrlRunnerCfg:
    """Configuration of the runner for irl algorithm."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for torch. Default is cuda:0."""

    expert_data_path: str = MISSING
    """The path to the expert data."""

    num_steps_per_env_rl: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    actor_critic: RslRlPpoActorCriticCfg = MISSING
    """The actor-critic module configuration."""

    reward: RewardCfg = MISSING
    """The reward module configuration."""

    rl_alg: RslRlPpoAlgorithmCfg = MISSING
    """The RL algorithm configuration."""

    irl_algorithm: IrlAlgorithmCfg = MISSING
    """The IRL algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
