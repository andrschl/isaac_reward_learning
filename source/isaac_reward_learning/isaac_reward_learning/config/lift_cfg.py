from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaac_reward_learning.config import RewardCfg, IrlAlgorithmCfg, IrlRunnerCfg

@configclass
class LiftCubeIrlRunnerCfg(IrlRunnerCfg):
    expert_data_path: str = ""
    num_steps_per_env_rl = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    logger = "wandb"
    actor_critic = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    reward = RewardCfg(
        reward_hidden_dims=[256, 128, 64],
        activation="elu",
        reward_is_linear=False,
        reward_features=None,
        num_reward_features=None,
    )
    rl_algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    irl_algorithm = IrlAlgorithmCfg(
        max_rollout_length=128,
        batch_size=1000,
        num_learning_epochs=5,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        reward_loss_coef=1e-4,
    )
