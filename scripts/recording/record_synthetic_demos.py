"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_demos", type=int, default=1000, help="Number of episodes to store in the dataset.")
parser.add_argument("--demo_length", type=int, default=100, help="Maximum length of a demonstration.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=150, help="Length of the recorded video (in steps).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True 

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import isaac_reward_learning
from isaac_reward_learning.utils import RobomimicDataCollector

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.concatenate_terms = True
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.max_episode_length = args_cli.demo_length

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    demo_dir = os.path.join("logs", "demos", agent_cfg.experiment_name)
    demo_dir = os.path.abspath(demo_dir)
    print(f"[INFO] Saving demos in directory: {demo_dir}")
    
    # name of the file to save data
    filename = "demos.hdf5"

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(demo_dir, "videos"),
            "step_trigger": lambda step: step % env.max_episode_length == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during demo collection.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rlrsl rl environment observations as dictionary
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # name of the environment (needed by robomimic)
    task_name = args_cli.task
    
    # create data-collector
    collector_interface = RobomimicDataCollector(task_name, demo_dir, filename, args_cli.num_demos)

    # reset the collector
    collector_interface.reset()

    # reset environment
    obs_dict, _ = env.env.reset()

    # run everything in inference mode
    with torch.inference_mode():
        while not collector_interface.is_stopped():

            # store signals before stepping
            # -- obs
            for key, value in {'full_obs': obs_dict["policy"]}.items():
                collector_interface.add(f"obs/{key}", value)

            # -- action 
            obs,_ = env.get_observations()
            actions = policy(obs)
            collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, _ = env.env.step(actions)
            dones = terminated | truncated

            # store reward and done signals
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()