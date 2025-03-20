# isaac_reward_learning
Library for inverse reinforcement learning and RLHF for robotic manipulation.

## Collecting trajectories from existing policy
- `python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless`
- `python scripts/recording/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --num_demos 1000 --num_envs 1000 --headless`
- `python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path 'logs/rsl_rl/franka_lift/demos/hdf_dataset.hdf5'`
