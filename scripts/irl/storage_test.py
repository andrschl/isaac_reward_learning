from isaac_reward_learning.storage import ExpertRolloutStorage
import torch
import numpy as np

file_path = 'scripts/irl/demos.hdf5'
storage = ExpertRolloutStorage(file_path, (4,), (1,), 0.99, max_traj_length=128, subsampling=1, device="cpu")


def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        motions = list(storage.mini_batch_generator(mini_batch_size=1, num_epochs=1))
        states = [m[0] for m in motions]
        actions = [m[1] for m in motions]
        return states[0][:num_samples].to(self.device), actions[0][:num_samples].to(self.device)



if __name__ == "__main__":
    for e in storage.mini_batch_generator(mini_batch_size=1, num_epochs=1):
        for elem in e:
            if isinstance(elem, torch.Tensor):
                print(elem.shape)
            elif isinstance(elem, np.ndarray):
                print(elem.shape)
            else:
                print(type(elem))
        break

    res = collect_reference_motions(storage, 200)
    print(res[0].shape)
    print(res[1].shape)

