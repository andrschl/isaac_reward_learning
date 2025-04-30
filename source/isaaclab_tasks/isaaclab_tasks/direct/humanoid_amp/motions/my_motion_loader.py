import numpy as np

file_name = 'source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz'

data = np.load(file_name)

for f in data.files:
    print(f, data[f].shape)

print(data['dof_names'], len(data['dof_names']))
print()
print(data['body_names'], len(data['body_names']))
print()
print(data.files)