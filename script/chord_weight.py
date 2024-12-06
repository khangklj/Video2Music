import numpy as np
from utilities.argument_funcs import parse_train_args
from utilities.constants import *
from dataset.vevo_dataset import create_vevo_datasets
import matplotlib.pyplot as plt

args = parse_train_args()[0]
train_dataset, val_dataset, test_dataset = create_vevo_datasets(
    dataset_root = "./dataset/", 
    max_seq_chord = args.max_sequence_chord, 
    max_seq_video = args.max_sequence_video, 
    vis_models = args.vis_models,
    motion_type = args.motion_type,
    emo_model = args.emo_model, 
    split_ver = SPLIT_VER, 
    random_seq = True, 
    is_video = args.is_video,
    augmentation = args.augmentation)

chord_count = np.array([1 for _ in range(CHORD_SIZE)]) # Including PAD and EOS

for data in train_dataset:
    for item in data['chord']:
        chord_count[item] += 1

for data in val_dataset:
    for item in data['chord']:
        chord_count[item] += 1

for data in test_dataset:
    for item in data['chord']:
        chord_count[item] += 1

print(chord_count)

# Generate indices for the data
indices = range(len(chord_count))

# # Plot the bar chart
# plt.figure(figsize=(20, 6))
# plt.bar(indices, chord_count, color='skyblue', edgecolor='black')

# # Labeling
# plt.xlabel('Index', fontsize=14)
# plt.ylabel('Value', fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Show the plot
# plt.tight_layout()
# plt.show()

chord_weight = 1.0 / chord_count
print(chord_weight)

# Plot the bar chart
plt.figure(figsize=(20, 6))
plt.bar(indices, chord_weight, color='skyblue', edgecolor='black')

# Labeling
plt.xlabel('Index', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()