import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Global variable to store indices
indices = []

# Global 2D array to count selections for each emotion-expert pair
counts = [[0 for _ in range(6)] for _ in range(6)]  # 6 emotions and 6 experts

def get_highest_emotion_indices(emotion_feature):
    global indices  # Declare that we are using the global variable
    indices = []  # Reset the global indices list

    # Remove the batch dimension for easier processing
    emotion_feature = emotion_feature.squeeze(0)

    for i in range(emotion_feature.size(0)):  # Iterate over the sequence length
        # Get the maximum probability and its index for each token
        max_prob, max_index = torch.max(emotion_feature[i], dim=0)

        # Check if all probabilities are zero
        if torch.all(emotion_feature[i] == 0):
            indices.append(-1)
        else:
            indices.append(max_index.item())  # Convert to Python integer
    return indices

def update_expert_counts(selected_expert):
    global indices  # Use the global indices
    global counts  # Use the global counts

    # Remove unnecessary dimensions for easier processing
    selected_expert = selected_expert.squeeze(1)   
    # Iterate over the global indices
    for i in range(selected_expert.size(0)):
      index = indices[i]
      if index != -1:  # Skip if index is -1
        # Get the selected experts for the current index
        experts = selected_expert[i]  # This will be a tensor of shape (2)

        # Increment the count for each selected expert
        for expert in experts:
          counts[index][expert.item()] += 1  # Increment count

def save_and_plot(filename='log/experts_emotion_count.json', plotname='log/experts_emotion_count_plot.png'):
  global counts
  # Create the directory if it does not exist
  directory = os.path.dirname(filename)
  if directory and not os.path.exists(directory):
      os.makedirs(directory) 
   # Save to a JSON file
  with open(filename, 'w') as file:
    json.dump(counts, file)
  
  with open(filename, 'r') as file:
    loaded_data = json.load(file)
    # Convert to a NumPy array for easier manipulation
    data_array = np.array(loaded_data)

    # Calculate the total contributions for each emotion
    totals = data_array.sum(axis=1, keepdims=True)

    # Calculate percentages
    percentage_array = (data_array *1.0 / totals) 

    emotion_list = ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]

    # Create a heatmap for the percentage contributions
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentage_array, annot=True, fmt=".4f", cmap='YlGnBu', 
                 xticklabels=[f'Expert {i + 1}' for i in range(percentage_array.shape[1])],
                 yticklabels=[f'{emotion}' for emotion in emotion_list])


    # Adding labels and title
    plt.title('Contribution of Each Expert for Each Emotion')
    plt.xlabel('Experts')
    plt.ylabel('Emotions')

    plt.tight_layout()

    # Save the plot
    plt.savefig(plotname, bbox_inches='tight')