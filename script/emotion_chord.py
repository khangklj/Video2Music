import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# print(sys.path)

import numpy as np
from dataset.vevo_dataset import create_vevo_datasets
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.constants import *
import json
from tqdm import tqdm

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

args = parse_eval_args()[0]

with open('./dataset/vevo_meta/chord_attr_inv.json', 'r') as file:
    chord_attr_inv_dict = json.load(file)

with open('./dataset/vevo_meta/chord_root_inv.json', 'r') as file:
    chord_root_inv_dict = json.load(file)

with open('./dataset/vevo_meta/chord_inv.json', 'r') as file:
    chord_inv_dict = json.load(file)

# print(chord_attr_inv_dict)
# print(chord_root_inv_dict)
# print(chord_inv_dict)

emotion_list = ['exciting', 'fearful', 'tense', 'sad', 'relaxing', 'neutral']

def calculating_mapping_table(mapping_table, count_table, calculating_dict, dataset, key='chord_attr', option=1):
    for i in tqdm(range(len(dataset))):
        chord_attr = dataset[i][key]
        chord_attr = [calculating_dict[str(c.item())] for c in chord_attr if str(c.item()) in calculating_dict.keys()]

        emotion = dataset[i]['emotion']

        for j in range(len(chord_attr)):
            if chord_attr[j] != 'N':            
                if option == 1:
                    k = np.argmax(emotion[j])
                    mapping_table[chord_attr[j]][k] += 1
                else:
                    mapping_table[chord_attr[j]] += np.array(emotion[j])
                
            count_table[chord_attr[j]] += 1

print('Caclulating emotion-chord mapping table')
dict_id = 2 # Change this

if dict_id == 0:
    calculating_dict = chord_attr_inv_dict
    key = 'chord_attr'
elif dict_id == 1:
    calculating_dict = chord_root_inv_dict
    key = 'chord_root'
else:
    calculating_dict = chord_inv_dict
    key = 'chord'

mapping_table = {attr: np.array([0 for _ in emotion_list], dtype=float) for attr in list(calculating_dict.values())}
count_table = {attr: 0 for attr in list(calculating_dict.values())}

train_dataset, eval_dataset, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        motion_type = args.motion_type,
        emo_model = args.emo_model, 
        split_ver = SPLIT_VER, 
        random_seq = True, 
        is_video = args.is_video)

option = 2
calculating_mapping_table(mapping_table, count_table, calculating_dict, train_dataset, key=key, option=option)
calculating_mapping_table(mapping_table, count_table, calculating_dict, eval_dataset, key=key, option=option)
calculating_mapping_table(mapping_table, count_table, calculating_dict, test_dataset, key=key, option=option)

mapping_table = {attr: mapping_table[attr] / count_table[attr] for attr in list(calculating_dict.values())[:-2]}
mapping_table.pop('N')

class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self.compact_lists = kwargs.pop("compact_lists", False)
        super().__init__(*args, **kwargs)

    def encode(self, o):
        if isinstance(o, dict):
            return "{\n" + ",\n".join(
                f'    "{k}": {self._encode_value(v)}' for k, v in o.items()
            ) + "\n}"
        return super().encode(o)

    def _encode_value(self, v):
        if self.compact_lists and (isinstance(v, np.ndarray) or isinstance(v, list)):
            return "\t[" + ", ".join(json.dumps(round(i, 2)) for i in v) + "]"
        if isinstance(v, dict):
            return "{\n" + ",\n".join(
                f'    "{k}": {self._encode_value(val)}' for k, val in v.items()
            ) + "\n}"
        return json.dumps(v)

with open('./dataset/vevo_meta/chord_count.json', 'w') as file:
    file.write(json.dumps(count_table, indent=4))

with open('./dataset/vevo_meta/emotion_mapping_table.json', 'w') as file:
    file.write(json.dumps(mapping_table, indent=4, cls=CustomJSONEncoder, compact_lists=True))
    
import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))

for chord, probabilities in mapping_table.items():
    plt.plot(emotion_list, probabilities, marker='o', label=chord)

# Add labels and title
plt.xlabel('Emotions')
plt.ylabel('Probability')
plt.title('Chord vs Emotion Probabilities')
plt.legend(title='Chords', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()