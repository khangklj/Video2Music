import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import Normalize
import argparse
import os
chordInvDicPath = "./dataset/vevo_meta/chord_inv.json"
chordRootInvDicPath = "./dataset/vevo_meta/chord_root_inv.json"
chordAttrInvDicPath = "./dataset/vevo_meta/chord_attr_inv.json"

with open(chordInvDicPath) as json_file:
    chordInvDic = json.load(json_file)
with open(chordRootInvDicPath) as json_file:
    chordRootInvDic = json.load(json_file)
with open(chordAttrInvDicPath) as json_file:
    chordAttrInvDic = json.load(json_file)

def compare_conf_matrix(conf_matrix_1, conf_matrix_2, type, output_dir):
    # Prepare directory path
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if type == "CHORD":
        topChordList = []
        with open("./dataset/vevo_meta/top_chord.txt", encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                if len(line_arr) == 3 :
                    chordID = line_arr[1]
                    topChordList.append( int(chordID) )
        topChordList = np.array(topChordList)
        topChordList = topChordList[:10]
        label_names = [ chordInvDic[str(label_id)] for label_id in topChordList ]
        tick_marks = np.arange(len(topChordList))
        title = "Confusion Matrix (Chord)"
    elif type == "CHORD_ROOT":
        chordRootList = np.arange(1, 13)
        label_names = [ chordRootInvDic[str(label_id)] for label_id in chordRootList ]
        tick_marks = np.arange(len(chordRootList))
        title = "Confusion Matrix (Chord root)"
    elif type == "CHORD_ATTR":
        chordAttrList = np.arange(1, 14)        
        label_names = [ chordAttrInvDic[str(label_id)] for label_id in chordAttrList ]
        tick_marks = np.arange(len(chordAttrList))
        title = "Confusion Matrix (Chord quality)"
    else:
        raise Exception("Invalid type")
  
    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    plt.subplots_adjust(right = 0.87)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.75])
    cmap = plt.cm.Blues

    for i, conf_matrix in enumerate([conf_matrix_1, conf_matrix_2]):
        ax[i].imshow(conf_matrix, cmap=cmap)
        ax[i].set_title(title, fontsize=15)
        ax[i].set_xlabel("Predicted label", fontsize=15)
        ax[i].set_ylabel("True label", fontsize=15)        
        ax[i].set_xticks(tick_marks, label_names, rotation=45, fontsize=12, fontweight='bold')
        ax[i].set_yticks(tick_marks, label_names, fontsize=12, fontweight='bold')
        thresh = conf_matrix.max() / 2.0
        for m in range(conf_matrix.shape[0]):
            for n in range(conf_matrix.shape[1]):
                value = conf_matrix[m, n]
                
                # Determine the font size based on the number of digits
                if value >= 1000:  # 4 digits or more
                    cell_fontsize = 12  # Smaller font size for 4-digit numbers
                elif value >= 100:  # 3 digits
                    cell_fontsize = 15  # Smaller font size for 3-digit numbers
                else:
                    cell_fontsize = 17  # Default font size for less than 2 digits                
                ax[i].text(n, m, format(conf_matrix[m, n], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[m, n] > thresh else "black", fontsize=cell_fontsize)
    norm = Normalize(vmin = min(np.min(conf_matrix_1), np.min(conf_matrix_2)), vmax = max(np.max(conf_matrix_1), np.max(conf_matrix_2)))            
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm = norm, cmap = cmap), cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)  # Change font size of color bar labels
    plt.savefig(output_dir + f"{type}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir_path_1", type=str, default=None, help="Directory path to confusion matrix 1")
    parser.add_argument("-dir_path_2", type=str, default=None, help="Directory path to confusion matrix 2")
    parser.add_argument("-output_dir", type=str, default='./output/confusion_matrix/', help="Output directory")
    args = parser.parse_known_args()[0]
    dir_path_1 = args.dir_path_1
    dir_path_2 = args.dir_path_2
    output_dir = args.output_dir

    with open(dir_path_1 + 'chord.npy', 'rb') as f:
        conf_matrix_1 = np.load(f, allow_pickle=True)
    with open(dir_path_1 + 'chord_root.npy', 'rb') as f:
        conf_matrix_root_1 = np.load(f, allow_pickle=True)
    with open(dir_path_1 + 'chord_attr.npy', 'rb') as f:
        conf_matrix_attr_1 = np.load(f, allow_pickle=True)

    with open(dir_path_2 + 'chord.npy', 'rb') as f:
        conf_matrix_2 = np.load(f, allow_pickle=True)
    with open(dir_path_2 + 'chord_root.npy', 'rb') as f:
        conf_matrix_root_2 = np.load(f, allow_pickle=True)
    with open(dir_path_2 + 'chord_attr.npy', 'rb') as f:
        conf_matrix_attr_2 = np.load(f, allow_pickle=True)

    compare_conf_matrix(conf_matrix_1, conf_matrix_2, type="CHORD", output_dir=output_dir)
    print("================")
    compare_conf_matrix(conf_matrix_root_1, conf_matrix_root_2, type="CHORD_ROOT", output_dir=output_dir)
    print("================")
    compare_conf_matrix(conf_matrix_attr_1, conf_matrix_attr_2, type="CHORD_ATTR", output_dir=output_dir)