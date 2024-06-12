import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.vevo_dataset import create_vevo_datasets

from model.music_transformer import MusicTransformer
from model.video_music_transformer import VideoMusicTransformer

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model_vevo import eval_model
from dataset.vevo_dataset import compute_vevo_accuracy, compute_vevo_correspondence, compute_hits_k, compute_hits_k_root_attr, compute_vevo_accuracy_root_attr, compute_vevo_correspondence_root_attr
import logging
import os
import sys
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

VIS_MODELS_ARR = [
    "2d/clip_l14p"
]

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler('log/log_eval2.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# main
def main( vm = "", isPrintArgs = True):
    args = parse_eval_args()

    if isPrintArgs:
        print_eval_args(args)

    if vm != "":
        args.vis_models = vm
        
    if args.is_video:
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
        args.model_weights = "./saved_models/" + version + "/best_loss_weights.pickle"
    else:
        vis_abbr_path = "no_video"
        args.model_weights = "./saved_models/" + version + "/best_loss_weights.pickle"
        
    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    
    _, _, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        emo_model = args.emo_model, 
        split_ver = SPLIT_VER, 
        random_seq = True, 
        is_video = args.is_video)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    total_vf_dim = 0
    if args.is_video:
        for vf in test_dataset[0]["semanticList"]:
            total_vf_dim += vf.shape[1]
        total_vf_dim += 1 # Scene_offset
        total_vf_dim += 1 # Motion
        
        # Emotion
        if args.emo_model.startswith("6c"):
            total_vf_dim += 6
        else:
            total_vf_dim += 5
    
    if args.music_gen_version == None:
        if args.is_video:
            model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, 
                        rpr=args.rpr).to(get_device())
        else:
            model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_chord=args.max_sequence_chord, 
                        rpr=args.rpr).to(get_device())
    elif args.music_gen_version == 1:
        model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, 
                        rpr=args.rpr, version=1).to(get_device())
        
    model.load_state_dict(torch.load(args.model_weights))

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    eval_loss_emotion_func = nn.BCEWithLogitsLoss()

    logging.info( f"VIS MODEL: {args.vis_models}" )
    logging.info("Evaluating:")
    model.eval()

    eval_metric_dict = eval_model(model, test_loader, 
                                eval_loss_func, eval_loss_emotion_func,
                                isVideo= args.is_video, isGenConfusionMatrix=True)
        
    eval_total_loss = eval_metric_dict["avg_total_loss"]
    eval_loss_chord = eval_metric_dict["avg_loss_chord"]
    eval_loss_emotion = eval_metric_dict["avg_loss_emotion"]
    eval_h1 = eval_metric_dict["avg_h1"]
    eval_h3 = eval_metric_dict["avg_h3"]
    eval_h5 = eval_metric_dict["avg_h5"]

    logging.info(f"Avg test loss (total): {eval_total_loss:.4f}" )
    logging.info(f"Avg test loss (chord): {eval_loss_chord:.4f}" )
    logging.info(f"Avg test loss (emotion): {eval_loss_emotion:.4f}" )
    logging.info(f"Avg test h1: {eval_h1:.4f}")
    logging.info(f"Avg test h3: {eval_h3:.4f}")
    logging.info(f"Avg test h5: {eval_h5:.4f}")

'''def eval_model(model, dataloader, 
               eval_loss_func, eval_loss_emotion_func,
               isVideo = True, isGenConfusionMatrix=False):
    model.eval()
    avg_acc     = -1
    avg_cor     = -1
    avg_acc_cor = -1

    avg_h1 = -1
    avg_h3 = -1
    avg_h5 = -1
    
    avg_loss_chord    = -1
    avg_loss_emotion    = -1
    avg_total_loss    = -1

    true_labels = []
    true_root_labels = []
    true_attr_labels = []
    
    pred_labels = []
    pred_root_labels = []
    pred_attr_labels = []
    
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        n_test_cor = 0 

        sum_loss_chord   = 0.0
        sum_loss_emotion  = 0.0
        sum_total_loss   = 0.0

        sum_acc    = 0.0
        sum_cor = 0.0

        sum_h1 = 0.0
        sum_h3 = 0.0
        sum_h5 = 0.0
        
        for batch in dataloader:
            x   = batch["x"].to(get_device())
            tgt = batch["tgt"].to(get_device())
            x_root   = batch["x_root"].to(get_device())
            tgt_root = batch["tgt_root"].to(get_device())
            x_attr   = batch["x_attr"].to(get_device())
            tgt_attr = batch["tgt_attr"].to(get_device())
            tgt_emotion = batch["tgt_emotion"].to(get_device())
            tgt_emotion_prob = batch["tgt_emotion_prob"].to(get_device())
            
            feature_semantic_list = [] 
            for feature_semantic in batch["semanticList"]:
                feature_semantic_list.append( feature_semantic.to(get_device()) )
            
            feature_key = batch["key"].to(get_device())
            feature_scene_offset = batch["scene_offset"].to(get_device())
            feature_motion = batch["motion"].to(get_device())
            feature_emotion = batch["emotion"].to(get_device())

            if isVideo:
                if IS_SEPERATED:
                    y_root, y_attr = model(x,
                            x_root,
                            x_attr,
                            feature_semantic_list, 
                            feature_key, 
                            feature_scene_offset,
                            feature_motion,
                            feature_emotion)

                    sum_acc += float(compute_vevo_accuracy_root_attr(y_root, y_attr, tgt))
                    cor = float(compute_vevo_correspondence_root_attr(y_root, y_attr, tgt, tgt_emotion, tgt_emotion_prob, EMOTION_THRESHOLD))
                    if cor >= 0 :
                        n_test_cor +=1
                        sum_cor += cor

                    sum_h1 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,1))
                    sum_h3 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,3))
                    sum_h5 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,5))
                    
                    y_root   = y_root.reshape(y_root.shape[0] * y_root.shape[1], -1)
                    y_attr   = y_attr.reshape(y_attr.shape[0] * y_attr.shape[1], -1)
                    
                    tgt_root = tgt_root.flatten()
                    tgt_attr = tgt_attr.flatten()
                    tgt_emotion = tgt_emotion.squeeze()

                    loss_chord_root = eval_loss_func.forward(y_root, tgt_root)
                    loss_chord_attr = eval_loss_func.forward(y_attr, tgt_attr)
                    loss_chord = loss_chord_root + loss_chord_attr

                    first_14 = tgt_emotion[:, :14]
                    last_2 = tgt_emotion[:, -2:]
                    tgt_emotion_attr = torch.cat((first_14, last_2), dim=1)

                    loss_emotion = eval_loss_emotion_func.forward(y_attr, tgt_emotion_attr)
                    total_loss = LOSS_LAMBDA * loss_chord + (1-LOSS_LAMBDA) * loss_emotion

                    sum_loss_chord += float(loss_chord)
                    sum_loss_emotion += float(loss_emotion)
                    sum_total_loss += float(total_loss)
                else:
                    y= model(x,
                            x_root,
                            x_attr,
                            feature_semantic_list, 
                            feature_key, 
                            feature_scene_offset,
                            feature_motion,
                            feature_emotion)
                    
                    sum_acc += float(compute_vevo_accuracy(y, tgt ))
                    cor = float(compute_vevo_correspondence(y, tgt, tgt_emotion, tgt_emotion_prob, EMOTION_THRESHOLD))
                    if cor >= 0 :
                        n_test_cor +=1
                        sum_cor += cor

                    sum_h1 += float(compute_hits_k(y, tgt,1))
                    sum_h3 += float(compute_hits_k(y, tgt,3))
                    sum_h5 += float(compute_hits_k(y, tgt,5))
                    
                    y   = y.reshape(y.shape[0] * y.shape[1], -1)

                    tgt = tgt.flatten()
                    tgt_root = tgt_root.flatten()
                    tgt_attr = tgt_attr.flatten()
                    
                    tgt_emotion = tgt_emotion.squeeze()

                    loss_chord = eval_loss_func.forward(y, tgt)
                    loss_emotion = eval_loss_emotion_func.forward(y, tgt_emotion)
                    total_loss = LOSS_LAMBDA * loss_chord + (1-LOSS_LAMBDA) * loss_emotion

                    sum_loss_chord += float(loss_chord)
                    sum_loss_emotion += float(loss_emotion)
                    sum_total_loss += float(total_loss)

                    if isGenConfusionMatrix:
                        pred = y.argmax(dim=1).detach().cpu().numpy()
                        pred_root = []
                        pred_attr = []

                        for i in pred:
                            if i == 0:
                                pred_root.append(0)
                                pred_attr.append(0)
                            elif i == 157:
                                pred_root.append(CHORD_ROOT_END)
                                pred_attr.append(CHORD_ATTR_END)
                            elif i == 158:
                                pred_root.append(CHORD_ROOT_PAD)
                                pred_attr.append(CHORD_ATTR_PAD)
                            else:
                                rootindex =  int( (i-1)/13 ) + 1
                                attrindex =  (i-1)%13 + 1
                                pred_root.append(rootindex)
                                pred_attr.append(attrindex)
                        
                        pred_root = np.array(pred_root)
                        pred_attr = np.array(pred_attr)

                        true = tgt.detach().cpu().numpy()
                        true_root = tgt_root.detach().cpu().numpy()
                        true_attr = tgt_attr.detach().cpu().numpy()
                        
                        pred_labels.extend(pred)
                        pred_root_labels.extend(pred_root)
                        pred_attr_labels.extend(pred_attr)
                        
                        true_labels.extend(true)
                        true_root_labels.extend(true_root)
                        true_attr_labels.extend(true_attr)
            else:
                if IS_SEPERATED:
                    y_root, y_attr  = model(x,
                        x_root,
                        x_attr,
                        feature_key)

                    sum_acc += float(compute_vevo_accuracy_root_attr(y_root, y_attr, tgt))
                    cor = float(compute_vevo_correspondence_root_attr(y_root, y_attr, tgt, tgt_emotion, tgt_emotion_prob, EMOTION_THRESHOLD))
                    if cor >= 0 :
                        n_test_cor +=1
                        sum_cor += cor

                    sum_h1 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,1))
                    sum_h3 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,3))
                    sum_h5 += float(compute_hits_k_root_attr(y_root, y_attr, tgt,5))
                    
                    y_root   = y_root.reshape(y_root.shape[0] * y_root.shape[1], -1)
                    y_attr   = y_attr.reshape(y_attr.shape[0] * y_attr.shape[1], -1)
                    
                    tgt_root = tgt_root.flatten()
                    tgt_attr = tgt_attr.flatten()
                    tgt_emotion = tgt_emotion.squeeze()

                    loss_chord_root = eval_loss_func.forward(y_root, tgt_root)
                    loss_chord_attr = eval_loss_func.forward(y_attr, tgt_attr)
                    loss_chord = loss_chord_root + loss_chord_attr

                    first_14 = tgt_emotion[:, :14]
                    last_2 = tgt_emotion[:, -2:]
                    tgt_emotion_attr = torch.cat((first_14, last_2), dim=1)
                    loss_emotion = eval_loss_emotion_func.forward(y_attr, tgt_emotion_attr)
                    
                    total_loss = LOSS_LAMBDA * loss_chord + (1-LOSS_LAMBDA) * loss_emotion

                    sum_loss_chord += float(loss_chord)
                    sum_loss_emotion += float(loss_emotion)
                    sum_total_loss += float(total_loss)
                else:
                    # use MusicTransformer no sep
                    y = model(x,
                            x_root,
                            x_attr,
                            feature_key)
                    
                    sum_acc += float(compute_vevo_accuracy(y, tgt ))
                    cor = float(compute_vevo_correspondence(y, tgt, tgt_emotion, tgt_emotion_prob, EMOTION_THRESHOLD))
                    
                    if cor >= 0 :
                        n_test_cor +=1
                        sum_cor += cor

                    sum_h1 += float(compute_hits_k(y, tgt,1))
                    sum_h3 += float(compute_hits_k(y, tgt,3))
                    sum_h5 += float(compute_hits_k(y, tgt,5))

                    tgt_emotion = tgt_emotion.squeeze()
                    
                    y   = y.reshape(y.shape[0] * y.shape[1], -1)
                    tgt = tgt.flatten()
                    loss_chord = eval_loss_func.forward(y, tgt)
                    loss_emotion = eval_loss_emotion_func.forward(y, tgt_emotion)
                    total_loss = loss_chord

                    sum_loss_chord += float(loss_chord)
                    sum_loss_emotion += float(loss_emotion)
                    sum_total_loss += float(total_loss)

        avg_loss_chord    = sum_loss_chord / n_test
        avg_loss_emotion    = sum_loss_emotion / n_test
        avg_total_loss    = sum_total_loss / n_test

        avg_acc     = sum_acc / n_test
        avg_cor     = sum_cor / n_test_cor
        
        avg_h1     = sum_h1 / n_test
        avg_h3     = sum_h3 / n_test
        avg_h5     = sum_h5 / n_test
        
        avg_acc_cor = (avg_acc + avg_cor)/ 2.0

    if isGenConfusionMatrix:
        chordInvDicPath = "./dataset/vevo_meta/chord_inv.json"
        chordRootInvDicPath = "./dataset/vevo_meta/chord_root_inv.json"
        chordAttrInvDicPath = "./dataset/vevo_meta/chord_attr_inv.json"
        
        with open(chordInvDicPath) as json_file:
            chordInvDic = json.load(json_file)
        with open(chordRootInvDicPath) as json_file:
            chordRootInvDic = json.load(json_file)
        with open(chordAttrInvDicPath) as json_file:
            chordAttrInvDic = json.load(json_file)

        # Confusion matrix (CHORD)
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
        mask = np.isin(true_labels, topChordList)
        true_labels = np.array(true_labels)[mask]
        pred_labels = np.array(pred_labels)[mask]

        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=topChordList)
        label_names = [ chordInvDic[str(label_id)] for label_id in topChordList ]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(topChordList))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

        # Confusion matrix (CHORD ROOT)        
        chordRootList = np.arange(1, 13)
        conf_matrix = confusion_matrix(true_root_labels, pred_root_labels, labels= chordRootList )
        
        label_names = [ chordRootInvDic[str(label_id)] for label_id in chordRootList ]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Chord root)")
        plt.colorbar()
        tick_marks = np.arange(len(chordRootList))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("confusion_matrix_root.png")
        plt.show()

        # Confusion matrix (CHORD ATTR)
        chordAttrList = np.arange(1, 14)
        conf_matrix = confusion_matrix(true_attr_labels, pred_attr_labels, labels= chordAttrList )
        
        label_names = [ chordAttrInvDic[str(label_id)] for label_id in chordAttrList ]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Chord quality)")
        plt.colorbar()
        tick_marks = np.arange(len(chordAttrList))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("confusion_matrix_quality.png")
        plt.show()

    return { "avg_total_loss" : avg_total_loss, 
             "avg_loss_chord" : avg_loss_chord, 
             "avg_loss_emotion": avg_loss_emotion, 
             "avg_acc" : avg_acc, 
             "avg_cor" : avg_cor, 
             "avg_acc_cor" : avg_acc_cor, 
             "avg_h1" : avg_h1, 
             "avg_h3" : avg_h3,
             "avg_h5" : avg_h5 }'''

if __name__ == "__main__":
    if len(VIS_MODELS_ARR) != 0 :
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()


