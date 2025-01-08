import gradio as gr
from pathlib import Path

import torch
import torchvision.models as models
import shutil
import os
import subprocess
import cv2
import math
import clip
import joblib
import numpy as np
from PIL import Image
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images
from utilities.constants import *
from utilities.chord_to_midi import *

from model.video_music_transformer import *
from model.video_regression import VideoRegression

import json
from midi2audio import FluidSynth
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import random
from moviepy.editor import *
import time

from tqdm import tqdm
from huggingface_hub import snapshot_download

from gradio import Markdown

from pytube import YouTube
from pydub import AudioSegment
import pandas as pd

from utilities.argument_generate_funcs import parse_generate_args, print_generate_args
from utilities.device import get_device, use_cuda

all_key_names = ['C major', 'G major', 'D major', 'A major',
                 'E major', 'B major', 'F major', 'Bb major',
                 'Eb major', 'Ab major', 'Db major', 'Gb major',
                 'A minor', 'E minor', 'B minor', 'F# minor',
                 'C# minor', 'G# minor', 'D minor', 'G minor',
                 'C minor', 'F minor', 'Bb minor', 'Eb minor',
                 ]

traspose_key_dic = {
    'F major' : -7,
    'Gb major' : -6,
    'G major' : -5,
    'Ab major' : -4,
    'A major' : -3,
    'Bb major' : -2,
    'B major' : -1,
    'C major' : 0,
    'Db major' : 1,
    'D major' : 2,
    'Eb major' : 3,
    'E major' : 4,
    'D minor' : -7,
    'Eb minor' : -6,
    'E minor' : -5,
    'F minor' : -4,
    'F# minor' : -3,
    'G minor' : -2,
    'G# minor' : -1,
    'A minor' : 0,
    'Bb minor' : 1,
    'B minor' : 2,
    'C minor' : 3,
    'C# minor' : 4
}

flatsharpDic = {
    'Db':'C#', 
    'Eb':'D#', 
    'Gb':'F#', 
    'Ab':'G#', 
    'Bb':'A#'
}

replace_instrument_index_dict = {
    13: 14,
    18: 10,
    22: 28,
    26: 14,
    29: 25,
    31: 11
}

arpeggio_instrument_list = [3, 7, 8, 11, 14, 27, 31, 37, 38, 39]
left_panning_instrument_list = [13, 14, 16, 25, 28, 29, 34, 39]
right_panning_instrument_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 18, 19, 22, 26, 27]
center_panning_instrument_list = [7, 15, 17, 20, 21, 23, 24, 30, 32, 32, 33, 35, 36, 37, 38]

left_panning_val = 32
center_panning_val = 64
right_panning_val = 96

low_velocity_instrument_list = [14]

max_conseq_N = 0
max_conseq_chord = 2
base_tempo = 120
# tempo_instrument = [105, 85, 100, 90, 115, 70, 130, 120, 75, 95, 80, base_tempo, 70, 125, 120, 120,
#                     95, 110, 100, 110, 80, 80, 100, 80, 90, 70, 75, 130, 100, 60, 95, base_tempo,
#                     105, 90, 125, 90, 105, 75, 100, 85]
tempo_instrument = [base_tempo] * 40
numerator, denominator = 4, 4 # Time signature (Nhịp 4/4)
duration = 2

min_loudness = 0  # Minimum loudness level in the input range
max_loudness = 50  # Maximum loudness level in the input range
min_velocity = 49  # Minimum velocity value in the output range
max_velocity = 112  # Maximum velocity value in the output range

def split_video_into_frames(video, frame_dir):
    output_path = os.path.join(frame_dir, f"%03d.jpg")
    cmd = f"ffmpeg -i {video} -vf \"select=bitor(gte(t-prev_selected_t\,1)\,isnan(prev_selected_t))\" -vsync 0 -qmin 1 -q:v 1 {output_path}"        
    subprocess.call(cmd, shell=True)

def gen_semantic_feature(frame_dir, semantic_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    file_names = os.listdir(frame_dir)
    sorted_file_names = sorted(file_names)

    output_path = semantic_dir / "semantic.npy"
    if torch.cuda.is_available():
        features = torch.FloatTensor(len(sorted_file_names), 768).fill_(0)
        features = features.to(device)
        
        for idx, file_name in enumerate(sorted_file_names):
            fpath = frame_dir / file_name
            image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)          
            with torch.no_grad():
                image_features = model.encode_image(image)
            features[idx] = image_features[0]
        features = features.cpu().numpy()
        np.save(output_path, features)
    else:
        features = torch.FloatTensor(len(sorted_file_names), 768).fill_(0)
        for idx, file_name in enumerate(sorted_file_names):
            fpath = frame_dir / file_name
            image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)          
            with torch.no_grad():
                image_features = model.encode_image(image)
            features[idx] = image_features[0]
        features = features.numpy()
        np.save(output_path, features)

def gen_emotion_feature(frame_dir, emotion_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    text = clip.tokenize(["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]).to(device)

    file_names = os.listdir(frame_dir)
    sorted_file_names = sorted(file_names)
    output_path = emotion_dir / "emotion.lab" 

    emolist = []
    for file_name in sorted_file_names:
        fpath = frame_dir / file_name
        image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)                
        with torch.no_grad():  
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        fp1 = format(probs[0][0], ".4f")
        fp2 = format(probs[0][1], ".4f")
        fp3 = format(probs[0][2], ".4f")
        fp4 = format(probs[0][3], ".4f")
        fp5 = format(probs[0][4], ".4f")
        fp6 = format(probs[0][5], ".4f")
        
        emo_val = str(fp1) +" "+ str(fp2) +" "+ str(fp3) +" "+ str(fp4) +" "+ str(fp5) + " " + str(fp6)
        emolist.append(emo_val)
    
    with open(output_path ,'w' ,encoding = 'utf-8') as f:
        f.write("time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob\n")
        for i in range(0, len(emolist) ):
            f.write(str(i) + " "+emolist[i]+"\n")

def gen_scene_feature(video, scene_dir, frame_dir):
    video_stream = open_video(str(video))
    
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video_stream, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    sec = 0
    scenedict = {}
    for idx, scene in enumerate(scene_list):
        end_int = math.ceil(scene[1].get_seconds())
        for s in range (sec, end_int):
            scenedict[s] = str(idx)
            sec += 1
    
    fpathname = scene_dir / "scene.lab"

    if len(scene_list) == 0:
        fsize = len( os.listdir(frame_dir) )
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, fsize):
                f.write(str(i) + " "+"0"+"\n")
    else:
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, len(scenedict)):
                f.write(str(i) + " "+scenedict[i]+"\n")

def gen_scene_offset_feature(scene_dir, scene_offset_dir):
    src = scene_dir / "scene.lab"
    tgt = scene_offset_dir / "scene_offset.lab"
    
    id_list = []
    with open(src, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            if len(line_arr) == 2 :
                time = int(line_arr[0])
                scene_id = int(line_arr[1])
                id_list.append(scene_id)

    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    
    with open(tgt,'w',encoding = 'utf-8') as f:
        for i in range(0, len(offset_list)):
            f.write(str(i) + " " + str(offset_list[i]) + "\n")

def gen_motion_feature(video, motion_dir):
    # Motion origin
    # cap = cv2.VideoCapture(str(video))
    # prev_frame = None
    # prev_time = 0
    # motion_value = 0
    # motiondict = {}

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    #     motiondict[0] = "0.0000"
    #     if prev_frame is not None and curr_time - prev_time >= 1:
    #         diff = cv2.absdiff(frame, prev_frame)
    #         diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    #         motion_value = diff_rgb.mean()
    #         motion_value = format(motion_value, ".4f")
    #         motiondict[int(curr_time)] = str(motion_value)
    #         prev_time = int(curr_time)
    #     prev_frame = frame.copy()
    # cap.release()
    # cv2.destroyAllWindows()
    # fpathname = motion_dir / "motion.lab"
    
    # with open(fpathname,'w',encoding = 'utf-8') as f:
    #     for i in range(0, len(motiondict)):
    #         f.write(str(i) + " "+motiondict[i]+"\n")

    # Motion option 1
    model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)   
    model.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten()
    )
    model = model.to(get_device())
    model.eval()
    transform = models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()

    cap = cv2.VideoCapture(str(video))
    prev_frame = None
    prev_time = 0

    features = [np.zeros(512)]
    while cap.isOpened():
        # Read the frame and get its time stamp
        ret, frame = cap.read()
        if not ret:
            break
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Calculate the RGB difference between consecutive frames per second
        if prev_frame is not None and curr_time - prev_time >= 1:
            diff = cv2.absdiff(frame, prev_frame)
            diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

            diff_image = transform(Image.fromarray(diff_rgb)).unsqueeze(0).to(get_device())
            with torch.no_grad():
                motion_features = model(diff_image).squeeze()

            motion_features = motion_features.cpu().numpy()
            features.append(motion_features)

            prev_time = int(curr_time)

        # Update the variables
        prev_frame = frame.copy()
    # Release the video file and close all windows
    cap.release()
    cv2.destroyAllWindows()

    features = np.stack(features, axis=0)
    fpathname = motion_dir / "motion.npy"
    np.save(fpathname, features)

def get_scene_offset_feature(scene_offset_dir, max_seq_chord=300, max_seq_video=300):
    feature_scene_offset = np.empty(max_seq_video)
    feature_scene_offset.fill(SCENE_OFFSET_PAD)
    fpath_scene_offset = scene_offset_dir / "scene_offset.lab" 

    with open(fpath_scene_offset, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            time = line_arr[0]
            time = int(time)
            if time >= max_seq_chord:
                break
            sceneID = line_arr[1]
            feature_scene_offset[time] = int(sceneID)+1

    feature_scene_offset = torch.from_numpy(feature_scene_offset)
    feature_scene_offset = feature_scene_offset.to(torch.float32)

    return feature_scene_offset

def get_motion_feature(motion_dir, max_seq_chord=300, max_seq_video=300):
    # fpath_motion = motion_dir / "motion.lab" 
    # feature_motion = np.empty(max_seq_video)
    # feature_motion.fill(MOTION_PAD)
    # with open(fpath_motion, encoding = 'utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         line_arr = line.split(" ")
    #         time = line_arr[0]
    #         time = int(time)
    #         if time >= max_seq_chord:
    #             break
    #         motion = line_arr[1]
    #         feature_motion[time] = float(motion)

    # feature_motion = torch.from_numpy(feature_motion)
    # feature_motion = feature_motion.to(torch.float32)

    # Motion option 1
    fpath_motion = motion_dir / "motion.npy" 
    feature_motion = np.zeros((max_seq_video, 512))
    loaded_motion = np.load(fpath_motion)
    if loaded_motion.shape[0] > max_seq_chord:
        feature_motion = loaded_motion[:max_seq_chord, :]
    else:
        feature_motion[:loaded_motion.shape[0], :] = loaded_motion

    feature_motion = torch.from_numpy(feature_motion)
    feature_motion = feature_motion.to(torch.float32)
    return feature_motion

def get_emotion_feature(emotion_dir, max_seq_chord=300, max_seq_video=300):
    fpath_emotion = emotion_dir / "emotion.lab" 
    feature_emotion = np.empty((max_seq_video, 6))
    feature_emotion.fill(EMOTION_PAD)

    with open(fpath_emotion, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            if line_arr[0] == "time":
                continue
            time = line_arr[0]
            time = int(time)
            if time >= max_seq_chord:
                break
            emo1, emo2, emo3, emo4, emo5, emo6 = \
                line_arr[1],line_arr[2],line_arr[3],line_arr[4],line_arr[5],line_arr[6]                    
            emoList = [ float(emo1), float(emo2), float(emo3), float(emo4), float(emo5), float(emo6) ]
            emoList = np.array(emoList)
            feature_emotion[time] = emoList

    feature_emotion = torch.from_numpy(feature_emotion)
    feature_emotion = feature_emotion.to(torch.float32)
    return feature_emotion

def get_semantic_feature(semantic_dir, max_seq_chord=300, max_seq_video=300):
    fpath_semantic = semantic_dir / "semantic.npy" 
    
    video_feature = np.load(fpath_semantic)
    dim_vf = video_feature.shape[1]

    video_feature_tensor = torch.from_numpy( video_feature )
    feature_semantic = torch.full((max_seq_video, dim_vf,), SEMANTIC_PAD , dtype=torch.float32, device=torch.device("cpu"))

    if(video_feature_tensor.shape[0] < max_seq_video):
        feature_semantic[:video_feature_tensor.shape[0]] = video_feature_tensor
    else:
        feature_semantic = video_feature_tensor[:max_seq_video]
    
    return feature_semantic

def text_clip(text: str, duration: int, start_time: int = 0):
    t = TextClip(text, font='Georgia-Regular', fontsize=24, color='white')
    t = t.set_position(("center", 20)).set_duration(duration)
    t = t.set_start(start_time)
    return t

def convert_format_id_to_offset(id_list):
    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    return offset_list

# By ChatGPT
def copy_track(multi_track_midi: MIDIFile, single_track_midi: MIDIFile, track_index: int = 0, tempo: int = base_tempo):
    """
    Copies the i-th track of a multi-track MIDIFile object to a single-track MIDIFile object.

    Args:
        multi_track_midi (MIDIFile): Multi-track MIDIFile object.
        single_track_midi (MIDIFile): Single-track MIDIFile object (initially empty).
        track_index (int): Index of the track to copy.
    """
    # Check if track_index is valid
    if track_index > multi_track_midi.numTracks or track_index < 0:
        raise ValueError(f"Track index {track_index} is out of range for multi-track MIDI.")

    single_track_midi.addTempo(0, 0, tempo)
    single_track_midi.tracks[1] = copy.deepcopy(multi_track_midi.tracks[track_index])
    # for event in events:
    #     if (event.evtname == "NoteOn"):
    #         single_track_midi.addNote(0, event.channel, event.pitch, event.tick / 960, event.duration, event.volume) 
    #     elif (event.evtname == "NoteOff"):
    #         single_track_midi.addNote(0, event.channel, event.pitch, event.tick / 960, event.duration, event.volume) 

def addChord(midifile, chord, chord_offset, density_val, trans_val, time, duration, 
             velocity, emotion_index, arpeggio_chord=False):
    if emotion_index in (1, 2):   # Fearful, Tense
        trans_val += -2
    elif emotion_index in (3,):    # Sad
        trans_val += -1
    elif emotion_index in (0, 4): # Exciting, Relaxing
        trans_val += 1
    else:                         # Neutral
        trans_val += 0
    
    # Inner Chord Notes
    first_velo = 1.1
    second_velo = 0.95
    third_velo = 0.98
    fourth_velo = 1.0
    fifth_velo = 0.95
    diminish_velo = 0.6 # only for arpeggio_chord=False

    if arpeggio_chord:
        if density_val == 0:
            if len(chord) >= 4:
                if chord_offset % 2 == 0:
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 1, duration,  int(velocity*second_velo))
                else:
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1, duration,  int(velocity*fourth_velo))
                
                if len(chord) == 5:
                    midifile.addNote(0, 0, chord[4]+trans_val, time + 2, duration,  int(velocity*fifth_velo))
        elif density_val == 1:
            if len(chord) >= 4:
                if chord_offset % 2 == 0:
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1, duration,  int(velocity*third_velo))
                else:
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 0, duration,  int(velocity*fourth_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1, duration,  int(velocity*third_velo))
                
                if len(chord) == 5:
                    midifile.addNote(0, 0, chord[4]+trans_val, time + 1.5, duration,  int(velocity*fifth_velo))
        elif density_val == 2:
            if len(chord) >= 4:
                if chord_offset % 2 == 0:
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1.5, duration,  int(velocity*fourth_velo))
                else:
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1.5, duration,  int(velocity*fourth_velo))
                
                if len(chord) == 5:
                    midifile.addNote(0, 0, chord[4]+trans_val, time + 2, duration,  int(velocity*fifth_velo))
        elif density_val == 3:
            if len(chord) >= 4:
                if chord_offset % 2 == 0:
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.25, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0.5, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.75, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1, duration,  int(velocity*fourth_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.5, duration,  int(velocity*third_velo))
                else:
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0.25, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0.75, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1, duration,  int(velocity*fourth_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.5, duration,  int(velocity*third_velo))
                
                if len(chord) == 5:
                    midifile.addNote(0, 0, chord[4]+trans_val, time + 2, duration,  int(velocity*fifth_velo))
        elif density_val == 4:
            if len(chord) >= 4:
                if chord_offset % 2 == 0:
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.25, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0.5, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.75, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1, duration,  int(velocity*fourth_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.25, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 1.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.75, duration,  int(velocity*third_velo))
                else:
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[0]+trans_val, time + 0.25, duration,  int(velocity*first_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 0.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 0.75, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[3]+trans_val, time + 1, duration,  int(velocity*fourth_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.25, duration,  int(velocity*third_velo))
                    midifile.addNote(0, 0, chord[1]+trans_val, time + 1.5, duration,  int(velocity*second_velo))
                    midifile.addNote(0, 0, chord[2]+trans_val, time + 1.75, duration,  int(velocity*third_velo))
                
                if len(chord) == 5:
                    midifile.addNote(0, 0, chord[4]+trans_val, time + 2, duration,  int(velocity*fifth_velo))
    else:
        if len(chord) >= 4:
            midifile.addNote(0, 0, chord[0] + trans_val, time, duration, int(velocity*first_velo*diminish_velo))
            midifile.addNote(0, 0, chord[1] + trans_val, time, duration, int(velocity*second_velo*diminish_velo))
            midifile.addNote(0, 0, chord[2] + trans_val, time, duration, int(velocity*third_velo*diminish_velo))
            midifile.addNote(0, 0, chord[3] + trans_val, time, duration, int(velocity*fourth_velo*diminish_velo))
            if len(chord) == 5:
                midifile.addNote(0, 0, chord[4] + trans_val, time, duration, int(velocity*fifth_velo*diminish_velo))

class Video2music:
    def __init__(
        self,
        name="amaai-lab/video2music",
        device="cuda:0",
        cache_dir=None,
        local_files_only=False,
    ):
        # path = snapshot_download(repo_id=name, cache_dir=cache_dir)

        #Adjust print args HERE!
        self.isPrintArgs = True
        args = parse_generate_args()[0]
        
        if self.isPrintArgs:
          print_generate_args(args)

        self.device = device       
        
        self.model_weights = args.model_weights
        self.modelReg_weights = args.modelReg_weights

        # 768 (sem) + 1 (scene) + 6 (emo) + 512 (mo1) (AMT)       
        self.total_vf_dim = 1287     

        # 768 (sem) + 6 (emo) (AMT)
        self.total_vf_dim_reg = 774
      
        self.max_seq_video = 300
        self.max_seq_chord = 300
      
        # self.model = VideoMusicTransformer(n_layers=6, num_heads=8,
        #             d_model=512, dim_feedforward=1024,
        #             max_sequence_midi=2048, max_sequence_video=300, 
        #             max_sequence_chord=300, total_vf_dim=self.total_vf_dim, rpr=RPR).to(device)

        if args.music_gen_version == None:        
            self.model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=self.total_vf_dim, 
                        rpr=args.rpr).to(get_device())    
        elif args.music_gen_version.startswith('1.'):
            self.model = VideoMusicTransformer_V1(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=self.total_vf_dim,
                        rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed).to(get_device())
        elif args.music_gen_version.startswith('2.'):
            self.model = VideoMusicTransformer_V2(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=self.total_vf_dim,
                        rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed,
                        balancing=args.balancing).to(get_device())
        elif args.music_gen_version.startswith('3.'):
            self.model = VideoMusicTransformer_V3(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=self.total_vf_dim,
                        rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed).to(get_device())
                  
        self.model.load_state_dict(torch.load(self.model_weights, map_location=get_device()))

        self.modelReg = VideoRegression(n_layers=args.n_layers_reg, d_model=args.d_model_reg, d_hidden=args.dim_feedforward_reg, use_KAN=args.use_KAN_reg, max_sequence_video=args.max_sequence_video, total_vf_dim=self.total_vf_dim_reg, regModel=args.regModel).to(get_device())        
        self.modelReg.load_state_dict(torch.load(self.modelReg_weights, map_location=get_device()))

        # self.key_detector = joblib.load(args.modelpathKey)

        self.model.eval()
        self.modelReg.eval()

        self.SF2_FILE = "soundfonts/default_sound_font.sf2"

    def generate(self, video, primer=None, key=None, transposition_value=0, custom_sound_font=False, temperature=1.0):
        feature_dir = Path("./feature")
        output_dir = Path("./output")
        if feature_dir.exists():
            shutil.rmtree(str(feature_dir))
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        
        feature_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        frame_dir = feature_dir / "vevo_frame"

        #video features
        semantic_dir = feature_dir / "vevo_semantic"
        emotion_dir = feature_dir / "vevo_emotion"
        scene_dir = feature_dir / "vevo_scene"
        scene_offset_dir = feature_dir / "vevo_scene_offset"
        motion_dir = feature_dir / "vevo_motion"

        frame_dir.mkdir(parents=True)
        semantic_dir.mkdir(parents=True)
        emotion_dir.mkdir(parents=True)
        scene_dir.mkdir(parents=True)
        scene_offset_dir.mkdir(parents=True)
        motion_dir.mkdir(parents=True)
        
        #music features
        chord_dir = feature_dir / "vevo_chord"
        loudness_dir = feature_dir / "vevo_loudness"
        note_density_dir = feature_dir / "vevo_note_density"
        
        chord_dir.mkdir(parents=True)
        loudness_dir.mkdir(parents=True)
        note_density_dir.mkdir(parents=True)

        split_video_into_frames(video, frame_dir)
        gen_semantic_feature(frame_dir, semantic_dir)
        gen_emotion_feature(frame_dir, emotion_dir)
        gen_scene_feature(video, scene_dir, frame_dir)
        gen_scene_offset_feature(scene_dir, scene_offset_dir)
        gen_motion_feature(video, motion_dir)

        feature_scene_offset = get_scene_offset_feature(scene_offset_dir)
        feature_motion = get_motion_feature(motion_dir)
        feature_emotion = get_emotion_feature(emotion_dir)
        feature_semantic = get_semantic_feature(semantic_dir)

        # cuda
        feature_scene_offset = feature_scene_offset.to(self.device)
        feature_motion = feature_motion.to(self.device)
        feature_emotion = feature_emotion.to(self.device)

        feature_scene_offset = feature_scene_offset.unsqueeze(0)
        feature_motion = feature_motion.unsqueeze(0)
        feature_emotion = feature_emotion.unsqueeze(0)

        feature_semantic = feature_semantic.to(self.device)
        feature_semantic = torch.unsqueeze(feature_semantic, 0)
        feature_semantic_list = feature_semantic.to(self.device)

        emotion_idx = torch.argmax(feature_emotion.mean(dim=0))
        if key != None:
            key = key.strip()
            if key[-3:] == 'min': # Minor
                feature_key = torch.tensor([1]).float()
            else: # Major
                feature_key = torch.tensor([0]).float()
        else: # Key is not given
            if emotion_idx in (1, 2, 3): # Minor
                key = 'A minor'
                feature_key = torch.tensor([1]).float()
            else: # Major
                key = 'C major'
                feature_key = torch.tensor([0]).float()
        
        feature_key = feature_key.to(self.device)

        with open('dataset/vevo_meta/chord.json') as json_file:
            chordDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        with open("dataset/vevo_meta/instrument_inv.json", "r") as file:
            instrument_inv_dict = json.load(file)

        if primer == None or primer.strip() == "":
            if emotion_idx in (1, 2, 3):
                primer = "Am"
            else:
                primer = "C"
        
        pChordList = primer.split()

        primerCID = []
        primerCID_root = []
        primerCID_attr = []
        
        for pChord in pChordList:
            if len(pChord) > 1:
                if pChord[1] == "b":
                    pChord = flatsharpDic [ pChord[0:2] ] + pChord[2:]
                type_idx = 0
                if pChord[1] == "#":
                    pChord = pChord[0:2] + ":" + pChord[2:]
                    type_idx = 2
                else:
                    pChord = pChord[0:1] + ":" + pChord[1:]
                    type_idx = 1
                if pChord[type_idx+1:] == "m":
                    pChord = pChord[0:type_idx] + ":min"
                if pChord[type_idx+1:] == "m6":
                    pChord = pChord[0:type_idx] + ":min6"
                if pChord[type_idx+1:] == "m7":
                    pChord = pChord[0:type_idx] + ":min7"
                if pChord[type_idx+1:] == "M6":
                    pChord = pChord[0:type_idx] + ":maj6"
                if pChord[type_idx+1:] == "M7":
                    pChord = pChord[0:type_idx] + ":maj7"
                if pChord[type_idx+1:] == "":
                    pChord = pChord[0:type_idx]
            
            print("pchord is ", pChord)
            chordID = chordDic[pChord]
            primerCID.append(chordID)
            
            chord_arr = pChord.split(":")
            if len(chord_arr) == 1:
                chordRootID = chordRootDic[chord_arr[0]]
                primerCID_root.append(chordRootID)
                primerCID_attr.append(0)
            elif len(chord_arr) == 2:
                chordRootID = chordRootDic[chord_arr[0]]
                chordAttrID = chordAttrDic[chord_arr[1]]
                primerCID_root.append(chordRootID)
                primerCID_attr.append(chordAttrID)
        
        primerCID = np.array(primerCID)
        primerCID = torch.from_numpy(primerCID)
        primerCID = primerCID.to(torch.long)
        primerCID = primerCID.to(self.device)

        primerCID_root = np.array(primerCID_root)
        primerCID_root = torch.from_numpy(primerCID_root)
        primerCID_root = primerCID_root.to(torch.long)
        primerCID_root = primerCID_root.to(self.device)
        
        primerCID_attr = np.array(primerCID_attr)
        primerCID_attr = torch.from_numpy(primerCID_attr)
        primerCID_attr = primerCID_attr.to(torch.long)
        primerCID_attr = primerCID_attr.to(self.device)

        # self.model.eval()
        # self.modelReg.eval()

        with torch.set_grad_enabled(False):
            chord_sequence = self.model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = 300, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord,
                                              temperature=temperature)
            
            # Loudness, Note density, Instrument
            ln_nd, inst = self.modelReg(
                        feature_semantic_list, 
                        feature_scene_offset,
                        feature_motion,
                        feature_emotion)
        
            ln_nd   = ln_nd.reshape(ln_nd.shape[0] * ln_nd.shape[1], -1)

            y_note_density, y_loudness = torch.split(ln_nd, split_size_or_sections=1, dim=1)
            y_note_density_np = y_note_density.cpu().numpy()
            y_note_density_np = np.round(y_note_density_np).astype(int)
            y_note_density_np = np.clip(y_note_density_np, 0, 40)

            y_loudness_np = y_loudness.cpu().numpy()
            y_loudness_np_lv = (y_loudness_np * 100).astype(int)
            y_loudness_np_lv = np.clip(y_loudness_np_lv, 0, 50)

            # feature_emotion = feature_emotion.permute(1, 0, 2)
            # window_size = 5
            # avg_kernel = torch.ones(1, 1, window_size).to(get_device()) / window_size
            # feature_emotion = torch.nn.functional.conv1d(feature_emotion, avg_kernel, padding=window_size//2)
            # feature_emotion = feature_emotion.permute(1, 0, 2).squeeze()
            emotion_indice = torch.argmax(feature_emotion.squeeze(), dim=1).cpu()

            velolistExp = []
            exponent = 0.3
            for i, item in enumerate(y_loudness_np_lv):
                loudness = item[0]
                velocity_exp = np.round(((loudness - min_loudness) / (max_loudness - min_loudness)) ** exponent * (max_velocity - min_velocity) + min_velocity)
                velocity_exp = int(velocity_exp)

                if emotion_indice[i] in (0, 1):     # Exciting, Fearful
                    velocity_exp += 2
                elif emotion_indice[i] in (2,):      # Tense
                    velocity_exp += 1
                elif emotion_indice[i] in (3, 4):   # Sad, Relaxing
                    velocity_exp += 0
                else: # Neutral
                    velocity_exp += -1

                velolistExp.append(velocity_exp)
            
            densitylist = []
            for i, item in enumerate(y_note_density_np):
                density = item[0]

                if emotion_indice[i] in (1, 2, 3):  # Fearful, Tense, Sad
                    density += -3
                elif emotion_indice[i] in (0, 4):   # Exciting, Relaxing
                    density += 3
                else:                               # Neutral
                    density += 0

                if density <= 6:
                    densitylist.append(0)
                elif density <= 12:
                    densitylist.append(1)
                elif density <= 18:
                    densitylist.append(2)
                elif density <= 24:
                    densitylist.append(3)
                else:
                    densitylist.append(4)
            
            # generated ChordID to ChordSymbol
            chord_genlist = []
            chordID_genlist= chord_sequence[0].cpu().numpy()
            for index in chordID_genlist:
                chord_genlist.append(chordInvDic[str(index)])
            
            chord_offsetlist = convert_format_id_to_offset(chord_genlist)
            f_path_midi = output_dir / "output.mid"
            f_path_flac = output_dir / "output.flac"
            f_path_video_out = output_dir / "output.mp4"

            # ChordSymbol to MIDI file with voicing
            inst = inst.squeeze(0) # inst shape = (300, 40)
            inst = torch.where(inst >= 0.35, 1.0, 0.0)
            # Save instrument file
            df = pd.DataFrame(inst.cpu().numpy())
            df.to_csv(os.path.join(output_dir, "inst.csv"), index=False)                
            
            num_inst = inst.shape[1]

            midi_list = [MIDIFile(1) for _ in range(num_inst)] # For instrument rendering
            
            generated_midi = MIDIFile(1) # For saving midi file
            generated_midi.addTempo(0, 0, base_tempo)
            
            midi_chords_orginal = []
            for index, k in enumerate(chord_genlist):
                k = k.replace(":", "")
                if k == "N":
                    midi_chords_orginal.append([])
                else:
                    midi_chords_orginal.append(Chord(k).getMIDI(key[0].lower(), 4))
            midi_chords = voice(midi_chords_orginal)

            if key != None:
                trans = traspose_key_dic[key]
            else:
                trans = transposition_value

            choosed_instrument = set()
            # For multi_track_midi
            for inst_id in range(num_inst):
                midi_list[inst_id].addTempo(0, 0, tempo_instrument[inst_id])

                if inst_id in left_panning_instrument_list:
                    panning_val = left_panning_val
                elif inst_id in center_panning_instrument_list:
                    panning_val = center_panning_val
                else:
                    panning_val = right_panning_val

                midi_list[inst_id].addControllerEvent(0, 0, 0, panning_val, 0)

                for i, chord in enumerate(midi_chords):
                    # For generated_midi
                    if inst_id == 0:
                        # print(chord)
                        addChord(generated_midi, chord, chord_offsetlist[i], densitylist[i], trans, 
                                 i * duration, duration, velolistExp[i], emotion_indice[i], 
                                 arpeggio_chord=True)
                    
                    # For multi_track_midi
                    if inst[i, inst_id] == 1.0:
                        arpeggio_chord = inst_id in arpeggio_instrument_list
                        arpeggio_chord |= emotion_indice[i] in (0, 1, 2) # Exciting, Fearful, Tense

                        velocity = velolistExp[i] * (1.15 if inst_id in low_velocity_instrument_list else 1.0)

                        choosed_instrument.add(inst_id)

                        addChord(midi_list[inst_id], chord, chord_offsetlist[i], densitylist[i], 
                                 trans, i * duration, duration, int(velocity), emotion_indice[i], 
                                 arpeggio_chord=arpeggio_chord)
                                    
            # Save generated_midi file
            with open(f_path_midi, "wb") as outputFile:
                generated_midi.writeFile(outputFile)
     
            # Convert midi to audio (e.g., flac)
            if custom_sound_font == False:
                fs = FluidSynth(sound_font=self.SF2_FILE)
                fs.midi_to_audio(str(f_path_midi), str(f_path_flac))
            else:
                flac_files = []
                for inst_id in choosed_instrument:
                    if inst_id not in replace_instrument_index_dict.keys():                        
                        instrument_name = instrument_inv_dict[str(inst_id)]
                        print(inst_id, instrument_name)
                        filename = f"{str(inst_id)}_{instrument_name}.sf2"
                        f_path_midi_instrument = os.path.join(output_dir, f"output_{instrument_name}.mid")

                        # Save single-tracks MIDI file
                        with open(f_path_midi_instrument, "wb") as outputFile:
                            midi_list[inst_id].writeFile(outputFile)
                    
                        f_path_sf = os.path.join("soundfonts", filename)
                        flac_output = os.path.join(output_dir, f"output_{instrument_name}.flac")
                        fs = FluidSynth(sound_font=f_path_sf)
                        fs.midi_to_audio(str(f_path_midi_instrument), str(flac_output))
                        flac_files.append(flac_output)

                base_audio_index = 5
                mixed = AudioSegment.from_file(flac_files[base_audio_index])
                for i, audio_path in enumerate(flac_files):
                    if base_audio_index == i:
                        continue
                    mixed = mixed.overlay(AudioSegment.from_file(audio_path))
                mixed.export(f_path_flac, format="flac")

            # Render generated music into input video
            audio_mp = mp.AudioFileClip(str(f_path_flac))
            video_mp = mp.VideoFileClip(str(video))

            assert video_mp.duration > 0 and audio_mp.duration > 0
            audio_mp = audio_mp.subclip(0, video_mp.duration)
            final = video_mp.set_audio(audio_mp)

            final.write_videofile(str(f_path_video_out), 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            )
            return Path(str(f_path_video_out))
