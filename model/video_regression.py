import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random
import numpy as np
from utilities.constants import *
from utilities.device import get_device
from datetime import datetime

import torch.nn.functional as F

class VideoRegression(nn.Module):
    def __init__(self, n_layers=2, d_model=64, dropout=0.1, max_sequence_video=300, total_vf_dim = 0, regModel="bilstm"):
        super(VideoRegression, self).__init__()
        self.nlayers    = n_layers
        self.d_model    = d_model
        self.dropout    = dropout
        self.max_seq_video    = max_sequence_video
        self.total_vf_dim = total_vf_dim
        self.regModel = regModel
        self.mlp_hidden_size = d_model * 4

        self.bilstm = nn.LSTM(self.total_vf_dim, self.d_model, self.nlayers, bidirectional=True)
        self.bigru = nn.GRU(self.total_vf_dim, self.d_model, self.nlayers, bidirectional=True)
        self.bifc = nn.Linear(self.d_model * 2, 2)

        self.lstm = nn.LSTM(self.total_vf_dim, self.d_model, self.nlayers)
        self.gru = nn.GRU(self.total_vf_dim, self.d_model, self.nlayers)
        self.fc = nn.Linear(self.d_model, 2)

        # First RNN layer (bidirectional)
        self.bigru1 = nn.GRU(self.total_vf_dim, self.d_model, self.nlayers, bidirectional=True, batch_first=True)
        
        # First MLP layer
        self.fc1 = nn.Linear(self.d_model * 2, self.mlp_hidden_size)  
        
        # Second RNN layer (non-bidirectional)
        self.bigru2 = nn.GRU(self.mlp_hidden_size, self.d_model, self.nlayers, bidirectional=True, batch_first=True)
        
        # Second MLP layer
        self.fc2 = nn.Linear(self.d_model * 2, 2)
    
    def forward(self, feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion):
        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        vf_concat = feature_semantic_list[0].float()
        for i in range(1, len(feature_semantic_list)):
            vf_concat = torch.cat( (vf_concat, feature_semantic_list[i].float()), dim=2)            
        
        vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) 
        vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) 
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) 

        vf_concat = vf_concat.permute(1,0,2)
        vf_concat = F.dropout(vf_concat, p=self.dropout, training=self.training)

        if self.regModel == "bilstm":
            out, _ = self.bilstm(vf_concat)
            out = out.permute(1,0,2)
            out = self.bifc(out)
        elif self.regModel == "bigru":
            out, _ = self.bigru(vf_concat)
            out = out.permute(1,0,2)
            out = self.bifc(out)
        elif self.regModel == "lstm":
            out, _ = self.lstm(vf_concat)
            out = out.permute(1,0,2)
            out = self.fc(out)
        elif self.regModel == "gru":
            out, _ = self.gru(vf_concat)
            out = out.permute(1,0,2)
            out = self.fc(out)
        elif self.regModel == "custom_RNN":
            vf_concat = vf_concat.permute(1,0,2)
            # print(f"vf_concat shape {vf_concat.shape}")
            # First RNN layer
            gru1_out, _ = self.bigru1(vf_concat)
            gru1_out = F.relu(gru1_out)            
            # print(f"RNN1 output shape: {gru1_out.shape}")
            
            # First MLP layer
            mlp1_out = self.fc1(gru1_out)
            mlp1_out = F.relu(mlp1_out)
            # print(f"MLP1 output shape: {mlp1_out.shape}")
            
            # Second RNN layer
            gru2_out, _ = self.bigru2(mlp1_out)
            gru2_out = F.relu(gru2_out)
            # print(f"RNN2 output shape: {gru2_out.shape}")
            
            # Second MLP layer
            out = self.fc2(gru2_out)
            # print(f"Final output shape: {out.shape}")
            
        return out
        
