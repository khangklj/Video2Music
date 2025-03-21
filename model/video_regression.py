import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
# from torchtune.modules import RMSNorm
import random
import numpy as np
from utilities.constants import *
from utilities.device import get_device
from datetime import datetime
from .moe import *
# from .custom_lstm import LSTM
# from .custom_gru import GRU
# from .custom_reg_model import myRNN
import torch.nn.functional as F
from efficient_kan import KANLinear

from .mamba import Mamba, MoEMamba, MambaConfig
from .bimamba import BiMambaEncoder

from .minGRU import minGRU
from .minGRULM import minGRULM

class advancedRNNBlock(nn.Module):
    def __init__(self, rnn_type='gru', ff_type='mlp', d_model=256, d_hidden=1024, dropout=0.1, bidirectional=True):
        super(advancedRNNBlock, self).__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.bidirectional = bidirectional

        if rnn_type == None:
            self.rnn_layer = nn.RNN(self.d_model, self.d_model, num_layers=1, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn_layer = nn.GRU(self.d_model, self.d_model, num_layers=1, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(self.d_model, self.d_model, num_layers=1, bidirectional=bidirectional, batch_first=True)

        if ff_type == 'mlp':
            self.ff_layer = nn.Sequential(
                nn.Linear(self.d_model * (2 if bidirectional else 1), self.d_hidden),
                nn.SiLU(),
                nn.Linear(self.d_hidden, self.d_model)
            )
        elif ff_type == 'moe':
            expert = nn.Sequential(
                GLUExpert(self.d_model * (2 if bidirectional else 1), self.d_hidden),
                nn.Linear(self.d_model * (2 if bidirectional else 1), self.d_model)
            )

            self.ff_layer = MoELayer(expert, self.d_model * (2 if bidirectional else 1), d_ff=self.d_hidden, n_experts=6, n_experts_per_token=1, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.last_layer = nn.Linear(self.d_model * (2 if bidirectional else 1), self.d_model)

    def forward(self, x):
        x_rnn, _ = self.rnn_layer(x)
        x_double = torch.cat((x, x), dim=-1)
        x = self.dropout1(x_rnn + x_double)

        x_ff = self.ff_layer(x)
        x_double = torch.cat((x, x), dim=-1)
        print(x.shape, x_ff.shape, x_double.shape)
        x = self.dropout2(x_ff + x_double)

        x = self.last_layer(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, rnn_output):
        dynamic_vector = self.proj(rnn_output)

        attention_scores = torch.sum(rnn_output * dynamic_vector, dim=-1, keepdim=True)
        attention_weights = F.softmax(attention_scores, dim=1)

        context = torch.sum(attention_weights * rnn_output, dim=1)
        return context, attention_weights

class CNN_GRU(nn.Module):
    def __init__(self, d_input, d_model, num_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, bidirectional=bidirectional, 
                          batch_first=True, dropout=dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # Shape: [batch_size, d_model, seq_len]
        x = x.permute(0, 2, 1)
        
        out, _ = self.gru(x)
        return out

class VideoRegression(nn.Module):
    def __init__(self, n_layers=2, d_model=64, d_hidden=1024, dropout=0.1, use_KAN=False, max_sequence_video=300, 
                 total_vf_dim=0, regModel="bilstm", scene_embed=False, chord_embed=False):
        super(VideoRegression, self).__init__()
        self.n_layers    = n_layers
        self.d_model    = d_model
        self.d_hidden = d_hidden
        self.dropout_layer    = nn.Dropout(dropout)
        self.max_seq_video    = max_sequence_video
        self.total_vf_dim = total_vf_dim
        self.regModel = regModel
        self.scene_embed = scene_embed
        self.chord_embed = chord_embed

        # Scene offsets embedding
        # if self.scene_embed:
        #     self.scene_embedding = nn.Embedding(SCENE_OFFSET_MAX, self.d_model)

        # self.Linear_vis     = nn.Linear(self.total_vf_dim, self.d_model)

        if self.regModel == "bilstm":
            self.model = nn.LSTM(self.d_model, self.d_model, self.n_layers, 
                                  bidirectional=True, dropout=dropout, batch_first=True)
        elif self.regModel == "bigru":
            self.model = nn.GRU(self.d_model, self.d_model, self.n_layers, 
                                bidirectional=True, dropout=dropout, batch_first=True)
        elif self.regModel == "lstm":
            self.model = nn.LSTM(self.d_model, self.d_model, self.n_layers, 
                                 dropout=dropout, batch_first=True)
        elif self.regModel == "gru":
            self.model = nn.GRU(self.d_model, self.d_model, self.n_layers, 
                                dropout=dropout, batch_first=True)          
        elif self.regModel == "cnngru":
            self.model = CNN_GRU(self.d_model, self.d_model, self.n_layers, 
                                 dropout=dropout, bidirectional=False)
        elif self.regModel == "cnnbigru":
            self.model = CNN_GRU(self.d_model, self.d_model, self.n_layers, 
                                 dropout=dropout, bidirectional=True)
        elif self.regModel == "mamba":
            config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers, 
                                 use_KAN=use_KAN, bias=True)
            self.model = Mamba(config)           
        elif self.regModel == "mamba+":
            config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers, 
                                 use_KAN=use_KAN, bias=True, use_version=1)
            self.model = Mamba(config)
        elif self.regModel == "moemamba":
            config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers, 
                                 d_state=self.d_hidden, d_conv=8, dropout=dropout, 
                                 use_KAN=use_KAN, bias=True)
            expert = GLUExpert(self.d_model, self.d_model * 2 + 1)
            moe_layer = SharedMoELayer(expert, self.d_model, n_experts=6, 
                                       n_experts_per_token=2, dropout=dropout)
            self.model = MoEMamba(moe_layer, config)
        elif self.regModel == "bimamba":
            config = MambaConfig(d_model=self.d_model, n_layers=1, dropout=dropout, 
                                 use_KAN=use_KAN, bias=True, use_version=0)
            self.model = BiMambaEncoder(config, self.d_hidden, n_encoder_layers=self.n_layers, 
                                        dropout=dropout)
        elif self.regModel == "bimamba+":
            config = MambaConfig(d_model=self.d_model, n_layers=1, dropout=dropout, 
                                 use_KAN=use_KAN, bias=True, use_version=1)
            self.model = BiMambaEncoder(config, self.d_hidden, n_encoder_layers=self.n_layers, 
                                        dropout=dropout)
        elif self.regModel == "moe_bimamba+":
            expert = GLUExpert(self.d_model, self.d_model * 2 + 1)
            moe_layer = MoELayer(expert, self.d_model, n_experts=6, n_experts_per_token=2, 
                                 dropout=dropout)
            config = MambaConfig(d_model=self.d_model, n_layers=1, dropout=dropout, 
                                 use_KAN=use_KAN, bias=True, use_version=1)
            self.model = BiMambaEncoder(config, self.d_hidden, n_encoder_layers=self.n_layers, 
                                        dropout=dropout, moe_layer=moe_layer)
        elif self.regModel == "sharedmoe_bimamba+":
            expert = GLUExpert(self.d_model, self.d_model * 2 + 1)
            moe_layer = SharedMoELayer(expert, self.d_model, n_experts=6, n_experts_per_token=2, 
                                       dropout=dropout)
            config = MambaConfig(d_model=self.d_model, n_layers=1, dropout=dropout, 
                                 use_KAN=use_KAN, bias=True, use_version=1)
            self.model = BiMambaEncoder(config, self.d_hidden, n_encoder_layers=self.n_layers, 
                                        dropout=dropout, moe_layer=moe_layer)
    
        projection = nn.Linear
        # projection = KANLinear

        self.in_proj = nn.Sequential(
            projection(self.total_vf_dim, self.d_model),
            nn.Dropout(dropout)
        )

        if self.regModel in ('gru', 'lstm', 'cnngru', \
                             'mamba', 'moemamba', 'mamba+', 'bimamba', 'bimamba+', \
                             'moe_bimamba+', 'sharedmoe_bimamba+', 'minGRU'):
            self.regressor = projection(self.d_model, 2)
            self.classifier = nn.Sequential(
                projection(self.d_model, INSTRUMENT_SIZE),
                nn.Sigmoid()
            )
        elif self.regModel in ('bigru', 'bilstm', 'cnnbigru'):
            self.regressor = projection(self.d_model * 2, 2)
            self.classifier = nn.Sequential(
                projection(self.d_model * 2, INSTRUMENT_SIZE),
                nn.Sigmoid()
            )

    def get_feature(self, feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion):
        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        # Semantic
        vf_concat = feature_semantic_list.float() 
        
        # Scene offset
        # vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1)

        # Motion
        # try:
        #     vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1)
        # except:
        #     vf_concat = torch.cat([vf_concat, feature_motion], dim=-1)
        
        # Emotion
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (batch_size, max_seq_video, total_vf_dim)
        
        # Video embedding
        # if not self.scene_embed:
        #     vf_concat = self.Linear_vis(vf_concat)
        # else:
        #     vf_concat = self.Linear_vis(vf_concat) + self.scene_embedding(feature_scene_offset.int())

        vf = self.in_proj(vf_concat)
        if self.regModel in ("bilstm", "bigru", "lstm", "gru"):
            out, _ = self.model(vf)
        elif self.regModel in ("mamba", "moemamba", "mamba+", 'bimamba', 'bimamba+', \
                               'moe_bimamba+', 'sharedmoe_bimamba+', 'cnngru', 'cnnbigru'):            
            out = self.model(vf)

        return out

    def forward(self, feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion):
        out = self.get_feature(feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion)
        
        loudness_notedensity = self.regressor(out)
        instrument = self.classifier(out)

        return loudness_notedensity, instrument