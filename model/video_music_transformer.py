import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random
import numpy as np
from utilities.constants import *
from utilities.device import get_device
from .positional_encoding import PositionalEncoding
from .rpr import TransformerDecoderRPR, TransformerDecoderLayerRPR
from efficient_kan import KANLinear
from .custom_transformer import *
from .grouped_query_attention import *
from .rotate_operation import *
from .moe import *
from datetime import datetime
import json
from gensim.models import Word2Vec
from third_party.log_experts import get_highest_emotion_indices

chordEmbeddingModelPath = './word2vec_filled.bin'

class VideoMusicTransformer_V1(nn.Module):
    def __init__(self, version_name='1.1', n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi =2048, max_sequence_video=300, 
                 max_sequence_chord=300, total_vf_dim=0, rms_norm=False, scene_embed=False,
                 chord_embed=False, dropTokenRate=0.0):
        super(VideoMusicTransformer_V1, self).__init__()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_video    = max_sequence_video
        self.max_seq_chord    = max_sequence_chord
        self.scene_embed = scene_embed
        self.chord_embed = chord_embed
        self.dropTokenRate = dropTokenRate

        # Scene offsets embedding
        if self.scene_embed:
            self.scene_embedding = nn.Embedding(SCENE_OFFSET_MAX, self.d_model)

        # Chord embedding
        if self.chord_embed:
            chord_embedding_model = Word2Vec.load(chordEmbeddingModelPath)
            embedding_weights = torch.tensor(chord_embedding_model.wv.vectors)
            embedding_weights.requires_grad = False
            self.chord_embedding_model = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # AMT + MoE + Positional Embedding
        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)
        
        self.total_vf_dim = total_vf_dim
        self.Linear_vis     = nn.Linear(self.total_vf_dim, self.d_model)
        self.Linear_chord     = nn.Linear(self.d_model+1, self.d_model)
    
        # Positional embedding
        self.positional_embedding = nn.Embedding(self.max_seq_chord, self.d_model)
        self.positional_embedding_video = nn.Embedding(self.max_seq_video, self.d_model)

        # Add condition (minor or major)
        self.condition_linear = nn.Linear(1, self.d_model)
        
        # Transformer
        if rms_norm:
            norm = RMSNorm(self.d_model)
        else:
            norm = nn.LayerNorm(self.d_model)

        self.n_experts = 6
        self.n_experts_per_token = 2
        if version_name in ('1.1', '1.3'):
            expert = GLUExpert(self.d_model, self.d_ff, self.dropout)
        else:
            expert = nn.Sequential(
                nn.Linear(self.d_model, self.d_model*2),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model*2, self.d_model)
            )
            
        att = nn.MultiheadAttention(self.d_model, self.nhead, self.dropout)
        if version_name in ('1.0', '1.1'):
            moelayer = MoELayer(expert, self.d_model, self.n_experts, self.n_experts_per_token, self.dropout)
        else:
            moelayer = SharedMoELayer(expert, self.d_model, n_experts=self.n_experts, 
                                      n_experts_per_token=self.n_experts_per_token, balancing=False,
                                      dropout=self.dropout)

        if version_name != '1.3.3':
            # Encoder
            encoder_layer = TransformerEncoderLayer(att, moelayer, pre_norm=False, norm=norm, dropout=self.dropout)
            encoder = TransformerEncoder(encoder_layer, self.nlayers, norm)

            # Decoder
            decoder_layer = TransformerDecoderLayer(att, att, moelayer, pre_norm=False, norm=norm, dropout=self.dropout)
            decoder = TransformerDecoder(decoder_layer, self.nlayers, norm)
        else:
            swiglu = GLUExpert(self.d_model, self.d_ff, self.dropout)
            shallow_encoder_layer = TransformerEncoderLayer(att, swiglu, pre_norm=False, norm=norm, dropout=self.dropout)
            shallow_decoder_layer = TransformerDecoderLayer(att, att, swiglu, pre_norm=False, norm=norm, dropout=self.dropout)

            deep_encoder_layer = TransformerEncoderLayer(att, moelayer, pre_norm=False, norm=norm, dropout=self.dropout)
            deep_decoder_layer = TransformerDecoderLayer(att, att, moelayer, pre_norm=False, norm=norm, dropout=self.dropout)

            rate = 3
            encoder_layers = nn.ModuleList([copy.deepcopy(shallow_encoder_layer) for _ in range(rate)] + 
                                    [copy.deepcopy(deep_encoder_layer) for _ in range(self.nlayers-rate)])
            
            decoder_layers = nn.ModuleList([copy.deepcopy(shallow_decoder_layer) for _ in range(rate)] + 
                                    [copy.deepcopy(deep_decoder_layer) for _ in range(self.nlayers-rate)])
            
            encoder = TransformerEncoderShorter(encoder_layers, norm)
            decoder = TransformerDecoderShorter(decoder_layers, norm)

        # Full model
        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_encoder=encoder, custom_decoder=decoder
        )
    
        if IS_SEPERATED:
            self.Wout_root       = nn.Linear(self.d_model, CHORD_ROOT_SIZE)
            self.Wout_attr       = nn.Linear(self.d_model, CHORD_ATTR_SIZE)
        else:
            self.Wout       = nn.Linear(self.d_model, CHORD_SIZE)

        self.softmax    = nn.Softmax(dim=-1)

        del norm, expert, att, moelayer
        torch.cuda.empty_cache()

    def forward(self, x, x_root, x_attr, feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None
        
        if not self.chord_embed:
            x_root = self.embedding_root(x_root)
            x_attr = self.embedding_attr(x_attr)
            x = x_root + x_attr
        else:
            x = self.chord_embedding_model(x)

        tmp_list = list()
        for i in range(x.shape[0]):
            tmp = torch.full((1, x.shape[1], 1), feature_key[i,0].item() if feature_key.dim() > 1 else feature_key.item())
            tmp_list.append(tmp)
        feature_key_padded = torch.cat(tmp_list, dim=0)
        # feature_key_padded = torch.full((x.shape[0], x.shape[1], 1), feature_key.item())

        feature_key_padded = feature_key_padded.to(get_device())
        # print(x.shape, feature_key_padded.shape, feature_key.shape)
        x = torch.cat([x, feature_key_padded], dim=-1)

        xf = self.Linear_chord(x)

        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        # Semantic
        vf_concat = feature_semantic_list.float() 
        
        # Scene offset
        if not self.scene_embed:
            vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)

        # Motion
        try:
            vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        except:
            vf_concat = torch.cat([vf_concat, feature_motion], dim=-1)
        
        # Emotion
        get_highest_emotion_indices(feature_emotion)
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        
        # Video embedding
        if not self.scene_embed:
            vf = self.Linear_vis(vf_concat)
        else:
            vf = self.Linear_vis(vf_concat) + self.scene_embedding(feature_scene_offset.int())
        
        # Drop Tokens
        if self.dropTokenRate != 0.0:
            batch_size, seq_len, d_model = vf.shape
            droptoken_mask = (torch.rand(batch_size, seq_len) > self.dropTokenRate).float().to(get_device())
            droptoken_mask = droptoken_mask.unsqueeze(-1).repeat(1, 1, d_model)
            vf = vf * droptoken_mask

        ### POSITIONAL EMBEDDING ###
        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        vf = vf.permute(1,0,2) # -> (max_seq_video, batch_size, d_model)

        # Generate position indices
        xf_position_indices = torch.arange(xf.shape[0]).unsqueeze(1).expand(xf.shape[0], xf.shape[1]).to(get_device())
        vf_position_indices = torch.arange(vf.shape[0]).unsqueeze(1).expand(vf.shape[0], vf.shape[1]).to(get_device())

        xf += self.positional_embedding(xf_position_indices)
        vf += self.positional_embedding_video(vf_position_indices)

        del xf_position_indices, vf_position_indices
        torch.cuda.empty_cache()

        ### TRANSFORMER ###
        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1,0,2)

        if IS_SEPERATED:
            y_root = self.Wout_root(x_out)
            y_attr = self.Wout_attr(x_out)
            del mask
            return y_root, y_attr
        else:
            y = self.Wout(x_out)
            del mask
            return y
        
    def generate(self, feature_semantic_list = [], feature_key=None, feature_scene_offset=None, feature_motion=None, feature_emotion=None,
                 primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0, max_conseq_N = 0, max_conseq_chord = 2):
        
        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], 
                                           feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion) )[..., :CHORD_END]
            
            token_probs = y[:, cur_i-1, :]
            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)
            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i % CHORD_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                # token_probs.shape : [1, 157] 
                # 0: N, 1: C, ... , 156: B:maj7
                # 157 chordEnd 158 padding
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0
                isMaxChord = True
                if cur_i >= max_conseq_chord :
                    preChord = gen_seq[0][cur_i-1].item()      
                    for k in range (1, max_conseq_chord):
                        if preChord != gen_seq[0][cur_i-1-k].item():
                            isMaxChord = False
                else:
                    isMaxChord = False
                
                if isMaxChord:
                    preChord = gen_seq[0][cur_i-1].item()
                    token_probs[0][preChord] = 0.0
                
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                gen_chord = chordInvDic[ str( next_token.item() ) ]
                
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = 1
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                    
                # Let the transformer decide to end if it wants to
                if(next_token == CHORD_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]

class VideoMusicTransformer_V2(nn.Module):
    def __init__(self, version_name='2.0', n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi =2048, max_sequence_video=300, 
                 max_sequence_chord=300, total_vf_dim=0, rms_norm=False, scene_embed=False,
                 chord_embed=False, dropTokenRate=0.0):
        super(VideoMusicTransformer_V2, self).__init__()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_video    = max_sequence_video
        self.max_seq_chord    = max_sequence_chord
        self.scene_embed = scene_embed
        self.dropTokenRate = dropTokenRate
        self.chord_embed = chord_embed
        self.version_name = version_name

        # Scene offsets embedding
        if self.scene_embed:
            self.scene_embedding = nn.Embedding(SCENE_OFFSET_MAX, self.d_model)

        # Chord embedding
        if self.chord_embed:
            chord_embedding_model = Word2Vec.load(chordEmbeddingModelPath)
            embedding_weights = torch.tensor(chord_embedding_model.wv.vectors)
            embedding_weights.requires_grad = False
            self.chord_embedding_model = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)
        
        projection = nn.Linear
        # projection = KANLinear

        self.total_vf_dim = total_vf_dim
        self.Linear_vis     = projection(self.total_vf_dim, self.d_model)
        self.Linear_chord     = projection(self.d_model+1, self.d_model)

        # Add condition (minor or major)
        self.condition_linear = projection(1, self.d_model)
        
        # Transformer
        # if rms_norm:
        #     norm = RMSNorm(self.d_model)
        # else:
        #     norm = nn.LayerNorm(self.d_model)
        norm = RMSNorm(self.d_model)
        pre_norm = True

        use_KAN = False

        RoPE = None
        if version_name in ('2.0'):
            # Positional embedding
            self.positional_embedding = nn.Embedding(self.max_seq_chord, self.d_model)
            self.positional_embedding_video = nn.Embedding(self.max_seq_video, self.d_model)
        elif version_name in ('2.1', '2.2'):
            RoPE = RotaryPositionalEmbeddings(self.d_model, max_sequence_video)

        self.n_experts = 6
        self.n_experts_per_token = 2

        expert = GLUExpert(self.d_model, self.d_ff, self.dropout)

        att = CustomMultiheadAttention(self.d_model, self.nhead, self.dropout, RoPE=RoPE)
        
        topk_scheduler = TopKScheduler(n_experts=self.n_experts, min_n_experts_per_token=self.n_experts_per_token, update_step=32)
        if self.version_name == '2.2':
            topk_scheduler = None
        temperature_scheduler = None

        balancing = False
          
        moelayer = SharedMoELayer(expert=expert, d_model=self.d_model, n_experts=self.n_experts, 
                                  n_experts_per_token=self.n_experts_per_token, dropout=self.dropout, 
                                  balancing=balancing, topk_scheduler=topk_scheduler, 
                                  temperature_scheduler=temperature_scheduler, use_KAN=use_KAN)

        swiglu = GLUExpert(self.d_model, self.d_ff, self.dropout)
        shallow_encoder_layer = TransformerEncoderLayer(att, swiglu, pre_norm=pre_norm, norm=norm, dropout=self.dropout)
        shallow_decoder_layer = TransformerDecoderLayer(att, att, swiglu, pre_norm=pre_norm, norm=norm, dropout=self.dropout)

        deep_encoder_layer = TransformerEncoderLayer(att, moelayer, pre_norm=pre_norm, norm=norm, dropout=self.dropout)
        deep_decoder_layer = TransformerDecoderLayer(att, att, moelayer, pre_norm=pre_norm, norm=norm, dropout=self.dropout)

        rate = 3
        encoder_layers = nn.ModuleList([copy.deepcopy(shallow_encoder_layer) for _ in range(rate)] + 
                                [copy.deepcopy(deep_encoder_layer) for _ in range(self.nlayers-rate)])
        
        decoder_layers = nn.ModuleList([copy.deepcopy(shallow_decoder_layer) for _ in range(rate)] + 
                                [copy.deepcopy(deep_decoder_layer) for _ in range(self.nlayers-rate)])
        
        encoder = TransformerEncoderShorter(encoder_layers, norm)
        decoder = TransformerDecoderShorter(decoder_layers, norm)

        # Full model
        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_encoder=encoder, custom_decoder=decoder
        )   
    
        if IS_SEPERATED:
            self.Wout_root       = projection(self.d_model, CHORD_ROOT_SIZE)
            self.Wout_attr       = projection(self.d_model, CHORD_ATTR_SIZE)
        else:
            self.Wout       = projection(self.d_model, CHORD_SIZE)

        self.softmax    = nn.Softmax(dim=-1)

        del RoPE, expert, att, moelayer
        torch.cuda.empty_cache()

    def forward(self, x, x_root, x_attr, feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None
        
        if not self.chord_embed:
            x_root = self.embedding_root(x_root)
            x_attr = self.embedding_attr(x_attr)
            x = x_root + x_attr
        else:
            x = self.chord_embedding_model(x)

        tmp_list = list()
        for i in range(x.shape[0]):
            tmp = torch.full((1, x.shape[1], 1), feature_key[i,0].item() if feature_key.dim() > 1 else feature_key.item())
            tmp_list.append(tmp)
        feature_key_padded = torch.cat(tmp_list, dim=0)

        feature_key_padded = feature_key_padded.to(get_device())
        x = torch.cat([x, feature_key_padded], dim=-1)

        xf = self.Linear_chord(x)

        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        # Semantic
        vf_concat = feature_semantic_list.float() 
        
        # Scene offset
        if not self.scene_embed:
            vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)

        # Motion
        try:
            vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        except:
            vf_concat = torch.cat([vf_concat, feature_motion], dim=-1)
        
        # Emotion
        get_highest_emotion_indices(feature_emotion)
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        
        # Video embedding
        if not self.scene_embed:
            vf = self.Linear_vis(vf_concat)
        else:
            vf = self.Linear_vis(vf_concat) + self.scene_embedding(feature_scene_offset.int())
        
        # Drop Tokens
        if self.dropTokenRate != 0.0:
            batch_size, seq_len, d_model = vf.shape
            droptoken_mask = (torch.rand(batch_size, seq_len) > self.dropTokenRate).float().to(get_device())
            droptoken_mask = droptoken_mask.unsqueeze(-1).repeat(1, 1, d_model)
            vf = vf * droptoken_mask

        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        vf = vf.permute(1,0,2) # -> (max_seq_video, batch_size, d_model)

        if self.version_name in ('2.0'):
            # Generate position indices
            xf_position_indices = torch.arange(xf.shape[0]).unsqueeze(1).expand(xf.shape[0], xf.shape[1]).to(get_device())
            vf_position_indices = torch.arange(vf.shape[0]).unsqueeze(1).expand(vf.shape[0], vf.shape[1]).to(get_device())

            xf += self.positional_embedding(xf_position_indices)
            vf += self.positional_embedding_video(vf_position_indices)

            del xf_position_indices, vf_position_indices
            torch.cuda.empty_cache()

        ### TRANSFORMER ###
        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1,0,2)

        del mask

        if IS_SEPERATED:
            y_root = self.Wout_root(x_out)
            y_attr = self.Wout_attr(x_out)
            return y_root, y_attr
        else:
            y = self.Wout(x_out)
            return y
        
    def generate(self, feature_semantic_list = [], feature_key=None, feature_scene_offset=None, feature_motion=None, feature_emotion=None,
                 primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0, max_conseq_N = 0, max_conseq_chord = 2):
        
        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], 
                                           feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion) )[..., :CHORD_END]
            
            token_probs = y[:, cur_i-1, :]
            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)
            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i % CHORD_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                # token_probs.shape : [1, 157] 
                # 0: N, 1: C, ... , 156: B:maj7
                # 157 chordEnd 158 padding
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0
                isMaxChord = True
                if cur_i >= max_conseq_chord :
                    preChord = gen_seq[0][cur_i-1].item()      
                    for k in range (1, max_conseq_chord):
                        if preChord != gen_seq[0][cur_i-1-k].item():
                            isMaxChord = False
                else:
                    isMaxChord = False
                
                if isMaxChord:
                    preChord = gen_seq[0][cur_i-1].item()
                    token_probs[0][preChord] = 0.0
                
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                gen_chord = chordInvDic[ str( next_token.item() ) ]
                
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = 1
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                    
                # Let the transformer decide to end if it wants to
                if(next_token == CHORD_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]

class VideoMusicTransformer_V3(nn.Module):
    def __init__(self, version_name='3.0', n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi =2048, max_sequence_video=300, 
                 max_sequence_chord=300, total_vf_dim=0, rms_norm=False, scene_embed=False,
                 chord_embed=False, dropTokenRate=0.0):
        super(VideoMusicTransformer_V3, self).__init__()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_video    = max_sequence_video
        self.max_seq_chord    = max_sequence_chord
        self.scene_embed = scene_embed
        self.dropTokenRate = dropTokenRate
        self.chord_embed = chord_embed
        self.version_name = version_name

        # Scene offsets embedding
        if self.scene_embed:
            self.scene_embedding = nn.Embedding(SCENE_OFFSET_MAX, self.d_model)

        # Chord embedding
        if self.chord_embed:
            chord_embedding_model = Word2Vec.load(chordEmbeddingModelPath)
            embedding_weights = torch.tensor(chord_embedding_model.wv.vectors)
            embedding_weights.requires_grad = False
            self.chord_embedding_model = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)
        
        projection = nn.Linear
        # projection = KANLinear

        self.total_vf_dim = total_vf_dim
        self.Linear_vis     = projection(self.total_vf_dim, self.d_model)
        self.Linear_chord     = projection(self.d_model+1, self.d_model)

        # Add condition (minor or major)
        self.condition_linear = projection(1, self.d_model)
        
        # Transformer
        use_KAN = False

        RoPE = RotaryPositionalEmbeddings(self.d_model * 2, max_sequence_video)
        norm = RMSNorm(self.d_model, elementwise_affine=True)
        # norm = nn.LayerNorm(self.d_model)

        self.n_experts = 6
        self.n_experts_per_token = 2

        expert = GLUExpert(self.d_model, self.d_ff, self.dropout)
        
        # topk_scheduler = TopKScheduler(n_experts=self.n_experts, min_n_experts_per_token=self.n_experts_per_token, update_step=32)
        topk_scheduler = None
        temperature_scheduler = None

        balancing = True
          
        moelayer = SharedMoELayer(expert=expert, d_model=self.d_model, n_experts=self.n_experts, 
                                  n_experts_per_token=self.n_experts_per_token, dropout=self.dropout, 
                                  balancing=balancing, topk_scheduler=topk_scheduler, 
                                  temperature_scheduler=temperature_scheduler, use_KAN=use_KAN)

        swiglu = GLUExpert(self.d_model, self.d_ff, self.dropout)
        att = CustomMultiheadAttention(self.d_model, self.nhead, dropout=self.dropout, RoPE=RoPE)
        difatt_list = [
            DifferentialMultiheadAttention(self.d_model, 
                                           self.nhead, 
                                           dropout=self.dropout, 
                                           RoPE=RoPE, 
                                           depth=d) 
                for d in range(self.nlayers)
        ]

        rate = 3

        pre_norm = True if version_name == '3.2' else False

        if version_name == '3.0':
            shallow_encoder_layer = TransformerEncoderLayer(att, swiglu, pre_norm=pre_norm, norm=norm, dropout=self.dropout)
            deep_encoder_layer = TransformerEncoderLayer(att, moelayer, pre_norm=pre_norm, norm=norm, dropout=self.dropout)
            
            encoder_layers = nn.ModuleList([copy.deepcopy(shallow_encoder_layer) for _ in range(rate)] +
                                          [copy.deepcopy(deep_encoder_layer) for _ in range(self.nlayers - rate)])
        elif version_name in ('3.1', '3.2'):
            encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(difatt_list[i], 
                                        swiglu, 
                                        pre_norm=pre_norm, 
                                        norm=norm, 
                                        dropout=self.dropout) for i in range(rate)] + [
                TransformerEncoderLayer(difatt_list[i], 
                                        moelayer, 
                                        pre_norm=pre_norm, 
                                        norm=norm, 
                                        dropout=self.dropout) for i in range(rate, self.nlayers)])
        
        decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(difatt_list[i], 
                                    difatt_list[i],
                                    swiglu, 
                                    pre_norm=pre_norm, 
                                    norm=norm, 
                                    dropout=self.dropout) for i in range(rate)] + [
            TransformerDecoderLayer(difatt_list[i], 
                                    difatt_list[i], 
                                    moelayer, 
                                    pre_norm=pre_norm, 
                                    norm=norm, 
                                    dropout=self.dropout) for i in range(rate, self.nlayers)])
        
        encoder = TransformerEncoderShorter(encoder_layers, norm)
        decoder = TransformerDecoderShorter(decoder_layers, norm)

        # Full model
        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_encoder=encoder, custom_decoder=decoder
        )   
    
        if IS_SEPERATED:
            self.Wout_root       = projection(self.d_model, CHORD_ROOT_SIZE)
            self.Wout_attr       = projection(self.d_model, CHORD_ATTR_SIZE)
        else:
            self.Wout       = projection(self.d_model, CHORD_SIZE)

        self.softmax    = nn.Softmax(dim=-1)

        del RoPE, expert, difatt_list, att, swiglu, moelayer
        torch.cuda.empty_cache()

    def forward(self, x, x_root, x_attr, feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None
        
        if not self.chord_embed:
            x_root = self.embedding_root(x_root)
            x_attr = self.embedding_attr(x_attr)
            x = x_root + x_attr
        else:
            x = self.chord_embedding_model(x)

        tmp_list = list()
        for i in range(x.shape[0]):
            tmp = torch.full((1, x.shape[1], 1), feature_key[i,0].item() if feature_key.dim() > 1 else feature_key.item())
            tmp_list.append(tmp)
        feature_key_padded = torch.cat(tmp_list, dim=0)

        feature_key_padded = feature_key_padded.to(get_device())
        x = torch.cat([x, feature_key_padded], dim=-1)

        xf = self.Linear_chord(x)

        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        # Semantic
        vf_concat = feature_semantic_list.float() 
        
        # Scene offset
        if not self.scene_embed:
            vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)

        # Motion
        try:
            vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        except:
            vf_concat = torch.cat([vf_concat, feature_motion], dim=-1)
        
        # Emotion
        get_highest_emotion_indices(feature_emotion)
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        
        # Video embedding
        if not self.scene_embed:
            vf = self.Linear_vis(vf_concat)
        else:
            vf = self.Linear_vis(vf_concat) + self.scene_embedding(feature_scene_offset.int())
        
        # Drop Tokens
        if self.dropTokenRate != 0.0:
            batch_size, seq_len, d_model = vf.shape
            droptoken_mask = (torch.rand(batch_size, seq_len) > self.dropTokenRate).float().to(get_device())
            droptoken_mask = droptoken_mask.unsqueeze(-1).repeat(1, 1, d_model)
            vf = vf * droptoken_mask

        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        vf = vf.permute(1,0,2) # -> (max_seq_video, batch_size, d_model)

        ### TRANSFORMER ###
        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1,0,2)

        del mask

        if IS_SEPERATED:
            y_root = self.Wout_root(x_out)
            y_attr = self.Wout_attr(x_out)
            return y_root, y_attr
        else:
            y = self.Wout(x_out)
            return y
        
    def generate(self, feature_semantic_list = [], feature_key=None, feature_scene_offset=None, feature_motion=None, feature_emotion=None,
                 primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0, max_conseq_N = 0, max_conseq_chord = 2):
        
        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], 
                                           feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion) )[..., :CHORD_END]
            
            token_probs = y[:, cur_i-1, :]
            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)
            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i % CHORD_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                # token_probs.shape : [1, 157] 
                # 0: N, 1: C, ... , 156: B:maj7
                # 157 chordEnd 158 padding
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0
                isMaxChord = True
                if cur_i >= max_conseq_chord :
                    preChord = gen_seq[0][cur_i-1].item()      
                    for k in range (1, max_conseq_chord):
                        if preChord != gen_seq[0][cur_i-1-k].item():
                            isMaxChord = False
                else:
                    isMaxChord = False
                
                if isMaxChord:
                    preChord = gen_seq[0][cur_i-1].item()
                    token_probs[0][preChord] = 0.0
                
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                gen_chord = chordInvDic[ str( next_token.item() ) ]
                
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = 1
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                    
                # Let the transformer decide to end if it wants to
                if(next_token == CHORD_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]

class VideoMusicTransformer(nn.Module):
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi =2048, max_sequence_video=300, 
                 max_sequence_chord=300, total_vf_dim = 0, rpr=False, scene_embed=False,
                 chord_embed=False):
        super(VideoMusicTransformer, self).__init__()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_video    = max_sequence_video
        self.max_seq_chord    = max_sequence_chord
        self.rpr        = rpr
        self.scene_embed = scene_embed
        self.chord_embed = chord_embed

        # Scene offsets embedding
        if self.scene_embed:
            self.scene_embedding = nn.Embedding(SCENE_OFFSET_MAX, self.d_model)
        
        # Chord embedding
        if self.chord_embed:
            chord_embedding_model = Word2Vec.load(chordEmbeddingModelPath)
            embedding_weights = torch.tensor(chord_embedding_model.wv.vectors)
            embedding_weights.requires_grad = False
            self.chord_embedding_model = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)
        
        self.total_vf_dim = total_vf_dim
        self.Linear_vis     = nn.Linear(self.total_vf_dim, self.d_model)
        self.Linear_chord     = nn.Linear(self.d_model+1, self.d_model)
    
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq_chord)
        self.positional_encoding_video = PositionalEncoding(self.d_model, self.dropout, self.max_seq_video)

        # Add condition (minor or major)
        self.condition_linear = nn.Linear(1, self.d_model)
        
        # Base transformer
        if(not self.rpr):
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff
            )
        # RPR Transformer
        else:
            decoder_norm = LayerNorm(self.d_model)
            decoder_layer = TransformerDecoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq_chord)
            decoder = TransformerDecoderRPR(decoder_layer, self.nlayers, decoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=decoder #, batch_first=True
            )        
    
        self.Wout_root       = nn.Linear(self.d_model, CHORD_ROOT_SIZE)
        self.Wout_attr       = nn.Linear(self.d_model, CHORD_ATTR_SIZE)
        self.Wout       = nn.Linear(self.d_model, CHORD_SIZE)
        self.softmax    = nn.Softmax(dim=-1)
    
    def forward(self, x, x_root, x_attr, feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None
        
        if not self.chord_embed:
            x_root = self.embedding_root(x_root)
            x_attr = self.embedding_attr(x_attr)
            x = x_root + x_attr
        else:
            x = self.chord_embedding_model(x)

        tmp_list = list()
        for i in range(x.shape[0]):
            tmp = torch.full((1, x.shape[1], 1), feature_key[i,0].item() if feature_key.dim() > 1 else feature_key.item())
            tmp_list.append(tmp)
        feature_key_padded = torch.cat(tmp_list, dim=0)
        # feature_key_padded = torch.full((x.shape[0], x.shape[1], 1), feature_key.item())
        feature_key_padded = feature_key_padded.to(get_device())
        # print(x.shape, feature_key_padded.shape, feature_key.shape)
        x = torch.cat([x, feature_key_padded], dim=-1)

        xf = self.Linear_chord(x)

        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        # Semantic
        vf_concat = feature_semantic_list[0].float()
        for i in range(1, len(feature_semantic_list)):
            vf_concat = torch.cat( (vf_concat, feature_semantic_list[i].float()), dim=2)            
        
        # Scene offset
        if not self.scene_embed:
            vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)

        # Motion
        try:
            vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        except:
            vf_concat = torch.cat([vf_concat, feature_motion], dim=-1)
        
        # Emotion
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        
        # Video embedding
        if not self.scene_embed:
            vf = self.Linear_vis(vf_concat)
        else:
            vf = self.Linear_vis(vf_concat) + self.scene_embedding(feature_scene_offset.int())
        
        ### POSITIONAL ENCODING ###
        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        vf = vf.permute(1,0,2) # -> (max_seq_video, batch_size, d_model)
        xf = self.positional_encoding(xf)
        vf = self.positional_encoding_video(vf)

        ### TRANSFORMER ###
        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1,0,2)

        if IS_SEPERATED:
            y_root = self.Wout_root(x_out)
            y_attr = self.Wout_attr(x_out)
            del mask
            return y_root, y_attr
        else:
            y = self.Wout(x_out)
            del mask
            return y
    
    def generate(self, feature_semantic_list = [], feature_key=None, feature_scene_offset=None, feature_motion=None, feature_emotion=None,
                 primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0, max_conseq_N = 0, max_conseq_chord = 2):
        
        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], 
                                           feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion) )[..., :CHORD_END]
            
            token_probs = y[:, cur_i-1, :]
            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)
            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i % CHORD_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                # token_probs.shape : [1, 157] 
                # 0: N, 1: C, ... , 156: B:maj7
                # 157 chordEnd 158 padding
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0
                isMaxChord = True
                if cur_i >= max_conseq_chord :
                    preChord = gen_seq[0][cur_i-1].item()      
                    for k in range (1, max_conseq_chord):
                        if preChord != gen_seq[0][cur_i-1-k].item():
                            isMaxChord = False
                else:
                    isMaxChord = False
                
                if isMaxChord:
                    preChord = gen_seq[0][cur_i-1].item()
                    token_probs[0][preChord] = 0.0
                
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                gen_chord = chordInvDic[ str( next_token.item() ) ]
                
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = 1
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                    
                # Let the transformer decide to end if it wants to
                if(next_token == CHORD_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]

