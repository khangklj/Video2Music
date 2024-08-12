import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba, MambaConfig, MambaBlock
from .custom_transformer import RMSNorm

class BiMambaEncoder(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, n_encoder_layers=2):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers

        self.layers = nn.ModuleList()
        for i in range(n_encoder_layers):
            if config.use_version == 0:
                self.layers.append(BiMambaEncoderLayer(config, dim_feedforward))
            elif config.use_version == 1:
                self.layers.append(BiMambaEncoderLayer_V1(config, dim_feedforward))

    def forward(self, x):
        for i in range(self.n_encoder_layers):
            x = self.layers[i](x)
        return x

class BiMambaEncoderLayer(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024):
        super().__init__()
        self.config = config
        self.mamba_forward = Mamba(config)
        self.mamba_backward = Mamba(config)
        self.d_ff = dim_feedforward

        # Norm and FF_layer
        # self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        # self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, config.d_model)
        )

    def forward(self, x):        
        x_flip = torch.flip(x, dims=[1])

        # Forward
        mamba_out_forward = self.mamba_forward(x)
        mamba_out_forward = self.norm1(mamba_out_forward)
        output_forward = self.feed_forward(mamba_out_forward) + mamba_out_forward

        # Backward
        mamba_out_backward = self.mamba_backward(x_flip)
        mamba_out_bacward = self.norm2(mamba_out_backward)
        output_backward = self.feed_forward(mamba_out_backward) + mamba_out_backward

        # Combine output
        output = output_forward + output_backward

        return output

class BiMambaEncoderLayer_V1(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        assert config.use_version == 1, "use_version should be 1 to use Mamba+"
        self.config = config
        # Use MambaBlock instead Mamba
        self.mamba_forward = MambaBlock(config)
        self.mamba_backward = MambaBlock(config)
        self.d_ff = dim_feedforward
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # self.norm1 = nn.LayerNorm(config.d_model)
        # self.norm2 = nn.LayerNorm(config.d_model)
        # self.norm3 = nn.LayerNorm(config.d_model)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, config.d_model)
        )
        
    def forward(self, x):
        _x = x
        # Flip
        x_flip = torch.flip(x, dims=[1])

        # Forward
        mamba_out_forward = self.mamba_forward(x)        
        # Add & Norm
        mamba_out_forward = self.dropout1(mamba_out_forward)
        mamaba_out_forward = self.norm1(mamba_out_forward + _x)
        
        # Backward
        mamba_out_backward = self.mamba_backward(x_flip)
        # Flip again
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])
        # Add & Norm
        mamba_out_backward = self.dropout2(mamba_out_backward)
        mamba_out_backward = self.norm2(mamba_out_backward + _x)
        
        # Combine output
        output = mamba_out_forward + mamba_out_backward

        # Feed forward network
        _output = output
        output = self.feed_forward(output)
        
        # Add & Norm
        output = self.dropout3(output)
        output = self.norm3(output + _output)
        
        return output
        
# class BiMambaEncoderLayer_V1(nn.Module):
#     def __init__(self, config: MambaConfig, dim_feedforward=1024, dropout=0.1):
#         super().__init__()
#         assert config.use_version == 1, "use_version should be 1 to use Mamba+"
#         self.config = config
#         self.mamba_forward = Mamba(config)
#         self.mamba_backward = Mamba(config)
#         self.d_ff = dim_feedforward
#         self.dropout = nn.Dropout(dropout)
        
#         self.norm = nn.LayerNorm(config.d_model)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(config.d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, config.d_model)
#         )
        
#     def forward(self, x):
#         x_flip = torch.flip(x, dims=[1])

#         # Forward
#         mamba_out_forward = self.mamba_forward(x)
        
#         # Backward
#         mamba_out_backward = self.mamba_backward(x_flip)
        
#         # Combine output
#         output = mamba_out_forward + mamba_out_backward

#         # Feed forward network
#         _output = output
#         output = self.feed_forward(output)

#         # Add & Norm        
#         output = self.norm(output + _output)
        
#         return output
