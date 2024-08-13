import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba, MambaConfig, MambaBlock
from .custom_transformer import RMSNorm

class BiMambaEncoder(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, n_encoder_layers=2, dropout=0.2):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers

        self.layers = nn.ModuleList()
        for i in range(n_encoder_layers):
            if config.use_version == 0:
                self.layers.append(BiMambaEncoderLayer(config, dim_feedforward, dropout))
            elif config.use_version == 1:
                self.layers.append(BiMambaEncoderLayer_V1(config, dim_feedforward, dropout))

    def forward(self, x):
        for i in range(self.n_encoder_layers):
            x = self.layers[i](x)
        return x

class BiMambaEncoderLayer(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.config = config
        # Use MambaBlock instead of Mamba
        self.mamba_forward = MambaBlock(config)
        self.mamba_backward = MambaBlock(config)
        self.d_ff = dim_feedforward

        # Norm and FF_layer
        # self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        # self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.norm4 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn1 = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, config.d_model)
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, config.d_model)
        )

    def forward(self, x):        
        x_flip = torch.flip(x, dims=[1])

        _x_f = x
        _x_b = x

        # Forward
        x_f = self.mamba_forward(x)
        # Add & Norm
        x_f = self.dropout(x_f)
        x_f = self.norm1(x_f + _x_f)
        # FFN
        _x_f =  x_f
        x_f = self.ffn1(x_f)
        # Add & Norm
        x_f = self.dropout(x_f)
        x_f = self.norm2(x_f + _x_f)

        # Backward
        x_b = self.mamba_backward(x_flip)
        # Flip
        x_b = torch.flip(x_b, dims=[1])
        # Add & Norm
        x_b = self.dropout(x_b)
        x_b = self.norm3(x_b + _x_b)
        # FFN
        _x_b = x_b
        x_b = self.ffn2(x_f)
        # Add & Norm
        x_b = self.dropout(x_b)
        x_b = self.norm4(x_b + _x_b)

        # Combine the output
        x = x_f + x_b
        
        return x

class BiMambaEncoderLayer_V1(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        assert config.use_version == 1, "use_version should be 1 to use Mamba+"
        self.config = config
        # Use MambaBlock instead of Mamba
        self.mamba_forward = MambaBlock(config)
        self.mamba_backward = MambaBlock(config)
        self.d_ff = dim_feedforward
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, config.d_model)
        )
        
    def forward(self, x):
        # Flip
        x_flip = torch.flip(x, dims=[1])
        
        _x = x
        
        # Forward
        x_f = self.mamba_forward(x)
        # Add & Norm
        x_f = self.dropout(x_f)
        x_f = self.norm1(x_f + _x)

        # Backward
        x_b = self.mamba_backward(x_flip)
        # Flip
        x_b = torch.flip(x_b, dims=[1])
        # Add & Norm
        x_b = self.dropout(x_b)
        x_b = self.norm2(x_b + _x)

        # Combine output
        x = x_f + x_b

        # FFN
        _x = x
        x = self.ffn(x)
        # Add & Norm
        x = self.dropout(x)
        x = self.norm3(x + _x)

        return x