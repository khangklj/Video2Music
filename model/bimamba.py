import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba, MambaConfig, RMSNorm

class BiMambaEncoder(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024, n_encoder_layers=2):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers

        self.layers = nn.ModuleList()
        for i in range(n_encoder_layers):
            if config.use_version == 0:
                self.layers.append(BiMambaEncoderLayer(config, dim_feedforward))
            else if config.use_version == 1:
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
        output_forward = self.feed_forward(self.norm1(mamba_out_forward)) + mamba_out_forward

        # Backward
        mamba_out_backward = self.mamba_backward(x_flip)
        output_backward = self.feed_forward(self.norm2(mamba_out_backward)) + mamba_out_backward

        # Combine output
        output = output_forward + output_backward

        return output
        
class BiMambaEncoderLayer_V1(nn.Module):
    def __init__(self, config: MambaConfig, dim_feedforward=1024):
        super().__init__()
        assert config.use_version == 1, "use_version should be 1 to use Mamba+"
        self.config = config
        self.mamba_forward = Mamba(config)
        self.mamba_backward = Mamba(config)
        self.d_ff = dim_feedforward
        
        self.norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, config.d_model)
        )
        
    def forward(self, x):
        x_flip = torch.flip(x, dims=[1])

        # Forward
        mamba_out_forward = self.mamba_forward(x)
        
        # Backward
        mamba_out_backward = self.mamba_backward(x_flip)
        
        # Combine output
        output = mamba_out_forward + mamba_out_backward
        output = self.feed_forward(self.norm(output)) + output
        
        return output
