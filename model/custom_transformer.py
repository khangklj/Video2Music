import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.init import *

from torch.nn.functional import linear, softmax, dropout

class MyRMSNorm(Module):
    def __init__(self, dim, eps=1e-6, batch_first=False): # dim = (seq_len, d_model)
        super(MyRMSNorm, self).__init__()
        self.norm = nn.RMSNorm(dim, eps)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first: # x.shape = (batch, seq_len, d_model)
            return self.norm(x)
        else: # x.shape = (seq_len, batch, d_model)
            x = x.permute(1,0,2)
            x = self.norm(x)
            x = x.permute(1,0,2)
            return x
        pass

class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output