import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        Initialize the LSTM cell.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            bias (bool): Whether to use bias in linear layers
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Linear layer for input-to-hidden transformation
        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        # Linear layer for hidden-to-hidden transformation
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM cell.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            hx: Tuple of (hidden state, cell state), each of shape (batch_size, hidden_size)
        
        Returns:
            hy: New hidden state tensor of shape (batch_size, hidden_size)
            cy: New cell state tensor of shape (batch_size, hidden_size)
        """
        # Initialize hidden and cell states if not provided
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
            hx = (hx, hx)
        
        # Unpack hidden and cell states
        hx, cx = hx
        
        # Compute gates
        gates = self.xh(input) + self.hh(hx)
        
        # Split gates into input, forget, cell, and output gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        # Apply activation functions to gates
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)
        
        # Update cell state
        cy = cx * f_t + i_t * g_t
        
        # Compute new hidden state
        hy = o_t * torch.tanh(cy)
        
        return (hy, cy)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            bias (bool): Whether to use bias in LSTM cells            
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # Create a list to hold LSTM cells
        self.lstm_cell_list = nn.ModuleList()
        
        # First LSTM cell takes input_size as input
        self.lstm_cell_list.append(LSTMCell(self.input_size, self.hidden_size, self.bias))
        
        # Subsequent LSTM cells take hidden_size as input
        for l in range(1, self.num_layers):
            self.lstm_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size, self.bias))
        
    def forward(self, input, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.lstm_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    hidden_l = self.lstm_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )
                hidden[layer] = hidden_l
            outs.append(hidden_l[0].unsqueeze(1))

        out = torch.cat(outs, dim=1)
        return out
