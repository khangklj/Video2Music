import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        """
        Initialize the LSTM cell.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size        
        
        # Linear layer for input-to-hidden transformation
        self.xh = nn.Linear(input_size, hidden_size * 4)
        # Linear layer for hidden-to-hidden transformation
        # self.hh = nn.Linear(hidden_size, hidden_size * 4)
        self.hh = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size * 4)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):
        """
        Forward pass of the LSTM cell.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            hx: Tuple of (hidden state, cell state), each of shape (batch_size, hidden_size)
        
        Returns:
            hy: New hidden state tensor of shape (batch_size, hidden_size)
            cy: New cell state tensor of shape (batch_size, hidden_size)
        """
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
        
        return hy, cy

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size=1, bidirectional=False):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            batch_size (int): Batch size (default: 1)
            bidirectional (bool): Bidirectional LSTM (default: False)
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        # Create a list to hold LSTM cells
        self.lstm_cell_list = nn.ModuleList()
        
        # First LSTM cell takes input_size as input
        self.lstm_cell_list.append(LSTMCell(self.input_size, self.hidden_size))
        
        # Subsequent LSTM cells take hidden_size as input
        for l in range(1, self.num_layers):
            self.lstm_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size))

        # Create additional LSTM cells for bidirectional LSTM
        if self.bidirectional:
            for l in range(self.num_layers):
                self.lstm_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size))

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM.
        
        Args:
            input: Input tensor of shape (sequence length, batch_size, input_size)
            hx: Initial hidden state and cell state (optional)
        
        Returns:
            output: Output tensor of shape (sequence length, batch_size, output_size)
            (h_n, c_n): Final hidden state and cell state
        """
        print(input.shape)
        seq_len, batch_size, _ = input.size()

        # Initialize hidden state and cell state if not provided
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).cuda()
                c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
                c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
        else:
            h0, c0 = hx

        output = []
        hidden = [(h0[i], c0[i]) for i in range(self.num_layers * (2 if self.bidirectional else 1))]

        # Process each time step
        for t in range(seq_len):
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the sequence
                    hidden_l = self.lstm_cell_list[layer](
                        input[t, :, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    # Subsequent layers take input from the previous layer
                    hidden_l = self.lstm_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )
                hidden[layer] = hidden_l

            # Store the output of the last layer
            output.append(hidden[-1][0])

            # Process the backward direction if bidirectional
            if self.bidirectional:
                for layer in range(self.num_layers, len(hidden)):
                    if layer == self.num_layers:
                        # First backward layer takes input from the sequence
                        hidden_l = self.lstm_cell_list[layer](
                            input[seq_len - t - 1, :, :],
                            (hidden[layer][0], hidden[layer][1])
                        )
                    else:
                        # Subsequent backward layers take input from the previous layer
                        hidden_l = self.lstm_cell_list[layer](
                            hidden[layer - 1][0],
                            (hidden[layer][0], hidden[layer][1])
                        )
                    hidden[layer] = hidden_l
                    output.append(hidden[layer][0])

        # Reshape the output to match the expected format
        output = torch.stack(output, dim=0)
        if self.bidirectional:
            output = output.view(seq_len, batch_size, self.hidden_size * 2)
        else:
            output = output.view(seq_len, batch_size, self.hidden_size)

        # Compute the final hidden and cell states
        h_n = torch.stack([hidden[i][0] for i in range(self.num_layers * (2 if self.bidirectional else 1))])
        c_n = torch.stack([hidden[i][1] for i in range(self.num_layers * (2 if self.bidirectional else 1))])

        return output, (h_n, c_n)
