import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.xh = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM cell.

        Args:
            input: Input tensor of shape (seq_len, batch_size, input_size) if batch_first is False,
                   or (batch_size, seq_len, input_size) if batch_first is True.
            hx: Tuple of (hidden state, cell state), each of shape (batch_size, hidden_size)

        Returns:
            hy: New hidden state tensor of shape (batch_size, hidden_size)
            cy: New cell state tensor of shape (batch_size, hidden_size)
        """
        # Initialize hidden and cell states if not provided
        if hx is None:
            if self.batch_first:
                hx = input.new_zeros(input.size(0), self.hidden_size)
            else:
                hx = input.new_zeros(input.size(1), self.hidden_size)
            hx = (hx, hx)

        # Unpack hidden and cell states
        hx, cx = hx

        # Print the shapes of the input and the weight matrices
        print(f"Input shape: {input.shape}")
        print(f"xh weight shape: {hx.shape}")

        # Compute gates
        if self.batch_first:
            gates = self.xh(input) + self.hh(hx)
        else:
            gates = self.xh(input.transpose(0, 1)) + self.hh(hx)

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
    def __init__(self, input_size, hidden_size, num_layers, bias=True, bidirectional=False, batch_first=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Create forward and backward LSTM cells
        self.forward_lstm_cell_list = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias) for i in range(num_layers)])
        if bidirectional:
            self.backward_lstm_cell_list = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias) for i in range(num_layers)])

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM.
        
        Args:
            input: Input tensor of shape (sequence length, batch_size, input_size) if batch_first=False
                   or (batch_size, sequence length, input_size) if batch_first=True
            hx: Initial hidden state and cell state (optional)
        
        Returns:
            out: Output tensor of shape (sequence length, batch_size, output_size) if batch_first=False
                 or (batch_size, sequence length, output_size) if batch_first=True
            (h_n, c_n): Final hidden state and cell state
                h_n: tensor of shape (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
                c_n: tensor of shape (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
        """
        # Initialize hidden state and cell state if not provided
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), input.size(1 if self.batch_first else 0), self.hidden_size).cuda()
                c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), input.size(1 if self.batch_first else 0), self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), input.size(1 if self.batch_first else 0), self.hidden_size)
                c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), input.size(1 if self.batch_first else 0), self.hidden_size)
        else:
            h0, c0 = hx

        outs = []
        forward_hidden = list()
        forward_cell = list()
        backward_hidden = list()
        backward_cell = list()

        # Initialize hidden and cell states for each layer
        for layer in range(self.num_layers):
            forward_hidden.append(h0[layer, :, :])
            forward_cell.append(c0[layer, :, :])
            if self.bidirectional:
                backward_hidden.append(h0[layer + self.num_layers, :, :])
                backward_cell.append(c0[layer + self.num_layers, :, :])

        # Process each time step
        sequence_length = input.size(0 if self.batch_first else 1)
        for t in range(sequence_length):
            # Process each forward layer
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the sequence
                    forward_hidden_l, forward_cell_l = self.forward_lstm_cell_list[layer](
                        input[t, :, :] if self.batch_first else input[t, :, :],
                        (forward_hidden[layer], forward_cell[layer])
                    )
                else:
                    # Subsequent layers take input from the previous layer
                    forward_hidden_l, forward_cell_l = self.forward_lstm_cell_list[layer](
                        forward_hidden[layer - 1],
                        (forward_hidden[layer], forward_cell[layer])
                    )
                forward_hidden[layer] = forward_hidden_l
                forward_cell[layer] = forward_cell_l

            # Process each backward layer if bidirectional
            if self.bidirectional:
                for layer in range(self.num_layers):
                    if layer == 0:
                        # First layer takes input from the sequence
                        backward_hidden_l, backward_cell_l = self.backward_lstm_cell_list[layer](
                            input[sequence_length - 1 - t, :, :] if self.batch_first else input[sequence_length - 1 - t, :, :],
                            (backward_hidden[layer], backward_cell[layer])
                        )
                    else:
                        # Subsequent layers take input from the previous layer
                        backward_hidden_l, backward_cell_l = self.backward_lstm_cell_list[layer](
                            backward_hidden[layer - 1],
                            (backward_hidden[layer], backward_cell[layer])
                        )
                    backward_hidden[layer] = backward_hidden_l
                    backward_cell[layer] = backward_cell_l

            # Concatenate forward and backward outputs
            if self.bidirectional:
                outs.append(torch.cat((forward_hidden[-1], backward_hidden[-1]), dim=1))
            else:
                outs.append(forward_hidden[-1])

        # Stack the outputs and return
        out = torch.stack(outs, dim=0 if self.batch_first else 1)

        # Collect the final hidden and cell states
        h_n = torch.stack(forward_hidden + backward_hidden, dim=0)
        c_n = torch.stack(forward_cell + backward_cell, dim=0)

        return out, (h_n, c_n)
