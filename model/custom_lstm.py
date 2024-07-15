import numpy as np
import torch
import torch.nn as nn

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.W_f = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) / np.sqrt(self.input_size + self.hidden_size)
        self.b_f = np.zeros((self.hidden_size, 1))
        self.W_i = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) / np.sqrt(self.input_size + self.hidden_size)
        self.b_i = np.zeros((self.hidden_size, 1))
        self.W_c = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) / np.sqrt(self.input_size + self.hidden_size)
        self.b_c = np.zeros((self.hidden_size, 1))
        self.W_o = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) / np.sqrt(self.input_size + self.hidden_size)
        self.b_o = np.zeros((self.hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.concatenate((x, h_prev), axis=1)

        # Calculate gates
        f = self.sigmoid(np.dot(self.W_f, concat.T) + self.b_f)
        i = self.sigmoid(np.dot(self.W_i, concat.T) + self.b_i)
        c_tilde = np.tanh(np.dot(self.W_c, concat.T) + self.b_c)
        o = self.sigmoid(np.dot(self.W_o, concat.T) + self.b_o)

        # Update cell state and hidden state
        c_next = f * c_prev + i * c_tilde
        h_next = o * np.tanh(c_next)

        return h_next.T, c_next.T

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.forward_lstm_layers = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size) for _ in range(self.num_layers)])
        self.backward_lstm_layers = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size) for _ in range(self.num_layers)])

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_length, _ = x.shape
        if h0 is None or c0 is None:
            h0 = np.zeros((self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size))
            c0 = np.zeros((self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size))

        h_forward = h0[:self.num_layers]
        c_forward = c0[:self.num_layers]
        h_backward = h0[self.num_layers:] if self.bidirectional else None
        c_backward = c0[self.num_layers:] if self.bidirectional else None

        output_forward = []
        output_backward = []

        for t in range(seq_length):
            # Forward pass
            for layer_idx in range(self.num_layers):
                h_forward[layer_idx], c_forward[layer_idx] = self.forward_lstm_layers[layer_idx].forward(x[:, t, :].T, h_forward[layer_idx], c_forward[layer_idx])

            output_forward.append(h_forward[-1])

            # Backward pass
            if self.bidirectional:
                for layer_idx in range(self.num_layers):
                    h_backward[layer_idx], c_backward[layer_idx] = self.backward_lstm_layers[layer_idx].forward(x[:, seq_length - 1 - t, :].T, h_backward[layer_idx], c_backward[layer_idx])

                output_backward.append(h_backward[-1])

        output_forward = np.stack(output_forward, axis=1)
        if self.bidirectional:
            output_backward = np.stack(output_backward[::-1], axis=1)
            output = np.concatenate((output_forward, output_backward), axis=2)
        else:
            output = output_forward

        if self.batch_first:
            output = output.transpose(1, 0, 2)

        return output, (h0, c0)
