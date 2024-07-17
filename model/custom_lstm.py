import torch
from torch import nn
import torch.nn.functional as F

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.forget_input, self.forget_hidden, self.forget_bias = self.create_gate_parameters()
        self.input_input, self.input_hidden, self.input_bias = self.create_gate_parameters()
        self.output_input, self.output_hidden, self.output_bias = self.create_gate_parameters()
        self.cell_input, self.cell_hidden, self.cell_bias = self.create_gate_parameters()

    def create_gate_parameters(self):
        input_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.input_dim))
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(input_weights)
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return input_weights, hidden_weights, bias

    def forward(self, x, h, c):
        # x has shape [batch_size, input_size]
        # h and c each has shape [batch_size, hidden_dim]
        forget_gate = F.sigmoid((x @ self.forget_input) + (h @ self.forget_hidden) + self.forget_bias)
        input_gate = F.sigmoid((x @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
        output_gate = F.sigmoid((x @ self.output_input) + (h @ self.output_hidden) + self.output_bias)
        input_activations = F.tanh((x @ self.cell_input) + (h @ self.cell_hidden) + self.cell_bias)
        c = (forget_gate * c) + (input_gate * input_activations)
        h = F.tanh(c) * output_gate
        return h, c

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers, self.bidirectional = input_dim, hidden_dim, num_layers, bidirectional
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)       

    def forward(self, x):
        # x has shape [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape
        if torch.cuda.is_available():
            h = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim).cuda()
            c = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim).cuda()
        else:
            h = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim)
            c = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim)

        output_hiddens, output_cells = [], []
        for i in range(self.num_layers):
            layer_outputs = []
            for j in range(seq_len):
                h_i, c_i = self.layers[i](x[:, j, :], h[i], c[i])
                layer_outputs.append(h_i.unsqueeze(1))
            output_hidden = torch.concat(layer_outputs, dim=1)
            output_hiddens.append(output_hidden)
            output_cells.append(c)
        output_hidden = torch.concat(output_hiddens, dim=2)
        output_cell = torch.concat(output_cells, dim=2)
        output_hidden = self.dropout(output_hidden)        
        return output_hidden, (output_hidden, output_cell)
