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
        input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
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
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=False, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers, self.bidirectional = input_dim, hidden_dim, num_layers, bidirectional
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LSTMCell(input_dim if len(self.layers) == 0 else hidden_dim * 2, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, input_dim)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, h=None):
        # x has shape [batch_size, seq_len, input_size]
        # h is a tuple containing h and c, each have shape [layer_num, batch_size, hidden_dim]
        if h is None:
            if torch.cuda.is_available():
                h = (torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda(),
                     torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
            else:
                h = (torch.zeros(self.num_layers, x.size(0), self.hidden_dim),
                     torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        hidden, cell = h
        all_hidden, all_cell = [], []
        for i in range(x.shape[1]):
            output_hidden, output_cell = self.layers[0](x[:, i], hidden[0], cell[0])
            all_hidden.append(output_hidden.unsqueeze(1))
            all_cell.append(output_cell.unsqueeze(1))
            new_hidden, new_cell = output_hidden.unsqueeze(0), output_cell.unsqueeze(0)
            for j in range(1, self.num_layers):
                output_hidden, output_cell = self.layers[j](self.dropout(output_hidden), hidden[j], cell[j])
                new_hidden = torch.cat((new_hidden, output_hidden.unsqueeze(0)), dim=0)
                new_cell = torch.cat((new_cell, output_cell.unsqueeze(0)), dim=0)
            hidden, cell = new_hidden, new_cell
        output_hidden = torch.cat(all_hidden, dim=1)
        output_cell = torch.cat(all_cell, dim=1)
        return self.dropout(output_hidden), (hidden, cell)
