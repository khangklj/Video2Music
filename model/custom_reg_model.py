import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )        
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights using Xavier initialization."""
        for layer in self.cell:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, h): 
        # x: (batch_size, input_dim) - a batch of tokens        
        # h: (batch_size, hidden_dim) - a batch of hidden states
            
        # update new hidden state
        new_h = self.cell(torch.cat((x, h), dim=1))
        
        # return new hidden state
        return new_h, new_h
class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    
        
        # Linear layer for input-to-hidden transformation
        self.xh = nn.Linear(input_dim, hidden_dim * 4, bias=bias)
        # Linear layer for hidden-to-hidden transformation
        self.hh = nn.Linear(hidden_dim, hidden_dim * 4, bias=bias)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights using Xavier initialization."""
        for layer in [self.xh, self.hh]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x, hx):
        # x: (batch_size, input_dim) - a batch of tokens        
        # hx: Tuple of (hidden_state, cell_state) each of size (batch_size, hidden_size)
        # return (hy, cy): Tuple of (hidden_state, cell_state) each of size (batch_size, hidden_dim)

        # Unpack hidden and cell states
        h, c = hx
        
        # Compute gates
        gates = self.xh(x) + self.hh(h)
        
        # Split gates into input, forget, cell, and output gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        # Apply activation functions to gates
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)
        
        # Update cell state
        cy = c * f_t + i_t * g_t
        
        # Compute new hidden state
        hy = o_t * torch.tanh(c)
        
        return (hy, cy)                     
        
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.xh = nn.Linear(input_dim, hidden_dim * 3, bias=bias)
        self.hh = nn.Linear(hidden_dim, hidden_dim * 3, bias=bias)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights using Xavier initialization."""
        for layer in [self.xh, self.hh]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, h):
        # x: (batch_size, input_dim) - a batch of tokens        
        # h: (batch_size, hidden_dim) - a batch of hidden states
        # return hy: (batch_size, hidden_dim)      
                
        x_t = self.xh(x)
        h_t = self.hh(h)

        # Split the transformations into reset, update, and new gate components
        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        # Compute reset gate
        reset_gate = torch.sigmoid(x_reset + h_reset)
        
        # Compute update gate
        update_gate = torch.sigmoid(x_upd + h_upd)
        
        # Compute candidate hidden state
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        # Compute final hidden state
        hy = (1 - update_gate) * new_gate + (update_gate * h)

        return hy           
            
def make_cell(input_dim, hidden_dim, cell_name='rnn'):
    if cell_name == 'rnn':
        return RNNCell(input_dim, hidden_dim)
    elif cell_name == 'lstm':
        return LSTMCell(input_dim, hidden_dim)
    elif cell_name == 'gru':
        return GRUCell(input_dim, hidden_dim)
    
# Many-to-Many RNN
class myRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, cell_name='rnn', num_layers=1, bidirectional=False):
        super(myRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_name = cell_name
        
        # Forward layers
        self.forward_layers = nn.ModuleList()
        self.forward_layers.append(make_cell(input_dim, hidden_dim, cell_name=cell_name))
        for i in range(self.num_layers - 1):
            self.forward_layers.append(make_cell(hidden_dim, hidden_dim, cell_name=cell_name))

        if bidirectional == True:            
            # Backward layers
            self.backward_layers = nn.ModuleList()
            self.backward_layers.append(make_cell(input_dim, hidden_dim, cell_name=cell_name))
            for i in range(self.num_layers - 1):
                self.backward_layers.append(make_cell(hidden_dim, hidden_dim, cell_name=cell_name))
            
        # Output layer
        self.regressor = nn.Linear(hidden_dim * (2 if bidirectional == True else 1), output_dim)
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim) - a batch of sequences
        # out: (batch_size, sequence_length, 2)        

        # Init        

        # h0 (num_layer, batch_size, hidden_dim)
        h0, c0 = (torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).cuda(), torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).cuda())

        output_forward_list = list()
        output_backward_list = list()       

        
        hidden_f = list()
        for layer in range(self.num_layers):
            hidden_f.append((h0[layer, :, :], c0[layer, :, :]) if self.cell_name == "lstm" else h0[layer, :, :])       
            
        # Forward
        for i in range(x.shape[1]): 
            for layer in range(self.num_layers):
                if self.cell_name == "lstm":
                    hidden_l = self.forward_layers[layer](x[:, i, :] if layer == 0 else hidden_f[layer-1][0], hidden_f[layer])
                else:
                    hidden_l = self.forward_layers[layer](x[:, i, :] if layer == 0 else hidden_f[layer-1], hidden_f[layer])

                hidden_f[layer] = hidden_l                

            output_forward_list.append(hidden_l[0].unsqueeze(1) if self.cell_name == "lstm" else hidden_l.unsqueeze(1))          
            
        # Backward
        if self.bidirectional == True:
            hidden_b = list()
            for layer in range(self.num_layers):
                hidden_b.append((h0[layer, :, :], c0[layer, :, :]) if self.cell_name == "lstm" else h0[layer, :, :]) 

            for i in reversed(range(x.shape[1])): 
                for layer in range(self.num_layers):
                    if self.cell_name == "lstm":
                        hidden_l = self.backward_layers[layer](x[:, i, :] if layer == 0 else hidden_b[layer-1][0], hidden_b[layer])                    
                    else:
                        hidden_l = self.backward_layers[layer](x[:, i, :] if layer == 0 else hidden_b[layer-1], hidden_b[layer])

                    hidden_b[layer] = hidden_l
                
                output_backward_list.append(hidden_l[0].unsqueeze(1) if self.cell_name == "lstm" else hidden_l.unsqueeze(1))


        if self.bidirectional == True:
            # f_out[i_th], b_out[i_th]: (batch_size, 1, hidden_dim) for i in (0 , seq_len)            
            # f_out, b_out: (batch_size, seq_length, hidden_dim)
            # out: (batch_size, seq_length, hidden_dim * 2)

            # f_out = torch.cat(output_forward_list, dim=1)
            # b_out = torch.cat(output_backward_list, dim=1)
            # out = torch.cat((f_out, b_out), dim=2)

            out = [ torch.cat((output_forward_list[i], output_backward_list[x.shape[1] - i - 1]), dim=2) for i in range(x.shape[1]) ]
            out = torch.cat(out, dim=1)

        else:
            # out: (batch_size, seq_length, hidden_dim)            
            out = torch.cat(output_forward_list, dim=1)        
        
        out = self.regressor(out)    
        
        return out
