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
        
        self.h = None # Hidden_state

    def forward(self, x): 
        # x: (batch_size, input_dim) - a batch of tokens
        
        # h: (batch_size, hidden_dim) - a batch of hidden states
        if self.h == None:
            self.h = torch.zeros((x.shape[0], self.hidden_dim)).cuda()
            
        # update new hidden state
        self.h = self.cell(torch.cat((x, self.h), dim=1))
        
        # return new hidden state
        return self.h
class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    
        
        self.forget_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.output_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, h, c):
        # x: (batch_size, input_dim) - a batch of tokens        
        # h: (batch_size, hidden_dim) - a batch of hidden states                     
        # c: (batch_size, hidden_dim) - a batch of cell states              
        
        x_h = torch.cat((x, h), dim=1)
                
        # forget gate
        f = self.forget_gate(x_h)        
        
        # input gate
        i = self.input_gate(x_h)
        
        # candidate hidden state
        can_mem = self.candidate(x_h)
        
        # output gate
        o = self.output_gate(x_h)
        
        # update new cell state
        new_c = c * f + can_mem * i
        
        # update new hidden state
        new_h = o * F.tanh(c)

        # return new hidden state
        return new_h, (new_h, new_c)
        
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.h = None # Hidden state
        
    def forward(self, x):
        # x: (batch_size, input_dim) - a batch of tokens
        
        # h: (batch_size, hidden_dim) - a batch of hidden states
        if self.h == None:
            self.h = torch.zeros((x.shape[0], self.hidden_dim)).cuda()
            
        x_h = torch.cat((x, self.h), dim=1)
            
        # reset gate
        r = self.reset_gate(x_h)
        
        # update gate
        z = self.update_gate(x_h)
        
        # candidate hidden state
        can_h = self.candidate(torch.cat((x, self.h * r), dim=1))
        
        # update new hidden state
        self.h = self.h * z + (1 - z) * can_h
        
        # return new hidden state
        return self.h
            
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
        # x: (batch_size, sequence_lenght, input_dim) - a batch of sequences
        print(f"x shape = {x.shape}")

        # Init
        h0, c0 = (torch.zeros((x.shape[0], self.hidden_dim)).cuda(), torch.zeros((x.shape[0], self.hidden_dim)).cuda())
        output_forward = None
        output_backward = None
        f_outputs = []
        b_outputs = []        
        
        for i in range(x.shape[1]):            
            for j in range(self.num_layers):
                # print(f"i = {i}, j = {j}")
                if j == 0:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](x[:, i, :], h0, c0)
                    else:
                        h_forward = self.forward_layers[j](x[:, i, :], h0)
                else:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](output_forward, h_forward, c_forward)
                    else:
                        h_forward = self.forward_layers[j](output_forward, h_forward)
                    
            # out_forward.append(h_forward)
            f_outputs.append(output_forward.unsqueeze(1))                   
            
        # Backward
        if self.bidirectional == True:           
            for i in reversed(range(x.shape[1])):
                for j in range(self.num_layers):
                    if j == 0:
                        if self.cell_name == "lstm":
                            output_backward, (h_forward, c_forward) = self.forward_layers[j](x[:, i, :], h0, c0)
                        else:
                            h_forward = self.forward_layers[j](x[:, i, :], h0)
                    else:
                        if self.cell_name == "lstm":
                            output_backward, (h_forward, c_forward) = self.forward_layers[j](output_backward, h_forward, c_forward)
                        else:
                            h_forward = self.forward_layers[j](output_backward, h_forward)
            
                b_outputs.append(output_backward.unsqueeze(1))     

        if self.bidirectional == True:            
            out = torch.cat((f_outputs, b_outputs), dim=1)
        else:            
            out = torch.cat(f_outputs, dim=1)
        
        print(f"Before regressor {out.shape}")
        out = self.regressor(out)
        print(f"After regressor {out.shape}")
        # output []
        return out
