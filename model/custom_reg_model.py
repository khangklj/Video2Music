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

    def forward(self, x, h): 
        # x: (batch_size, input_dim) - a batch of tokens        
        # h: (batch_size, hidden_dim) - a batch of hidden states
            
        # update new hidden state
        new_h = self.cell(torch.cat((x, h), dim=1))
        
        # return new hidden state
        return new_h, new_h
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
        
    def forward(self, x, h):
        # x: (batch_size, input_dim) - a batch of tokens        
        # h: (batch_size, hidden_dim) - a batch of hidden states        
            
        x_h = torch.cat((x, h), dim=1)
            
        # reset gate
        r = self.reset_gate(x_h)
        
        # update gate
        z = self.update_gate(x_h)
        
        # candidate hidden state
        can_h = self.candidate(torch.cat((x, h * r), dim=1))
        
        # update new hidden state
        new_h = h * z + (1 - z) * can_h
        
        # return new hidden state
        return new_h, new_h
            
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
        last_seq_index = x.shape[1] - 1

        h0, c0 = (torch.zeros((x.shape[0], self.hidden_dim)).cuda(), torch.zeros((x.shape[0], self.hidden_dim)).cuda())

        output_forward = None
        output_backward = None

        output_forward_list, output_backward_list = ([], [])

        h_forward_list, h_backward_list = ([], [])
        
        temp_h_list = [None for _ in range(self.num_layers)]
        if self.cell_name == "lstm":
            c_forward_list, c_backward_list = ([], [])
            temp_c_list = [None for _ in range(self.num_layers)]
                  
        # Forward
        for i in range(x.shape[1]): 
            for j in range(self.num_layers):
                if i == 0 and j == 0:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](x[:, i, :], h0, c0)
                    else:
                        output_forward, h_forward = self.forward_layers[j](x[:, i, :], h0)
                elif i == 0:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](h_forward, h0, c0)
                    else:
                        output_forward, h_forward = self.forward_layers[j](h_forward, h0)
                elif j == 0:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](x[:, i, :], temp_h_list[j], temp_c_list[j])
                    else:
                        output_forward, h_forward = self.forward_layers[j](x[:, i, :], temp_h_list[j])
                else:
                    if self.cell_name == "lstm":
                        output_forward, (h_forward, c_forward) = self.forward_layers[j](h_forward, temp_h_list[j], temp_c_list[j])
                    else:
                        output_forward, h_forward = self.forward_layers[j](h_forward, temp_h_list[j])

                # Store temp hidden and cell state
                temp_h_list[j] = h_forward
                if self.cell_name == "lstm":
                    temp_c_list[j] = c_forward

                if i == last_seq_index:
                    print("First input sequence encounter")
                    h_forward_list.append(h_forward)
                    if self.cell_name == "lstm":
                        c_forward_list.append(c_forward)
            output_forward_list.append(output_forward.unsqueeze(1))           
            
        # Clear temp list
        temp_h_list = [None for _ in range(self.num_layers)]
        temp_c_list = [None for _ in range(self.num_layers)]


        # Backward
        if self.bidirectional == True:           
            for i in reversed(range(x.shape[1])): 
                for j in range(self.num_layers):
                    if i == last_seq_index and j == 0:
                        if self.cell_name == "lstm":
                            output_backward, (h_backward, c_backward) = self.backward_layers[j](x[:, i, :], h0, c0)
                        else:
                            output_backward, h_backward = self.backward_layers[j](x[:, i, :], h0)
                    elif i == last_seq_index:
                        if self.cell_name == "lstm":
                            output_backward, (h_backward, c_backward) = self.backward_layers[j](h_backward, h0, c0)
                        else:
                            output_backward, h_backward = self.backward_layers[j](h_backward, h0)
                    elif j == 0:
                        if self.cell_name == "lstm":
                            output_backward, (h_backward, c_backward) = self.backward_layers[j](x[:, i, :], temp_h_list[j], temp_c_list[j])
                        else:
                            output_backward, h_backward = self.backward_layers[j](x[:, i, :], temp_h_list[j])
                    else:
                        if self.cell_name == "lstm":
                            output_backward, (h_backward, c_backward) = self.backward_layers[j](h_backward, temp_h_list[j], temp_c_list[j])
                        else:
                            output_backward, h_backward = self.backward_layers[j](h_backward, temp_h_list[j])

                    # Store temp hidden and cell state
                    temp_h_list[j] = h_backward
                    if self.cell_name == "lstm":
                        temp_c_list[j] = c_backward

                    if i == 0:
                        print("Last input sequence encounter")
                        h_backward_list.append(h_backward)
                        if self.cell_name == "lstm":
                            c_backward_list.append(c_backward)
                output_backward_list.append(output_backward.unsqueeze(1))    

        if self.bidirectional == True:            
            # f_out, b_out: (batch_size, seq_length, hidden_dim)
            # out: (batch_size, seq_length, hidden_dim * 2)
            f_out = torch.cat(output_forward_list, dim=1)
            b_out = torch.cat(output_backward_list, dim=1)
            out = torch.cat((f_out, b_out), dim=2)
        else:
            # out: (batch_size, seq_length, hidden_dim)            
            out = torch.cat(output_forward_list, dim=1)        
        
        out = self.regressor(out)    
        
        return out
