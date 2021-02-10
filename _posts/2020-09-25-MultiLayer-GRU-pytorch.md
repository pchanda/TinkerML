

```python
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import torch.nn.functional as F
```


```python
torch.cuda.is_available()
```




    True




```python
pad_char = '#'
all_chars = 'ABCDE'+pad_char
n_chars = len(all_chars)

# Find char index from all_chars, e.g. "a" = 0
def charToIndex(char):
    return all_chars.find(char)

# Just for demonstration, turn a char into a <1 x n_chars> Tensor
def charToTensor_one_hot(char):
    tensor = torch.zeros(1, n_chars)
    tensor[0][charToIndex(char)] = 1
    return tensor

def charToTensor(char):
    tensor = torch.zeros(1,dtype=torch.long)
    tensor[0] = charToIndex(char)
    return tensor

# Turn a line into a <line_length x 1 x n_chars>,
# or an array of one-hot char vectors
def seqToTensor_one_hot(seq):
    tensor = torch.zeros(len(seq),1, n_chars)
    for idx, char in enumerate(seq):
        tensor[idx][0][charToIndex(char)] = 1
    return tensor

def seqToTensor(seq):
    tensor = torch.zeros(len(seq), dtype=torch.long)
    for idx, char in enumerate(seq):
        tensor[idx] = int(charToIndex(char))
    return tensor


for ch in all_chars:
    print(ch,':',charToTensor_one_hot(ch))
```

    A : tensor([[1., 0., 0., 0., 0., 0.]])
    B : tensor([[0., 1., 0., 0., 0., 0.]])
    C : tensor([[0., 0., 1., 0., 0., 0.]])
    D : tensor([[0., 0., 0., 1., 0., 0.]])
    E : tensor([[0., 0., 0., 0., 1., 0.]])
    # : tensor([[0., 0., 0., 0., 0., 1.]])


## Prepare some tensor data for input : 2 character sequences


```python
sequences = ['AABC','AAAACC']

batch_size = len(sequences)

max_seqlen = 10

seq_tensors = []
for seq in sequences:
    seq_tensor = seqToTensor_one_hot(seq)
    seq_tensors.append(torch.squeeze(seq_tensor))
    
pad_char_tensor = charToTensor_one_hot(pad_char) #tensor corresponding to pad_char
            
batch_tensor = pad_char_tensor.repeat(batch_size, max_seqlen,1)

#print('batch_names_tensor',batch_names_tensor.shape)

for i,t in enumerate(seq_tensors):
    num_chars = t.shape[0]
    batch_tensor[i,-num_chars:,:] = t #Left padding is done with pad_char
    
print('input tensor shape=',batch_tensor.shape)
print('input tensor=\n',batch_tensor)
```

    input tensor shape= torch.Size([2, 10, 6])
    input tensor=
     tensor([[[0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 1.],
             [1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0.]]])


## Experiment 1: Simple GRU Layer (3 layers)
### Takes input an entire sequence


```python
class SimpleGRU(nn.Module):

    def __init__(self, input_size, num_layers=1, bidirectional=False, hidden_dim=10, printVars=False):
        super().__init__()
        
        print()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        
        self.num_layers = num_layers        
        self.num_directions = 2 if bidirectional==True else 1        
        
        #dim=2 as we are doing softmax across the last dimension of output_size
        self.softmax = nn.Softmax(dim=2)
        
        self.hidden = None
        self.printVars = printVars #run the print statements in forward ?
        
        #initialize biases and weights to some fixed value for testing
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.3)
            elif 'weight' in name:
                nn.init.constant(param, 0.3) #nn.init.xavier_normal(param)
                
                

    def init_hidden(self, batch_size):
        
        hidden_dim = self.hidden_dim
        weight = next(self.parameters()).data
        h_0 = weight.new(self.num_directions*self.num_layers, batch_size, hidden_dim).zero_()
        return h_0


    def forward(self, batch_of_words):
        
        batch_size = batch_of_words.shape[0]
        
        #This is stateless GRU, so hidden states are initialized for every forward pass.
        #The hidden states are not preserved across batches.
        self.hidden = self.init_hidden(batch_size)
        
        h_0 = self.hidden #initial hidden state, shape (num_direction*num_layers , batch_size, hidden_dim)
        x = batch_of_words
        
        if self.printVars:
            print('forward: h_0.shape',h_0.shape)
            print('forward: input to gru, x =',x.shape)
        
        output, self.hidden = self.gru(x, self.hidden)
        
        #output: output features h_t from the last layer of the GRU for each timestep=t
        #self.hidden : tensor containing the hidden state for the last timestep t = seq_len
        
        if self.printVars:
            print('\ngru_output=',output.shape,'\n',output) #output from final layer for all timesteps.
            print('\nh_out=',self.hidden.shape,'\n',self.hidden) #hidden state from last timestep for all layers
        
        
        return output, self.hidden
                    
        
```


```python
n_hidden = 4
n_layers = 3

print('batch_size=',batch_tensor.shape[0])
print('input_size =',n_chars)
print('n_hidden =',n_hidden)
print('n_layers = ',n_layers)


gru_rnn = SimpleGRU(n_chars, num_layers=n_layers, bidirectional=False,hidden_dim=n_hidden,printVars=False) 

print('input to gru = ',batch_tensor.shape)
emissions,h_n = gru_rnn(batch_tensor)

print('\n all_step_gru_output=',emissions.shape,'\n',emissions)
print('\n final_step_hidden=',h_n.shape,'\n',h_n)
```

    batch_size= 2
    input_size = 6
    n_hidden = 4
    n_layers =  3
    
    input to gru =  torch.Size([2, 10, 6])
    
     all_step_gru_output= torch.Size([2, 10, 4]) 
     tensor([[[0.1908, 0.1908, 0.1908, 0.1908],
             [0.3296, 0.3296, 0.3296, 0.3296],
             [0.4295, 0.4295, 0.4295, 0.4295],
             [0.5042, 0.5042, 0.5042, 0.5042],
             [0.5623, 0.5623, 0.5623, 0.5623],
             [0.6090, 0.6090, 0.6090, 0.6090],
             [0.6475, 0.6475, 0.6475, 0.6475],
             [0.6800, 0.6800, 0.6800, 0.6800],
             [0.7077, 0.7077, 0.7077, 0.7077],
             [0.7317, 0.7317, 0.7317, 0.7317]],
    
            [[0.1908, 0.1908, 0.1908, 0.1908],
             [0.3296, 0.3296, 0.3296, 0.3296],
             [0.4295, 0.4295, 0.4295, 0.4295],
             [0.5042, 0.5042, 0.5042, 0.5042],
             [0.5623, 0.5623, 0.5623, 0.5623],
             [0.6090, 0.6090, 0.6090, 0.6090],
             [0.6475, 0.6475, 0.6475, 0.6475],
             [0.6800, 0.6800, 0.6800, 0.6800],
             [0.7077, 0.7077, 0.7077, 0.7077],
             [0.7317, 0.7317, 0.7317, 0.7317]]], grad_fn=<TransposeBackward0>)
    
     final_step_hidden= torch.Size([3, 2, 4]) 
     tensor([[[0.7692, 0.7692, 0.7692, 0.7692],
             [0.7692, 0.7692, 0.7692, 0.7692]],
    
            [[0.7272, 0.7272, 0.7272, 0.7272],
             [0.7272, 0.7272, 0.7272, 0.7272]],
    
            [[0.7317, 0.7317, 0.7317, 0.7317],
             [0.7317, 0.7317, 0.7317, 0.7317]]], grad_fn=<StackBackward>)


    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:24: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:22: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.


## Experiment 2: Simple GRU Layer from individual cells (3 layers)


```python
class Stacked_GRU_Cells(nn.Module):
    """ Implements a three layer GRU cell with an output linear layer back to the size of the output categories"""

    def __init__(self, input_size, hidden_dim=10):
        super().__init__()
        
        self.gru_0 = nn.GRUCell(input_size, hidden_dim)
        self.gru_1 = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_2 = nn.GRUCell(hidden_dim, hidden_dim)
        
        #initialize biases and weights to some fixed value for testing 
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.3)
            elif 'weight' in name:
                nn.init.constant(param, 0.3)
                #nn.init.xavier_normal(param)

    def forward(self, x, h_in):
        h_out =  torch.zeros(h_in.size())
        
        h_out[0] = self.gru_0(x, h_in[0])
        h_out[1] = self.gru_1(h_out[0], h_in[1])
        h_out[2] = self.gru_2(h_out[1], h_in[2])
        
        x = h_out[2]
        return x, h_out
    
    
def forward_RNN_pass(gru_rnn, batch_tensor,hidden_dim):
    batch_size = batch_tensor.shape[0]
    seq_len = batch_tensor.shape[1]
    h_init = torch.zeros(3, batch_size, hidden_dim)
    print('Initial hidden state = ',h_init.shape)
    
    h = h_init
    #To gather outputs from all timesteps
    gru_out = torch.zeros([batch_size,seq_len,hidden_dim])
    
    for position in range(seq_len):
        logits, h = gru_rnn(batch_tensor[:, position, :], h)
        gru_out[:,position,:] = logits #store gru output from this timestep
        
    all_step_output = gru_out #output from final layer for all timesteps
    final_step_hidden = h #hidden state from final timestep for all layers
    return all_step_output, final_step_hidden
```


```python
n_hidden = 4
n_layers = 3 #fixed

print('batch_size=',batch_tensor.shape[0])
print('input_size =',n_chars)
print('n_hidden =',n_hidden)
print('n_layers = ',n_layers)

gru_rnn = Stacked_GRU_Cells(n_chars, hidden_dim=n_hidden)

print('input to gru = ',batch_tensor.shape)
final_step_output, final_step_hidden = forward_RNN_pass(gru_rnn,batch_tensor,n_hidden)

print('\n all_step_gru_output=',final_step_output.shape,'\n',final_step_output)
print('\n final_step_hidden=',final_step_hidden.shape,'\n',final_step_hidden)
```

    batch_size= 2
    input_size = 6
    n_hidden = 4
    n_layers =  3
    input to gru =  torch.Size([2, 10, 6])
    Initial hidden state =  torch.Size([3, 2, 4])
    
     all_step_gru_output= torch.Size([2, 10, 4]) 
     tensor([[[0.1908, 0.1908, 0.1908, 0.1908],
             [0.3296, 0.3296, 0.3296, 0.3296],
             [0.4295, 0.4295, 0.4295, 0.4295],
             [0.5042, 0.5042, 0.5042, 0.5042],
             [0.5623, 0.5623, 0.5623, 0.5623],
             [0.6090, 0.6090, 0.6090, 0.6090],
             [0.6475, 0.6475, 0.6475, 0.6475],
             [0.6800, 0.6800, 0.6800, 0.6800],
             [0.7077, 0.7077, 0.7077, 0.7077],
             [0.7317, 0.7317, 0.7317, 0.7317]],
    
            [[0.1908, 0.1908, 0.1908, 0.1908],
             [0.3296, 0.3296, 0.3296, 0.3296],
             [0.4295, 0.4295, 0.4295, 0.4295],
             [0.5042, 0.5042, 0.5042, 0.5042],
             [0.5623, 0.5623, 0.5623, 0.5623],
             [0.6090, 0.6090, 0.6090, 0.6090],
             [0.6475, 0.6475, 0.6475, 0.6475],
             [0.6800, 0.6800, 0.6800, 0.6800],
             [0.7077, 0.7077, 0.7077, 0.7077],
             [0.7317, 0.7317, 0.7317, 0.7317]]], grad_fn=<CopySlices>)
    
     final_step_hidden= torch.Size([3, 2, 4]) 
     tensor([[[0.7692, 0.7692, 0.7692, 0.7692],
             [0.7692, 0.7692, 0.7692, 0.7692]],
    
            [[0.7272, 0.7272, 0.7272, 0.7272],
             [0.7272, 0.7272, 0.7272, 0.7272]],
    
            [[0.7317, 0.7317, 0.7317, 0.7317],
             [0.7317, 0.7317, 0.7317, 0.7317]]], grad_fn=<CopySlices>)


    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:16: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
      app.launch_new_instance()
    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:14: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
      


## Compare these results with the output and hidden state tensors from Experiment 1
## They should be same
