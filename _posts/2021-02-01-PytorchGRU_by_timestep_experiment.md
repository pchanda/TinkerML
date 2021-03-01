---
layout: post
title: Pytorch - simple GRU experiment
categories: ['DeepLearning','Pytorch']
---

Create a simple GRU layer using pytorch. Feed a tensor of shape `batch_size`x`num_steps`x`input_size` and 
observe the GRU output. Next feed the same input tensor one time-step at a time ensuring that the 
previous timestep hidden state becomes initial state for the current timestep. The outputs should be same.


```python
import torch
from torch import nn
import numpy as np
```

```python
class SimpleGRU(nn.Module):
    
    def __init__(self, input_size, num_hiddens, num_layers=1):
        
        super(SimpleGRU, self).__init__()
        self.rnn = nn.GRU(input_size, num_hiddens, num_layers, batch_first=True)
        self.num_hiddens = num_hiddens
        
    def forward(self, X, hidden_state):
        # Input X: (`batch_size`, `num_steps`, `input_size`)
        # hidden state : (`num_layers`, `batch_size`, `num_hiddens`)  
        output, state = self.rnn(X, hidden_state)      
        # `output` shape: (`batch_size`, `num_steps`, `num_hiddens`)
        # `state` shape:  (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
    
    
def init_weights(m):
    #initialize biases and weights
    for name, param in m.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.constant(m._parameters[name],0.5)
```


```python
batch_size = 2
input_size = 3
num_hiddens = 4
num_steps = 5

gru = SimpleGRU(input_size, num_hiddens)
gru.eval()
```




    SimpleGRU(
      (rnn): GRU(3, 4, batch_first=True)
    )




```python
X = np.array( [ [ [1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5] ] , [ [5,5,5], [4,4,4], [3,3,3], [2,2,2], [1,1,1] ] ] )
X_t = torch.from_numpy(X).float()
print('Input data shape',X_t.shape)
weight = next(gru.parameters()).data
```

    Input data shape torch.Size([2, 5, 3])


## Feed the Input tensor (batch_size x num_steps x input_size) to the rnn


```python
h_0 = weight.new(1, batch_size, num_hiddens).zero_()
output, state = gru(X_t , h_0)

print('\nOutput')
#Permute to put timestep as the first dimension
output = output.permute(1,0,2)
print(output)
```

    
    Output
    tensor([[[-0.2491,  0.2479,  0.0823, -0.4551],
             [-0.0281,  0.0319,  0.0572, -0.5499]],
    
            [[-0.3922,  0.3499,  0.1561, -0.7259],
             [-0.0770,  0.0814,  0.1185, -0.7706]],
    
            [[-0.4632,  0.3994,  0.2157, -0.8596],
             [-0.1636,  0.1619,  0.1781, -0.8770]],
    
            [[-0.4985,  0.4244,  0.2613, -0.9263],
             [-0.3064,  0.2687,  0.2214, -0.9217]],
    
            [[-0.5163,  0.4368,  0.2947, -0.9605],
             [-0.4597,  0.3370,  0.2271, -0.8697]]], grad_fn=<PermuteBackward>)


## Now feed the inputs to the rnn one timestep at a time. The output should be same at each time step


```python
#Now feed the inputs to the rnn one timestep at a time. 
print('Output')
h_st = weight.new(1, batch_size, num_hiddens).zero_()
for timestep in range(num_steps):
    X_in = X_t[:,timestep,:].view(batch_size,1,input_size)
    #Previous timestep hidden state becomes initial state for this timestep
    op, h_st = gru(X_in , h_st)
    print(op)

```

    Output
    tensor([[[-0.2491,  0.2479,  0.0823, -0.4551]],
    
            [[-0.0281,  0.0319,  0.0572, -0.5499]]], grad_fn=<TransposeBackward1>)
    tensor([[[-0.3922,  0.3499,  0.1561, -0.7259]],
    
            [[-0.0770,  0.0814,  0.1185, -0.7706]]], grad_fn=<TransposeBackward1>)
    tensor([[[-0.4632,  0.3994,  0.2157, -0.8596]],
    
            [[-0.1636,  0.1619,  0.1781, -0.8770]]], grad_fn=<TransposeBackward1>)
    tensor([[[-0.4985,  0.4244,  0.2613, -0.9263]],
    
            [[-0.3064,  0.2687,  0.2214, -0.9217]]], grad_fn=<TransposeBackward1>)
    tensor([[[-0.5163,  0.4368,  0.2947, -0.9605]],
    
            [[-0.4597,  0.3370,  0.2271, -0.8697]]], grad_fn=<TransposeBackward1>)

