---
layout: post
title: Comaparing using TimeDistributed and not with dense when return_sequences=True in Keras. The results are identical. 
categories: ['DeepLearning']
---

```python
from __future__ import print_function

import sys
import os
import pandas as pd
import numpy as np

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.models import Model, load_model, Sequential

import encoding

from keras import backend as K
import tensorflow as tf
```

    Using TensorFlow backend.



```python
n_chars = 4      # Input feature dimension
n_hidden = 5     # LSTM hidden state dimension, so output dimension of each timestep
n_categories = 2 # Output dimension of final dense layer.

inputs = Input(shape=(None, n_chars)) #n_chars = feature size
lstm = LSTM(n_hidden, return_sequences=True)(inputs) #return the output of all time steps

#Initialize the two
initializer = tf.keras.initializers.Constant(0.1)

#create two dense layers. 
#feed LSTM output of all timesteps to dense layer.
dense1 = Dense(n_categories,kernel_initializer=initializer)(lstm)

#feed LSTM output of all timesteps to dense layer with timedistributed.
dense2 = TimeDistributed(Dense(n_categories,kernel_initializer=initializer))(lstm)

model = Model(inputs=inputs, outputs=[dense1, dense2])

#you can initialize the LSTM hidden and cell states to 0 if you want. 

#hidden_states = K.variable(value=np.zeros([1, n_hidden]))
#cell_states = K.variable(value=np.zeros([1, n_hidden]))
#model.layers[1].states[0] = hidden_states
#model.layers[1].states[1] = cell_states 

```


```python
print(model.summary())

X = K.constant(np.ones([1,3,4]))

o1,o2 = model(X)

O1 = K.eval(o1)
O2 = K.eval(o2)

#Both dense1 and dense2 outputs should be same.

print('dense1 output\n',O1.shape,'\n',O1)

print('------------')

print('dense2 output\n',O2.shape,'\n',O2)


```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, 4)      0                                            
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   (None, None, 5)      200         input_1[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, None, 2)      12          lstm_1[0][0]                     
    __________________________________________________________________________________________________
    time_distributed_1 (TimeDistrib (None, None, 2)      12          lstm_1[0][0]                     
    ==================================================================================================
    Total params: 224
    Trainable params: 224
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    dense1 output
     (1, 3, 2) 
     [[[0.03041531 0.03041531]
      [0.05476438 0.05476438]
      [0.07346614 0.07346614]]]
    ------------
    dense2 output
     (1, 3, 2) 
     [[[0.03041531 0.03041531]
      [0.05476438 0.05476438]
      [0.07346614 0.07346614]]]

