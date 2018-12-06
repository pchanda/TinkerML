---
layout: post
title: Implementation of End-2-End Memory Network for Language Modeling
categories: ['Tensorflow','DeepLearning','NLP','Memory Networks']
---

Tensorflow implementation of [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895v4) for the language modeling task. 
I tried to name the variable as closely as possible to that in the paper following the equations to help understand the paper. Don't forget to change the "input_file" to your input file.
Some of the ideas are borrowed from earlier [implementation](https://github.com/carpedm20/MemN2N-tensorflow). The python notebook of the code can be found [here](https://github.com/pchanda/pchanda.github.io/blob/master/data/Mem-eNd-2-eNd.ipynb)

## Model parameters and input configurations

```python
# code for Mem-N-to-N for language modelling.

import numpy as np
import os
import math
import tensorflow as tf
import sys
import random
from collections import Counter

input_file = 'ptb.train.txt' #change this to your file of input.
config = {
        'batch_size'    : 128,     # batch_size
        'emb_dim'       : 150,     # embedding dimension for words
        'mem_size'      : 100,     # memory size
        'init_q'        : 0.1, 
        'n_epochs'      : 50,     # no. of epochs
        'n_hops'        : 6,     # no. of hops in memory
        'n_words'       : None,
        'init_lr'       : 0.001, # initial learning rate
        'std_dev'       : 0.05,
        'lin_dim'       : 75,      # no. of units to have linear activation
        'max_grad_norm' : 50     #clip gradients to this norm.
}

# read words and convert it to unique integers (from https://github.com/carpedm20/MemN2N-tensorflow/)
def read_data(fname, count, word2idx):
    with open(fname) as f:
        lines = f.readlines()
    words = []
    for line in lines:
        words.extend(line.split())
    if len(count) == 0:
        count.append(['<eos>', 0])
    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())
    if len(word2idx) == 0:
        word2idx['<eos>'] = 0
    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])
    print("Read %s words from %s" % (len(data), fname))
    return data

count = list()
word2idx = dict()
train_data = read_data(input_file, count, word2idx)
config['n_words'] = len(word2idx)

batch_size = config['batch_size']
e_dim = config['emb_dim']
l_dim = config['lin_dim']
mem_size = config['mem_size']
n_epochs = config['n_epochs']
n_hops = config['n_hops']
n_words = config['n_words'] = len(word2idx)
current_lr = config['init_lr']
std_dev = config['std_dev']
init_q = config['init_q']
max_grad_norm = config['max_grad_norm']
```
## Define the tensorflow model. 
The comments should explain the code.

```python
print('Defining the tensorflow model...')

# Define the tensorflow model. The Variable names are made to follow the paper as closely as possible.
input_q = tf.placeholder(tf.float32, shape=[None, e_dim],name="q") #the question q, will be set to all 0.1.
input_x = tf.placeholder(tf.int32, [None, mem_size], name="x") # the context word ids
input_time = tf.placeholder(tf.int32, [None, mem_size], name="time") # to lookup temporal encoding
input_y = tf.placeholder(tf.float32, [None, n_words], name="target") # id of next word to predict (target)

# Matrices for input memory representation
A = tf.Variable(tf.random_normal([n_words, e_dim], stddev=std_dev),name="A")   #embedding matrix A for input memory representation
T_A = tf.Variable(tf.random_normal([n_words, e_dim], stddev=std_dev),name="T_A") #embedding matrix for temporal encoding
# Input memory vectors : m_i = sum A_ij * x_ij + T_A_i
x_in_A   = tf.nn.embedding_lookup(A, input_x) # embedding lookup, shape: batch_size x mem_size x e_dim 
T_A_i = tf.nn.embedding_lookup(T_A, input_time) #T_A(i), shape: batch_size x mem_size x e_dim
mem_in = tf.add(x_in_A, T_A_i) #input memory vectors m_i, shape: batch_size x mem_size x e_dim

# Matrices for output memory representation
C = tf.Variable(tf.random_normal([n_words, e_dim], stddev=std_dev),name="C") #embedding matrix C for output memory representation
T_C = tf.Variable(tf.random_normal([n_words, e_dim], stddev=std_dev),name="T_C") #embedding matrix for temporal encoding
# Output memory vectors : c_i = sum C_ij * x_ij + T_C_i
x_in_C   = tf.nn.embedding_lookup(C, input_x) # embedding lookup, shape: batch_size x mem_size x e_dim
T_C_i = tf.nn.embedding_lookup(T_C, input_time) #T_C(i), shape: batch_size x mem_size x e_dim
mem_out = tf.add(x_in_C, T_C_i) #output memory vectors c_i, shape: batch_size x mem_size x e_dim

# For linear mapping of input u between hops
Hw = tf.Variable(tf.random_normal([e_dim, e_dim], stddev=std_dev),name="Hw")
Hb = tf.Variable(tf.random_normal([e_dim], stddev=std_dev),name="Hb")

u_k = input_q #initialize u_k for first hop in memory, shape : batch_size x edim

for k in range(n_hops): #k indexes the hops in memory
    print('hop in memory :',k,' input u_k:',u_k)
    u_k_3d = tf.reshape(u_k, [-1, e_dim, 1]) # reshape to shape: batch_size x e_dim x 1
    
    # p_i = Softmax(u^T m_i) (equation 1)
    probs = tf.nn.softmax(tf.matmul(mem_in, u_k_3d)) # shape: batch_size x mem_size x 1 
    
    # o = sum p_i c_i (equation 2)
    o_k = tf.matmul(mem_out, probs, transpose_a=True) # shape: batch_size x e_dim x 1
    o_k_2d = tf.reshape(o_k, [-1, e_dim]) # shape: batch_size x e_dim
    
    #apply a linear mapping H to u : u_mapped = Hw u + Hb 
    u_k_mapped = tf.add(tf.matmul(u_k,Hw),Hb)
    
    # u_(k+1) = u_k + o_k (equation 4)
    u_k_next_hop = tf.add(u_k_mapped,o_k_2d)
    
    #apply ReLU to a slice of the units, rest of the unit activations are linear.  
    u_k_next_hop_linear = tf.slice(u_k_next_hop, [0,0], [-1,l_dim]) #slice of u_k_next_hop to have linear activations
    u_k_next_hop_relu = tf.slice(u_k_next_hop, [0,l_dim], [-1,e_dim - l_dim]) # remaining slice to have ReLU activations
    u_k_next_hop_relu = tf.nn.relu(u_k_next_hop_relu)
    u_k_next_hop = tf.concat(axis=1, values=[u_k_next_hop_linear,u_k_next_hop_relu])
    u_k = u_k_next_hop #update u_k for the next hop in memory
    print('-------------')
    
W = tf.Variable(tf.random_normal([n_words, e_dim], stddev=std_dev),name="W") # final weight matrix W as in the paper.
a_hat = tf.matmul(u_k, W, transpose_b=True)  # shape : batch_size x n_words (equation 3), the output logits.

print('Model specification complete...')
```

## Define the ops for model optimization.


```python
print('Defining the ops for model optimization ...')
#Define the ops to estimate loss and optimize the above model. 

#change the softmax_cross_entropy_with_logits_v2 to softmax_cross_entropy_with_logits for older versions of tensorflow.
model_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=a_hat, labels=input_y)

lr = tf.Variable(current_lr)
opt = tf.train.GradientDescentOptimizer(lr) #optimizer

params = [A, T_A, C, T_C, Hw, Hb, W] #list of Variables to optimize
# get a List of (gradient, variable) pairs as returned by compute_gradients(...)
grads_and_vars = opt.compute_gradients(model_loss,params)

#clip the gradients using l2 norm of each variable separately, not used.
#clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], max_grad_norm), gv[1]) for gv in grads_and_vars] 

# Better: clip the gradients using l2 norm of the whole gradient of all variables. 
all_gradients = [gv[0] for gv in grads_and_vars]
clipped_grads_global = tf.clip_by_global_norm(all_gradients,max_grad_norm)[0] #should be a list of clipped tensors
clipped_grads_and_vars_global = [(clipped_grads_global[i],gv[1]) for i,gv in enumerate(grads_and_vars)]

optim = opt.apply_gradients(clipped_grads_and_vars_global)
```


## Define the data structures to provide data input to the model. Create and run session for training.


```python
# Define the data structures to provide data input to the model.
q = np.ndarray([batch_size, e_dim], dtype=np.float32)
x = np.ndarray([batch_size, mem_size])
time = np.ndarray([batch_size, mem_size], dtype=np.int32)
target = np.zeros([batch_size, n_words]) # each word is one-hot-encoded

q.fill(init_q) # fill with all 0.1

for t in range(mem_size):
    time[:,t].fill(t)

def train_one_epoch(epoch_no,sess,data):    
    # No. of loops in one epoch
    N = int(math.ceil(len(data) / batch_size))
    total_loss = 0
    
    for idx in range(1,N+1):
        target.fill(0)
        
        for b in range(batch_size):
            t_idx = random.randrange(mem_size, len(data)) #choose a word index beyond mem_size. 
            target[b][data[t_idx]] = 1 #set the word at the chosen index to be the target word to predict.
            # the context
            x[b] = data[t_idx - mem_size : t_idx] #set to the mem_size words preceeding the target word.
            
        f_dict = {
            input_q: q, 
            input_x: x, 
            input_time: time, 
            input_y: target 
        }
        _, batch_loss = sess.run([optim,model_loss],feed_dict=f_dict)
        total_loss += np.sum(batch_loss)
        cost = total_loss/(idx*batch_size)
        print('epoch=',epoch_no,' batch=',idx,' avg_loss=',cost)
        
    cost = total_loss/(N*batch_size)    
    print('epoch=',epoch_no,' avg_loss=',cost, "epoch perplexity=",np.exp(cost))    
         
# Define session to run the model with data            
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch_no in range(n_epochs):
        print('Running epoch = ',epoch_no)
        train_one_epoch(epoch_no,sess,train_data)

```

Happy training :)
