---
layout: post
title: Tensorflow Multinomial Logistic Regression
categories: ['Tensorflow']
---

Implementing simple multiclass logistic regression with Tensorflow.


```python
# Multinomial (or multiclass) logistic regression (aka softmax regression) with tensorflow

import pandas as pd
import tensorflow as tf
import numpy as np

#define the static parameters for the program

NUM_FEATURES = 6 # each training sample has 6 features 
NUM_CLASSES = 3  # each training label can be one of three classes (0,1,2)
LEARNING_RATE = 0.01 # learning rate for optimizer
MAX_STEPS = 10000 # number of iterations to run

```


```python
#define placeholders for the input data matrix and labels. Label is just a 1-d vector of classes.
X = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES], name="input_matrix")
Y = tf.placeholder(tf.int32, shape=[None], name="input_labels")
```


```python
# define the logistic regression parameters to learn i.e. the weights and bias(intercept).
initializer = tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32)
weights = tf.get_variable(name="weights", shape=[NUM_FEATURES,NUM_CLASSES], initializer = initializer, dtype=tf.float32)
bias = tf.Variable(tf.zeros([1, NUM_CLASSES]), name="bias_or_intercept")
```


```python
# define the model, simple XW + intercept, exponentiation will be done by softmax_cross_entropy_with_logits below.
logits = tf.matmul(X,weights) + bias
```


```python
# convert the 1-d class vector to one-hot representation so that each row is a valid probability distribution.
one_hot_labels = tf.one_hot(Y,NUM_CLASSES)
```


```python
# loss, note that softmax_cross_entropy_with_logits will exponentiate the logits before normalization
cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=one_hot_labels)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
```


```python
# get prediction accuracies
probs = tf.nn.softmax(logits)
# predicted class is just the index of the largest probability.
preds = tf.argmax(probs,axis=1)
num_correct_predictions = tf.equal(preds, tf.argmax(one_hot_labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(num_correct_predictions, tf.float32))
```


```python
# optimization 
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train_step = optimizer.minimize(cross_entropy)
```


```python
# generate some data 
def gen_data(N,M,K):
  '''
  N : Number of samples 
  M = dimension of each sample (no. of features)
  K = No. of classes
  '''
  X = np.random.normal(size=[N,M])
  W = np.random.normal(loc=5,size=[M,K])
  logits = np.exp(np.matmul(X,W))
  probs = np.copy(logits)
  S = np.sum(logits,axis=1)
  for i in range(0,len(S)):
    probs[i,:] = logits[i,:]/S[i]

  classes = np.argmax(probs,axis=1)
  classes = np.expand_dims(classes,axis=1)

  data = np.copy(X)
  data = np.concatenate((data,classes),axis=1)

  np.savetxt("data.csv", data, delimiter=",",header="A,B,C,D,E,F,Class",comments="")

```


```python
# generate data
gen_data(1000,NUM_FEATURES,NUM_CLASSES) #should write a file 'data.csv'
    
# next define tensorflow session to run training the model with the data.
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # read the data just generated. 
    df = pd.read_csv('./data.csv')

    for step in range(MAX_STEPS):
       #read data using pandas
       data = np.copy(df.as_matrix())
       np.random.shuffle(data)
       X_val = data[:,:NUM_FEATURES]
       Y_val = data[:,-1]
       feed_dict_val = {X:X_val,Y:Y_val}
       _,loss_v,preds_v,accuracy_v = sess.run([train_step,cross_entropy_mean,preds,accuracy],feed_dict=feed_dict_val)
       if step % 500 == 0:
          print(step,loss_v,accuracy_v)

```
Training output: 

```python
    0 1.10255 0.272
    500 0.173238 0.98
    1000 0.117338 0.993
    1500 0.0910595 0.995
    2000 0.0746965 0.997
    2500 0.0631384 0.997
    3000 0.0543661 0.997
    3500 0.0473964 0.998
    4000 0.041684 0.998
    4500 0.0368968 0.998
    5000 0.0328183 0.998
    5500 0.0292986 0.998
    6000 0.0262295 0.998
    6500 0.0235297 0.999
    7000 0.0211367 0.999
    7500 0.0190013 0.999
    8000 0.0170849 0.999
    8500 0.0153565 0.999
    9000 0.0137921 0.999
    9500 0.0123726 0.999
```    
