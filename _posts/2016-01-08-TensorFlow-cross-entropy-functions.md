---
layout: post
title: Comparing Cross Entropy Functions: softmax\_cross_entropy\_with\_logits , sparse\_softmax\_cross\_entropy\_with\_logits
published: true
categories: ['Tensorflow']
---

Compare softmax_cross_entropy_with_logits and sparse_softmax_cross_entropy_with_logits. Both functions should compute the same results. softmax_cross_entropy_with_logits computes the cross entropy using one-hot encoded labels. But sparse_softmax_cross_entropy_with_logits computes the cross entropy directly on the sparse labels instead of converting them with
one-hot encoding.


```python
import tensorflow as tf
import numpy as np

# generate a batch of 10 labels. Each label is a number between 0 and 5.

batch_size = 10
dims = 5
labels = np.random.randint(0,dims,size=[batch_size])
print('labels=',labels)
```

Output :
```python
    labels= [4 4 4 2 0 3 2 3 0 3]
```    


```python
# define a session
sess = tf.Session()

logits = tf.random_uniform([batch_size,dims], maxval=3, dtype=tf.float32)
print('logits=',logits.eval(session=sess))

```
Output :
```python
    logits= [[ 0.84077489  1.63235271  0.0433495   2.39548254  0.18592823]
     [ 2.75065231  2.68303108  1.14573097  2.05643392  0.06653416]
     [ 2.1297276   1.68026018  1.73126042  0.33460987  0.2684083 ]
     [ 2.6060648   0.31207395  0.1748203   1.3421334   2.75206184]
     [ 0.15627015  2.41854954  2.58276772  1.09528756  1.66697431]
     [ 2.50246286  2.27673841  2.87560749  0.11324608  1.99942732]
     [ 1.33319628  0.51725185  1.51783133  1.60997236  0.86610389]
     [ 1.83224201  2.89943123  1.91320646  1.18612969  0.19320595]
     [ 0.86170614  1.22675371  1.79078543  2.61112833  0.42935801]
     [ 2.78914213  0.11310089  2.37467527  0.28515887  1.8839035 ]]
```    


```python
one_hot_labels = tf.one_hot(labels, dims)
print('labels=',labels)
print('one-hot labels=',one_hot_labels.eval(session=sess))

```
Output :
```python
    labels= [4 4 4 2 0 3 2 3 0 3]
    one-hot labels= [[ 0.  0.  0.  0.  1.]
     [ 0.  0.  0.  0.  1.]
     [ 0.  0.  0.  0.  1.]
     [ 0.  0.  1.  0.  0.]
     [ 1.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  0.  0.  1.  0.]
     [ 1.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.]]
```    


```python
cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=one_hot_labels)
cross_entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(labels))

ce_1, ce_2 = sess.run([cross_entropy_1, cross_entropy_2])
print('cross entropy 1 = ',ce_1)
print('cross entropy 2 = ',ce_2)
print(ce_1 == ce_2)

```

Output :
```python
    cross entropy 1 =  [ 1.37368214  2.30929518  1.32668495  2.6403358   1.88928187  1.93529272
      0.63746631  1.03484416  1.61433876  2.217134  ]
    cross entropy 2 =  [ 1.37368214  2.30929518  1.32668495  2.6403358   1.88928187  1.93529272
      0.63746631  1.03484416  1.61433876  2.217134  ]
    [ True  True  True  True  True  True  True  True  True  True]
```    
