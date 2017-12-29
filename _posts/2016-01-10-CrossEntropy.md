

```python
'''
Compare softmax_cross_entropy_with_logits and sparse_softmax_cross_entropy_with_logits.
Both functions should compute the same results. softmax_cross_entropy_with_logits computes
cross entropy using one-hot encoded labels. But sparse_softmax_cross_entropy_with_logits
computes the cross entropy directly on the sparse labels instead of converting them with
one-hot encoding.
'''

import tensorflow as tf
import numpy as np

# generate a batch of 10 labels. Each label is a number between 0 and 5.

batch_size = 10
dims = 5
labels = np.random.randint(0,dims,size=[batch_size])
print('labels=',labels)
```

    labels= [0 2 1 0 3 2 2 2 2 4]
    


```python
# define a session
sess = tf.Session()

logits = tf.random_uniform([batch_size,dims], maxval=3, dtype=tf.float32)
print('logits=',logits.eval(session=sess))

```

    logits= [[ 1.57716537  0.23490572  2.71676803  2.26921558  2.69666767]
     [ 0.99039853  2.07970381  1.14561582  1.66387045  2.92577028]
     [ 2.5476172   0.3643012   1.38589883  0.1428659   2.85149765]
     [ 0.43635714  2.56703377  0.28222704  2.741184    2.81704545]
     [ 2.63674664  1.97506392  1.899176    1.83821082  0.23850274]
     [ 2.51350951  1.6208986   0.9738543   0.6040746   2.99954128]
     [ 0.32376552  1.80199957  0.78665614  2.17520523  0.42237961]
     [ 1.77702069  2.16901112  2.05192995  0.56366837  1.78758466]
     [ 0.992329    0.08740246  0.94638884  2.2003231   1.93380189]
     [ 1.29982674  0.72387493  0.57704258  1.61114717  2.45938969]]
    


```python
one_hot_labels = tf.one_hot(labels, dims)
print('labels=',labels)
print('one-hot labels=',one_hot_labels.eval(session=sess))

```

    labels= [0 2 1 0 3 2 2 2 2 4]
    one-hot labels= [[ 1.  0.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  1.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  0.  1.  0.  0.]
     [ 0.  0.  0.  0.  1.]]
    


```python
cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=one_hot_labels)
cross_entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(labels))

ce_1, ce_2 = sess.run([cross_entropy_1, cross_entropy_2])
print('cross entropy 1 = ',ce_1)
print('cross entropy 2 = ',ce_2)
print(ce_1 == ce_2)

```

    cross entropy 1 =  [ 1.28006458  1.24166942  2.40354967  3.15004444  0.94967198  1.88296843
      1.14264107  2.96087337  1.95642412  2.20150757]
    cross entropy 2 =  [ 1.28006458  1.2416693   2.40354967  3.15004444  0.94967204  1.88296843
      1.14264107  2.96087337  1.95642412  2.20150757]
    [ True False  True  True False  True  True  True  True  True]
    


```python

```
