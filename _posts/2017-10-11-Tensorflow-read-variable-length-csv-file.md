---
layout: post
title: Read csv file with variable number of fields using Tensorflow
---



```python

import tensorflow as tf
import numpy as np

filename = 'in.txt'
data_read = []
with open(filename,'r') as fin:
    for line in fin:
        x = line.strip()
        data_read  += [x]
        

data_tensor = tf.convert_to_tensor(data_read, dtype=tf.string)
sparse_tensor = tf.string_split(data_tensor,',') #sparse tensor
dense_tensor  = tf.sparse_to_dense(sparse_tensor.indices,sparse_tensor.dense_shape,sparse_tensor.values,default_value='NA')


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    X,Y = sess.run([data_tensor,dense_tensor])
    print('Input data with shape {0} = \n'.format(X.shape),X,'\n')
    print('Padded data with shape {0} = \n'.format(Y.shape),Y)
```

## Output 
The output array is nicely padded with 'NA' for the missing values. 

```python
    Input data with shape (5,) = 
     ['record_1,1,2,3,4' 'record_2,10,20,30' 'record_3,5,6,7,8,9,10,11,12'
     'record_4,41,42' 'record_5,100,200,300,400,500,600,700,800,900,1000'] 
    
    Padded data with shape (5, 11) = 
     [['record_1' '1' '2' '3' '4' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA']
     ['record_2' '10' '20' '30' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA']
     ['record_3' '5' '6' '7' '8' '9' '10' '11' '12' 'NA' 'NA']
     ['record_4' '41' '42' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA' 'NA']
     ['record_5' '100' '200' '300' '400' '500' '600' '700' '800'
      '900' '1000']]
```    
