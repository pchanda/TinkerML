---
layout: post
title: Tensorflow TensorArray Simple Example
categories: ['Tensorflow']
---

A small example on how to use Tensorflow TensorArray. 


```python
import numpy as np
import tensorflow as tf    

matrix = tf.placeholder(tf.int32, shape=(5, 3), name="input_matrix")
matrix_rows = tf.shape(matrix)[0] #should be 5

# each element of the tensor_array corresponds to each row of the matrix
ta = tf.TensorArray(dtype=tf.int32, size=matrix_rows)

init_state = (0, ta)

#Also can write as :-   condition = lambda i, _: i < matrix_rows
def condition(i,ta):
    return (i < matrix_rows)

#Also can write as :-   body = lambda i, ta: (i + 1, ta.write(i, matrix[i] * (i+1)))
def body(i,ta):    
    ta = ta.write(i, matrix[i] * (i+1)) # at index i of the tensor_array, write (i+1) * matrix_row[i]  
    i = tf.add(i,1) # do this for all the elements of the tensor_array
    return i,ta

n, ta_final = tf.while_loop(condition, body, init_state)

#get the final result
ta_final_result = ta_final.stack()

#run the graph
with tf.Session() as sess:
    # print the output of ta_final_result
    a,b = sess.run([n,ta_final_result], feed_dict={matrix: np.ones(shape=(5,3), dtype=np.int32)})
    print('no of loops completed = ',a)
    print('Final content of tensorarray = ',b)
    
```

Output:

```python
    no of loops completed =  5
    Final content of tensorarray =  [[1 1 1]
     [2 2 2]
     [3 3 3]
     [4 4 4]
     [5 5 5]]
```    
