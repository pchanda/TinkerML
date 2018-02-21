

```python
import tensorflow as tf
import numpy as np

filename = 'in.txt'
data_read = []
with open(filename,'r') as fin:
    for line in fin:
        x = line.strip()
        data_read  += [x]
        
#data_read = np.array(data_read)

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

    Input data with shape (5,) = 
     [b'record_1,1,2,3,4' b'record_2,10,20,30' b'record_3,5,6,7,8,9,10,11,12'
     b'record_4,41,42' b'record_5,100,200,300,400,500,600,700,800,900,1000'] 
    
    Padded data with shape (5, 11) = 
     [[b'record_1' b'1' b'2' b'3' b'4' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA']
     [b'record_2' b'10' b'20' b'30' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA']
     [b'record_3' b'5' b'6' b'7' b'8' b'9' b'10' b'11' b'12' b'NA' b'NA']
     [b'record_4' b'41' b'42' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA' b'NA']
     [b'record_5' b'100' b'200' b'300' b'400' b'500' b'600' b'700' b'800'
      b'900' b'1000']]
    
