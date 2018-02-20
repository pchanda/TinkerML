

```python
import tensorflow as tf
import sys

'''   Tree to be modeled 

             [id:4
              LC:0
              RC:3]
             /    \
            /      \
           /        [id:3
          /          LC:1
         /           RC:2]
        /            /    \
       /            /      \
      /            /        \
     /            /          \
    /            /            \
  the          old           cat  
[id : 0       [id : 1       [id : 1
 LC :-1        LC :-1        LC :-1
 RC :-1]       RC :-1]       RC :-1]

'''

vocab = {'the': 0, 'old': 1, 'cat': 2} # word to integer map (node ids in the tree above)
node_words = ['the', 'old', 'cat', '', '']
is_leaf = [True, True, True, False, False] 
left_children = [-1, -1, -1, 1, 0]   # indices of left children nodes in the tree
right_children = [-1, -1, -1, 2, 3]  # indices of right children nodes in the tree

node_word_indices = [vocab[word] if word else -1 for word in node_words]
embed_size = 5

with tf.variable_scope('Embeddings'):
    embeddings = tf.get_variable('embeddings', [len(vocab), embed_size])

with tf.variable_scope('Composition'):
    W1 = tf.get_variable('W1',[2 * embed_size, embed_size])
    b1 = tf.get_variable('b1', [1, embed_size])

node_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True,clear_after_read=False, infer_shape=False)


def embed_word(word_index):
    return tf.expand_dims(tf.gather(embeddings, word_index), 0)


def combine_children(left_tensor, right_tensor):
    return tf.nn.relu(tf.matmul(tf.concat(axis=1, values=[left_tensor, right_tensor]), W1) + b1)


def loop_body(node_tensors, i):
    node_is_leaf = tf.gather(is_leaf, i)
    node_word_index = tf.gather(node_word_indices, i)
    left_child = tf.gather(left_children, i)   # index of left child
    right_child = tf.gather(right_children, i) # index of right child
    
    # function to be called when cond is True
    def f1(): return embed_word(node_word_index)  
    # function to be called when cond is False
    def f2(): return combine_children(node_tensors.read(left_child),node_tensors.read(right_child)) 
    
    # new_node_tensor is either a tensor from a leaf node, 
    # or tensor obtained by combining left and right child tensors for a non-leaf node.
    new_node_tensor = tf.cond( node_is_leaf, f1, f2) # cond = node_is_leaf

    node_tensors = node_tensors.write(i, new_node_tensor)
    i = tf.add(i, 1)
    return node_tensors, i


def loop_cond(node_tensors, i):
    return tf.less(i, tf.squeeze(tf.shape(is_leaf)))


node_tensors, i = tf.while_loop(loop_cond, loop_body, [node_tensors, 0],parallel_iterations=1)
z = node_tensors.stack()

```


```python
model = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(model)
    node_tensors_,i_ = sess.run([z,i])
    print('Completed ',i_,' loops')
    print('Node Tensors : ',node_tensors_)
```

    WARNING:tensorflow:From C:\local\Anaconda3-4.1.1-Windows-x86_64\lib\site-packages\tensorflow\python\util\tf_should_use.py:107: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    Completed  5  loops
    Node Tensors :  [[[-0.60116285  0.67330247  0.46631962  0.00616211  0.61571413]]
    
     [[-0.03535682  0.65925997 -0.46702185 -0.70694923  0.31417328]]
    
     [[ 0.62857825 -0.07381994  0.70534652 -0.80007017  0.4966436 ]]
    
     [[ 0.62187374  0.          0.00622888  0.23077905  0.36061448]]
    
     [[ 0.82129437  0.          0.          1.02332771  0.06781948]]]
    
