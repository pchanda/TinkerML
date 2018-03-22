
This post shows how to implement a simple graph convolutional deep learning method to predict interfaces between protein residues, i.e. given a pair of interacting proteins, can we classify a pair of amino acid residues as interacting or not. This is based on the paper published in NIPS 2017 (Protein Interface Prediction using Graph Convolutional Networks) by Fout. et al.

The code here follows the published repository, and uses the [data] (https://zenodo.org/record/1127774#.WrJ5uWrwaM9). First I start with the description of the training data and then how we can build a graph convolutional model step by step using the data to solve the protein interface prediction problem. First load the training data 'train.cpkl' as :


```python
import pickle
import os
#assumes train.cpkl is placed in a local directory 'data'.
train_data_file = os.path.join('./data/','train.cpkl')   
train_list, train_data = pickle.load(open(train_data_file,'rb'),encoding='latin1')
print('No of molecules in Training = ', len(train_data))
molecule_pair = train_data[0]
print(molecule_pair.keys())
```
Output:
```
No of molecules in Training =  175
dict_keys(['l_vertex', 'complex_code', 'r_hood_indices', 'l_hood_indices', 'r_vertex', 'label', 'r_edge', 'l_edge'])
```    

Here, the training data has 175 pairs of molecules. Each molecule-pair is a dictionary with 8 keys. It has entries for 'ligand molecule' denoted by "l\_" and receptor molecule denoted by "r\_".


```python
print("l_vertex = [%d,%d]"%(molecule_pair['l_vertex'].shape))
print("l_edge = [%d,%d,%d]"%(molecule_pair['l_edge'].shape))
print("l_hood_indices = [%d,%d,%d]"%(molecule_pair['l_hood_indices'].shape))

print('--------------------------------------------------')

print("r_vertex = [%d,%d]"%(molecule_pair['r_vertex'].shape))
print("r_edge = [%d,%d,%d]"%(molecule_pair['r_edge'].shape))
print("r_hood_indices = [%d,%d,%d]"%(molecule_pair['r_hood_indices'].shape))

print('--------------------------------------------------')

print("label = [%d,%d]"%(molecule_pair['label'].shape))
print(molecule_pair['label'][0:5,:])
```
Output:
```
    l_vertex = [185,70]
    l_edge = [185,20,2]
    l_hood_indices = [185,20,1]
    --------------------------------------------------
    r_vertex = [362,70]
    r_edge = [362,20,2]
    r_hood_indices = [362,20,1]
    --------------------------------------------------
    label = [1683,3]
    [[111 358   1]
     [ 11 287   1]
     [115 301   1]
     [ 10 286   1]
     [ 34 352   1]]
```    

Thus for this molecule pair, the ligand molecule has 185 residues each with 20 neighbors (fixed in this dataset). The receptor molecule has 362 residues each with 20 neighbors again. The label indicates the +1/-1 status of every residue pair indicating whether the pair of residues are interacting or not. Note that not all pairs of residues are present in the label as the authors have chosen to downsample the negative examples for an overall ratio of 10:1 of negative to positive examples. With this description, lets move onto define the placeholder tensors for building the graph convolutional network.


```python
import tensorflow as tf

in_nv_dims = train_data[0]["l_vertex"].shape[-1]
in_ne_dims = train_data[0]["l_edge"].shape[-1]
in_nhood_size = train_data[0]["l_hood_indices"].shape[1]

in_vertex1 = tf.placeholder(tf.float32,[None,in_nv_dims],"vertex1")
in_vertex2 = tf.placeholder(tf.float32,[None,in_nv_dims],"vertex2")
in_edge1 = tf.placeholder(tf.float32,[None,in_nhood_size,in_ne_dims],"edge1")
in_edge2 = tf.placeholder(tf.float32,[None,in_nhood_size,in_ne_dims],"edge2")
in_hood_indices1 = tf.placeholder(tf.int32,[None,in_nhood_size,1],"hood_indices1")
in_hood_indices2 = tf.placeholder(tf.int32,[None,in_nhood_size,1],"hood_indices2")

input1 = in_vertex1, in_edge1, in_hood_indices1
input2 = in_vertex2, in_edge2, in_hood_indices2

examples = tf.placeholder(tf.int32,[None,2],"examples")
labels = tf.placeholder(tf.float32,[None],"labels")
dropout_keep_prob = tf.placeholder(tf.float32,shape=[],name="dropout_keep_prob")
```

I will describe the following network architecture : 
ligand-residue -->  ligand_graph_conv_layer1 -->  ligand_graph_conv_layer2   
                                                                         \ 
                                                                          --> merged --> dense1 --> dense2 --> prediction 
                                                                         / 
receptor-residue --> receptor_graph_conv_layer1 --> receptor_graph_conv_layer1
So input1 will represent the input to th network containing information for ligand residue of the pair, analogously, input2 will represent the input to th network containing information for receptor residue of the pair. We will use the following function to perform the graph convolution (e.g. in ligand_graph_conv_layer1 and receptor_graph_conv_layer2) :


```python
import numpy as np

def initializer(init, shape):  #helper function to initialize a tensor
    if init == "zero":
        return tf.zeros(shape)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)

def nonlinearity(nl): #helper function to determine the type of non-linearity
    if nl == "relu":
        return tf.nn.relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x
    
# the function that defines the ops for graph onvolution.    
def node_average_model(input, params, filters=None, dropout_keep_prob=1.0, trainable=True):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value
    if params is None:
        # create new weights
        Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc", trainable=trainable)  # (v_dims, filters)
        Wn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn", trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
    else:
        Wn, Wc = params["Wn"], params["Wc"]
        filters = Wc.get_shape()[-1].value
        b = params["b"]
    params = {"Wn": Wn, "Wc": Wc, "b": b}
    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    v_Wn = tf.matmul(vertices, Wn, name="v_Wn")  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wn, nh_indices), 1), tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)
    nonlin = nonlinearity("relu")
    sig = Zn + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params
```

The equation behind the tensor operations in the above function 'node_average_model' are in 
![an image alt text]({{ site.baseurl }}/images/GraphConvEq.png "Graph-Conv-Equation"){:height="30%" width="30%"}


```python
#layer 1
layer_no = 1
name = "ligand_graph_conv_layer1"
with tf.name_scope(name):
        output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
        input1 = output, in_edge1, in_hood_indices1

name = "receptor_graph_conv_layer1"
with tf.name_scope(name):
        output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
        input2 = output, in_edge2, in_hood_indices2

#layer 2
layer_no = 2
name = "ligand_graph_conv_layer2"
with tf.name_scope(name):
        output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
        input1 = output, in_edge1, in_hood_indices1

name = "receptor_graph_conv_layer2"
with tf.name_scope(name):
        output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
        input2 = output, in_edge2, in_hood_indices2
```

Note that the weights are shared between ligand\_graph\_conv_layer1 receptor\_graph\_conv\_layer1. Similarly weights are shared between ligand\_graph\_conv_layer2 receptor\_graph\_conv\_layer2. Next add the layers to merge the outputs from 'ligand\_graph\_conv\_layer2' and 'receptor\_graph\_conv\_layer2' - essentially concatenating the tensors along the feature dimension with both possible orders : (ligand,receptor) and (receptor,ligand). 


```python
# merged layers
layer_no = 3
name = "merged"
merge_input1 = input1[0] 
merge_input2 = input2[0]
with tf.name_scope(name):
    m_out1 = tf.gather(merge_input1, examples[:, 0])
    m_out2 = tf.gather(merge_input2, examples[:, 1])
    # concatenate in both possible orders : (ligand,receptor) and (receptor,ligand).
    output1 = tf.concat([m_out1, m_out2], axis=0)
    output2 = tf.concat([m_out2, m_out1], axis=0)
    merged_output = tf.concat((output1, output2), axis=1)
```

Next add the two densely connected layers using the merged output. Define the function for the dense layer:


```python
def dense(input, out_dims=None, dropout_keep_prob=1.0, nonlin=True, trainable=True):
    input = tf.nn.dropout(input, dropout_keep_prob)
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
    b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
    Z = tf.matmul(input, W) + b
    if(nonlin):
        nonlin = nonlinearity("relu")
        Z = nonlin(Z)
    Z = tf.nn.dropout(Z, dropout_keep_prob)
    return Z
```

And connect it with merged_output:


```python
# dense layer 1 
layer_no = 4
name = "dense1"
with tf.name_scope(name):
        dense1_output = dense(merged_output, out_dims=512, dropout_keep_prob=0.5, nonlin=True, trainable=True)

# dense layer 2
layer_no = 5
name = "dense2"
with tf.name_scope(name):
        dense2_output = dense(dense1_output, out_dims=1, dropout_keep_prob=0.5, nonlin=False, trainable=True)
```

Add the final layer, essentially averaging the predictions from both orders of combinations. See 
![an image alt text]({{ site.baseurl }}/images/GraphConvArch1.png "Architecture"){:height="30%" width="30%"}


```python
# add layer to get mean prediction across both the orders (ligand,receptor) and (receptor,ligand)
layer_no = 6
name = "do_prediction"
with tf.name_scope(name):
     preds = tf.reduce_mean(tf.stack(tf.split(dense2_output, 2)), 0)

pn_ratio = 0.1        
learning_rate = 0.05

#add loss op
# Loss and optimizer
with tf.name_scope("loss"):
     scale_vector = (pn_ratio * (labels - 1) / -2) + ((labels + 1) / 2)
     logits = tf.concat([-preds, preds], axis=1)
     labels_stacked = tf.stack([(labels - 1) / -2, (labels + 1) / 2], axis=1)
     loss = tf.losses.softmax_cross_entropy(labels_stacked, logits, weights=scale_vector)
     with tf.name_scope("optimizer"):
        # generate an op which trains the model
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

    C:\local\Anaconda3-4.1.1-Windows-x86_64\lib\site-packages\tensorflow\python\ops\gradients_impl.py:96: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    


```python
def build_feed_dict(minibatch):
   feed_dict = {
                    in_vertex1: minibatch["l_vertex"], in_edge1: minibatch["l_edge"],
                    in_vertex2: minibatch["r_vertex"], in_edge2: minibatch["r_edge"],
                    in_hood_indices1: minibatch["l_hood_indices"],
                    in_hood_indices2: minibatch["r_hood_indices"],
                    examples: minibatch["label"][:, :2],
                    labels: minibatch["label"][:, 2],
                    dropout_keep_prob: dropout_keep
   }
   return feed_dict


num_epochs =  1 #change this while real training.
minibatch_size = 128
dropout_keep = 0.5

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Training Graph Conv Model ...")

for epoch in range(0, num_epochs):
     """
     Trains model for one pass through training data, one protein at a time
     Each protein is split into minibatches of paired examples.
     Features for the entire protein is passed to model, but only a minibatch of examples are passed
     """
     prot_perm = np.random.permutation(len(train_data))
     ii = 0
     nn = 0
     avg_loss = 0
     # loop through each protein
     for protein in prot_perm:
         # extract just data for this protein to create minibatches.
         prot_data = train_data[protein]
         pair_examples = prot_data["label"]
         n = len(pair_examples)
         shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
         # loop through each minibatch or ligand-receptor pairs for this protein.
         for i in range(int(n / minibatch_size)):
             # extract data for this minibatch
             index = int(i * minibatch_size)
             example_pairs = pair_examples[shuffle_indices[index: index + minibatch_size]]
             minibatch = {} 
             #copy data to minibatch.
             for feature_type in prot_data:
                 if feature_type == "label":
                    minibatch["label"] = example_pairs
                 else:
                     minibatch[feature_type] = prot_data[feature_type]
             # train the model
             feed_dict = build_feed_dict(minibatch)
             _,loss_v = sess.run([train_op,loss], feed_dict=feed_dict)
             avg_loss += loss_v
             ii += 1
             print(ii," Batch loss = ",loss_v)
         nn += n
     print("Epoch_end =",epoch,", avg_loss = ",avg_loss/ii," nn = ",nn)
```

    Training Graph Conv Model ...
    1  Batch loss =  0.112723
    2  Batch loss =  0.118632
    3  Batch loss =  0.125916
    4  Batch loss =  0.162415
    5  Batch loss =  0.117011
    6  Batch loss =  0.112846
    7  Batch loss =  0.135485
    8  Batch loss =  0.119522
    9  Batch loss =  0.112589
    10  Batch loss =  0.143433
    11  Batch loss =  0.12488
    12  Batch loss =  0.122961
    13  Batch loss =  0.129658
    14  Batch loss =  0.131469
    15  Batch loss =  0.125211
    16  Batch loss =  0.139312
    17  Batch loss =  0.135089
    18  Batch loss =  0.135078
    19  Batch loss =  0.105462
    20  Batch loss =  0.113002
    21  Batch loss =  0.119641
    22  Batch loss =  0.150954
    23  Batch loss =  0.113552
    24  Batch loss =  0.154757
    25  Batch loss =  0.143836
    26  Batch loss =  0.107087
    27  Batch loss =  0.102819
    28  Batch loss =  0.14731
    29  Batch loss =  0.125966
    30  Batch loss =  0.133721
    31  Batch loss =  0.129521
    32  Batch loss =  0.0976463
    33  Batch loss =  0.117646
    34  Batch loss =  0.112419
    35  Batch loss =  0.125968
    36  Batch loss =  0.128602
    37  Batch loss =  0.14359
    38  Batch loss =  0.139197
    39  Batch loss =  0.109954
    40  Batch loss =  0.150609
    41  Batch loss =  0.110643
    42  Batch loss =  0.133368
    43  Batch loss =  0.119719
    44  Batch loss =  0.120153
    45  Batch loss =  0.131292
    46  Batch loss =  0.102735
    47  Batch loss =  0.127481
    48  Batch loss =  0.129746
    49  Batch loss =  0.128009
    50  Batch loss =  0.122334
    51  Batch loss =  0.131407
    52  Batch loss =  0.130154
    53  Batch loss =  0.1263
    54  Batch loss =  0.107479
    55  Batch loss =  0.131873
    56  Batch loss =  0.103274
    57  Batch loss =  0.127612
    58  Batch loss =  0.12548
    59  Batch loss =  0.127341
    60  Batch loss =  0.13273
    61  Batch loss =  0.151274
    62  Batch loss =  0.12298
    63  Batch loss =  0.140298
    64  Batch loss =  0.142882
    65  Batch loss =  0.140179
    66  Batch loss =  0.143005
    67  Batch loss =  0.0910168
    68  Batch loss =  0.125118
    69  Batch loss =  0.115503
    70  Batch loss =  0.129032
    71  Batch loss =  0.138347
    72  Batch loss =  0.104913
    73  Batch loss =  0.127579
    74  Batch loss =  0.133182
    75  Batch loss =  0.128452
    76  Batch loss =  0.106694
    77  Batch loss =  0.109749
    78  Batch loss =  0.155299
    79  Batch loss =  0.11459
    80  Batch loss =  0.118971
    81  Batch loss =  0.123413
    82  Batch loss =  0.132117
    83  Batch loss =  0.137846
    84  Batch loss =  0.131562
    85  Batch loss =  0.11585
    86  Batch loss =  0.111554
    87  Batch loss =  0.133289
    88  Batch loss =  0.144359
    89  Batch loss =  0.124313
    90  Batch loss =  0.146351
    91  Batch loss =  0.105612
    92  Batch loss =  0.136741
    93  Batch loss =  0.112244
    94  Batch loss =  0.135856
    95  Batch loss =  0.109173
    96  Batch loss =  0.113237
    97  Batch loss =  0.13422
    98  Batch loss =  0.130335
    99  Batch loss =  0.114828
    100  Batch loss =  0.123207
    101  Batch loss =  0.113455
    102  Batch loss =  0.13737
    103  Batch loss =  0.116505
    104  Batch loss =  0.127497
    105  Batch loss =  0.141288
    106  Batch loss =  0.0945996
    107  Batch loss =  0.123875
    108  Batch loss =  0.149487
    109  Batch loss =  0.108243
    110  Batch loss =  0.143293
    111  Batch loss =  0.119057
    112  Batch loss =  0.139804
    113  Batch loss =  0.128941
    114  Batch loss =  0.153747
    115  Batch loss =  0.121998
    116  Batch loss =  0.113878
    117  Batch loss =  0.0901378
    118  Batch loss =  0.147252
    119  Batch loss =  0.127487
    120  Batch loss =  0.117426
    121  Batch loss =  0.132588
    122  Batch loss =  0.0976086
    123  Batch loss =  0.126303
    124  Batch loss =  0.11436
    125  Batch loss =  0.144222
    126  Batch loss =  0.154531
    127  Batch loss =  0.128244
    128  Batch loss =  0.132711
    129  Batch loss =  0.121743
    130  Batch loss =  0.124462
    131  Batch loss =  0.0942012
    132  Batch loss =  0.127063
    133  Batch loss =  0.132181
    134  Batch loss =  0.142005
    135  Batch loss =  0.122829
    136  Batch loss =  0.140341
    137  Batch loss =  0.122431
    138  Batch loss =  0.114213
    139  Batch loss =  0.147699
    140  Batch loss =  0.121639
    141  Batch loss =  0.157541
    142  Batch loss =  0.129441
    143  Batch loss =  0.126008
    144  Batch loss =  0.149518
    145  Batch loss =  0.105663
    146  Batch loss =  0.0946201
    147  Batch loss =  0.11572
    148  Batch loss =  0.133595
    149  Batch loss =  0.106985
    150  Batch loss =  0.124599
    151  Batch loss =  0.171802
    152  Batch loss =  0.109453
    153  Batch loss =  0.120346
    154  Batch loss =  0.145639
    155  Batch loss =  0.107599
    156  Batch loss =  0.108338
    157  Batch loss =  0.152498
    158  Batch loss =  0.136628
    159  Batch loss =  0.0911053
    160  Batch loss =  0.112747
    161  Batch loss =  0.143374
    162  Batch loss =  0.12516
    163  Batch loss =  0.115421
    164  Batch loss =  0.132778
    165  Batch loss =  0.118317
    166  Batch loss =  0.123265
    167  Batch loss =  0.12243
    168  Batch loss =  0.156015
    169  Batch loss =  0.111732
    170  Batch loss =  0.131084
    171  Batch loss =  0.125385
    172  Batch loss =  0.114489
    173  Batch loss =  0.109037
    174  Batch loss =  0.142734
    175  Batch loss =  0.126374
    176  Batch loss =  0.122952
    177  Batch loss =  0.122473
    178  Batch loss =  0.12181
    179  Batch loss =  0.116114
    180  Batch loss =  0.148441
    181  Batch loss =  0.136903
    182  Batch loss =  0.104524
    183  Batch loss =  0.115036
    184  Batch loss =  0.132523
    185  Batch loss =  0.151824
    186  Batch loss =  0.102957
    187  Batch loss =  0.119657
    188  Batch loss =  0.120926
    189  Batch loss =  0.146649
    190  Batch loss =  0.139291
    191  Batch loss =  0.112789
    192  Batch loss =  0.134158
    193  Batch loss =  0.128348
    194  Batch loss =  0.138805
    195  Batch loss =  0.128692
    196  Batch loss =  0.118422
    197  Batch loss =  0.131117
    198  Batch loss =  0.134077
    199  Batch loss =  0.108832
    200  Batch loss =  0.105266
    201  Batch loss =  0.149131
    202  Batch loss =  0.121887
    203  Batch loss =  0.148044
    204  Batch loss =  0.126677
    205  Batch loss =  0.0830073
    206  Batch loss =  0.136651
    207  Batch loss =  0.135826
    208  Batch loss =  0.129278
    209  Batch loss =  0.142465
    210  Batch loss =  0.134016
    211  Batch loss =  0.112271
    212  Batch loss =  0.110705
    213  Batch loss =  0.132927
    214  Batch loss =  0.129177
    215  Batch loss =  0.114406
    216  Batch loss =  0.128163
    217  Batch loss =  0.116657
    218  Batch loss =  0.109973
    219  Batch loss =  0.124327
    220  Batch loss =  0.14077
    221  Batch loss =  0.11391
    222  Batch loss =  0.129053
    223  Batch loss =  0.146554
    224  Batch loss =  0.136961
    225  Batch loss =  0.110062
    226  Batch loss =  0.13711
    227  Batch loss =  0.135186
    228  Batch loss =  0.110945
    229  Batch loss =  0.104167
    230  Batch loss =  0.131179
    231  Batch loss =  0.136631
    232  Batch loss =  0.139425
    233  Batch loss =  0.134309
    234  Batch loss =  0.123445
    235  Batch loss =  0.102315
    236  Batch loss =  0.111375
    237  Batch loss =  0.103883
    238  Batch loss =  0.121621
    239  Batch loss =  0.132004
    240  Batch loss =  0.122441
    241  Batch loss =  0.116498
    242  Batch loss =  0.158253
    243  Batch loss =  0.133474
    244  Batch loss =  0.137083
    245  Batch loss =  0.128099
    246  Batch loss =  0.116571
    247  Batch loss =  0.152801
    248  Batch loss =  0.12017
    249  Batch loss =  0.132542
    250  Batch loss =  0.125515
    251  Batch loss =  0.12217
    252  Batch loss =  0.108992
    253  Batch loss =  0.126421
    254  Batch loss =  0.107528
    255  Batch loss =  0.124457
    256  Batch loss =  0.10868
    257  Batch loss =  0.145783
    258  Batch loss =  0.121082
    259  Batch loss =  0.153636
    260  Batch loss =  0.109316
    261  Batch loss =  0.116261
    262  Batch loss =  0.109776
    263  Batch loss =  0.123441
    264  Batch loss =  0.105888
    265  Batch loss =  0.114575
    266  Batch loss =  0.172647
    267  Batch loss =  0.143648
    268  Batch loss =  0.12092
    269  Batch loss =  0.123744
    270  Batch loss =  0.10496
    271  Batch loss =  0.148057
    272  Batch loss =  0.135142
    273  Batch loss =  0.140481
    274  Batch loss =  0.101506
    275  Batch loss =  0.133047
    276  Batch loss =  0.13722
    277  Batch loss =  0.127906
    278  Batch loss =  0.139121
    279  Batch loss =  0.114132
    280  Batch loss =  0.118856
    281  Batch loss =  0.10958
    282  Batch loss =  0.111821
    283  Batch loss =  0.141391
    284  Batch loss =  0.115819
    285  Batch loss =  0.119286
    286  Batch loss =  0.119778
    287  Batch loss =  0.156907
    288  Batch loss =  0.135099
    289  Batch loss =  0.123273
    290  Batch loss =  0.127166
    291  Batch loss =  0.108552
    292  Batch loss =  0.111786
    293  Batch loss =  0.126405
    294  Batch loss =  0.149084
    295  Batch loss =  0.152319
    296  Batch loss =  0.11906
    297  Batch loss =  0.12278
    298  Batch loss =  0.0965698
    299  Batch loss =  0.145954
    300  Batch loss =  0.110265
    301  Batch loss =  0.124922
    302  Batch loss =  0.113295
    303  Batch loss =  0.139471
    304  Batch loss =  0.139746
    305  Batch loss =  0.127998
    306  Batch loss =  0.152518
    307  Batch loss =  0.106823
    308  Batch loss =  0.103645
    309  Batch loss =  0.160446
    310  Batch loss =  0.129521
    311  Batch loss =  0.112677
    312  Batch loss =  0.109796
    313  Batch loss =  0.135862
    314  Batch loss =  0.105746
    315  Batch loss =  0.119608
    316  Batch loss =  0.108873
    317  Batch loss =  0.132077
    318  Batch loss =  0.138006
    319  Batch loss =  0.128394
    320  Batch loss =  0.134215
    321  Batch loss =  0.141045
    322  Batch loss =  0.138039
    323  Batch loss =  0.112189
    324  Batch loss =  0.120976
    325  Batch loss =  0.131858
    326  Batch loss =  0.139265
    327  Batch loss =  0.0940616
    328  Batch loss =  0.118399
    329  Batch loss =  0.111824
    330  Batch loss =  0.126266
    331  Batch loss =  0.135843
    332  Batch loss =  0.146795
    333  Batch loss =  0.133322
    334  Batch loss =  0.118313
    335  Batch loss =  0.128517
    336  Batch loss =  0.130211
    337  Batch loss =  0.127917
    338  Batch loss =  0.116984
    339  Batch loss =  0.113979
    340  Batch loss =  0.14612
    341  Batch loss =  0.13265
    342  Batch loss =  0.118659
    343  Batch loss =  0.139642
    344  Batch loss =  0.102538
    345  Batch loss =  0.118219
    346  Batch loss =  0.15528
    347  Batch loss =  0.11793
    348  Batch loss =  0.137639
    349  Batch loss =  0.111827
    350  Batch loss =  0.107642
    351  Batch loss =  0.13271
    352  Batch loss =  0.146026
    353  Batch loss =  0.108758
    354  Batch loss =  0.140466
    355  Batch loss =  0.11374
    356  Batch loss =  0.130259
    357  Batch loss =  0.13414
    358  Batch loss =  0.127991
    359  Batch loss =  0.118707
    360  Batch loss =  0.139002
    361  Batch loss =  0.11254
    362  Batch loss =  0.128158
    363  Batch loss =  0.137271
    364  Batch loss =  0.104326
    365  Batch loss =  0.100022
    366  Batch loss =  0.141949
    367  Batch loss =  0.146232
    368  Batch loss =  0.133353
    369  Batch loss =  0.112807
    370  Batch loss =  0.115129
    371  Batch loss =  0.118075
    372  Batch loss =  0.145529
    373  Batch loss =  0.120531
    374  Batch loss =  0.136836
    375  Batch loss =  0.131317
    376  Batch loss =  0.143854
    377  Batch loss =  0.129002
    378  Batch loss =  0.110246
    379  Batch loss =  0.1176
    380  Batch loss =  0.123676
    381  Batch loss =  0.136431
    382  Batch loss =  0.119353
    383  Batch loss =  0.103672
    384  Batch loss =  0.110367
    385  Batch loss =  0.134429
    386  Batch loss =  0.128478
    387  Batch loss =  0.129441
    388  Batch loss =  0.138341
    389  Batch loss =  0.142446
    390  Batch loss =  0.114987
    391  Batch loss =  0.117855
    392  Batch loss =  0.0927494
    393  Batch loss =  0.12757
    394  Batch loss =  0.135385
    395  Batch loss =  0.113608
    396  Batch loss =  0.143603
    397  Batch loss =  0.117241
    398  Batch loss =  0.137408
    399  Batch loss =  0.130739
    400  Batch loss =  0.115707
    401  Batch loss =  0.117257
    402  Batch loss =  0.114394
    403  Batch loss =  0.157189
    404  Batch loss =  0.100698
    405  Batch loss =  0.112702
    406  Batch loss =  0.129219
    407  Batch loss =  0.125926
    408  Batch loss =  0.128511
    409  Batch loss =  0.110403
    410  Batch loss =  0.137475
    411  Batch loss =  0.156256
    412  Batch loss =  0.128077
    413  Batch loss =  0.130055
    414  Batch loss =  0.121926
    415  Batch loss =  0.101231
    416  Batch loss =  0.143054
    417  Batch loss =  0.132307
    418  Batch loss =  0.111368
    419  Batch loss =  0.126784
    420  Batch loss =  0.125894
    421  Batch loss =  0.104337
    422  Batch loss =  0.105611
    423  Batch loss =  0.154686
    424  Batch loss =  0.136542
    425  Batch loss =  0.108872
    426  Batch loss =  0.129587
    427  Batch loss =  0.137436
    428  Batch loss =  0.112008
    429  Batch loss =  0.139117
    430  Batch loss =  0.1321
    431  Batch loss =  0.105692
    432  Batch loss =  0.129192
    433  Batch loss =  0.125098
    434  Batch loss =  0.0981711
    435  Batch loss =  0.121818
    436  Batch loss =  0.115823
    437  Batch loss =  0.151208
    438  Batch loss =  0.127764
    439  Batch loss =  0.118362
    440  Batch loss =  0.113721
    441  Batch loss =  0.129063
    442  Batch loss =  0.125407
    443  Batch loss =  0.117862
    444  Batch loss =  0.108663
    445  Batch loss =  0.129175
    446  Batch loss =  0.130944
    447  Batch loss =  0.129643
    448  Batch loss =  0.145448
    449  Batch loss =  0.12276
    450  Batch loss =  0.155663
    451  Batch loss =  0.137412
    452  Batch loss =  0.110028
    453  Batch loss =  0.0958658
    454  Batch loss =  0.146221
    455  Batch loss =  0.117329
    456  Batch loss =  0.144665
    457  Batch loss =  0.117828
    458  Batch loss =  0.153555
    459  Batch loss =  0.129293
    460  Batch loss =  0.125997
    461  Batch loss =  0.11798
    462  Batch loss =  0.121729
    463  Batch loss =  0.126727
    464  Batch loss =  0.127156
    465  Batch loss =  0.125534
    466  Batch loss =  0.0903611
    467  Batch loss =  0.146026
    468  Batch loss =  0.112383
    469  Batch loss =  0.136214
    470  Batch loss =  0.142393
    471  Batch loss =  0.146886
    472  Batch loss =  0.132593
    473  Batch loss =  0.131639
    474  Batch loss =  0.137307
    475  Batch loss =  0.10363
    476  Batch loss =  0.111388
    477  Batch loss =  0.115914
    478  Batch loss =  0.123138
    479  Batch loss =  0.124402
    480  Batch loss =  0.14015
    481  Batch loss =  0.157557
    482  Batch loss =  0.124564
    483  Batch loss =  0.117618
    484  Batch loss =  0.112005
    485  Batch loss =  0.14349
    486  Batch loss =  0.104426
    487  Batch loss =  0.114975
    488  Batch loss =  0.131149
    489  Batch loss =  0.149378
    490  Batch loss =  0.156155
    491  Batch loss =  0.11786
    492  Batch loss =  0.122593
    493  Batch loss =  0.101729
    494  Batch loss =  0.15085
    495  Batch loss =  0.117519
    496  Batch loss =  0.122759
    497  Batch loss =  0.121506
    498  Batch loss =  0.11691
    499  Batch loss =  0.14254
    500  Batch loss =  0.132812
    501  Batch loss =  0.128544
    502  Batch loss =  0.13677
    503  Batch loss =  0.116636
    504  Batch loss =  0.12369
    505  Batch loss =  0.114108
    506  Batch loss =  0.142053
    507  Batch loss =  0.111282
    508  Batch loss =  0.144043
    509  Batch loss =  0.116926
    510  Batch loss =  0.108258
    511  Batch loss =  0.129053
    512  Batch loss =  0.149104
    513  Batch loss =  0.121791
    514  Batch loss =  0.113726
    515  Batch loss =  0.104606
    516  Batch loss =  0.137945
    517  Batch loss =  0.137956
    518  Batch loss =  0.124449
    519  Batch loss =  0.137641
    520  Batch loss =  0.132792
    521  Batch loss =  0.127351
    522  Batch loss =  0.118741
    523  Batch loss =  0.111816
    524  Batch loss =  0.156651
    525  Batch loss =  0.114354
    526  Batch loss =  0.122647
    527  Batch loss =  0.117322
    528  Batch loss =  0.140059
    529  Batch loss =  0.121318
    530  Batch loss =  0.137282
    531  Batch loss =  0.114057
    532  Batch loss =  0.115467
    533  Batch loss =  0.131905
    534  Batch loss =  0.119786
    535  Batch loss =  0.127475
    536  Batch loss =  0.126835
    537  Batch loss =  0.126488
    538  Batch loss =  0.143901
    539  Batch loss =  0.118289
    540  Batch loss =  0.108662
    541  Batch loss =  0.0976597
    542  Batch loss =  0.129028
    543  Batch loss =  0.11282
    544  Batch loss =  0.132276
    545  Batch loss =  0.170812
    546  Batch loss =  0.118716
    547  Batch loss =  0.122772
    548  Batch loss =  0.133656
    549  Batch loss =  0.124074
    550  Batch loss =  0.111997
    551  Batch loss =  0.138832
    552  Batch loss =  0.126262
    553  Batch loss =  0.138298
    554  Batch loss =  0.13926
    555  Batch loss =  0.11647
    556  Batch loss =  0.1168
    557  Batch loss =  0.0974051
    558  Batch loss =  0.142643
    559  Batch loss =  0.11922
    560  Batch loss =  0.137985
    561  Batch loss =  0.140215
    562  Batch loss =  0.130395
    563  Batch loss =  0.108334
    564  Batch loss =  0.143939
    565  Batch loss =  0.111093
    566  Batch loss =  0.124767
    567  Batch loss =  0.1071
    568  Batch loss =  0.110163
    569  Batch loss =  0.136052
    570  Batch loss =  0.124678
    571  Batch loss =  0.161129
    572  Batch loss =  0.131092
    573  Batch loss =  0.121579
    574  Batch loss =  0.116202
    575  Batch loss =  0.119885
    576  Batch loss =  0.117583
    577  Batch loss =  0.129276
    578  Batch loss =  0.138398
    579  Batch loss =  0.132948
    580  Batch loss =  0.126074
    581  Batch loss =  0.119283
    582  Batch loss =  0.119227
    583  Batch loss =  0.107671
    584  Batch loss =  0.139778
    585  Batch loss =  0.126641
    586  Batch loss =  0.115611
    587  Batch loss =  0.136404
    588  Batch loss =  0.136311
    589  Batch loss =  0.116957
    590  Batch loss =  0.103796
    591  Batch loss =  0.147935
    592  Batch loss =  0.153078
    593  Batch loss =  0.119727
    594  Batch loss =  0.110418
    595  Batch loss =  0.126816
    596  Batch loss =  0.0982776
    597  Batch loss =  0.141116
    598  Batch loss =  0.125384
    599  Batch loss =  0.123194
    600  Batch loss =  0.14427
    601  Batch loss =  0.142771
    602  Batch loss =  0.129346
    603  Batch loss =  0.107997
    604  Batch loss =  0.140308
    605  Batch loss =  0.12148
    606  Batch loss =  0.137675
    607  Batch loss =  0.14098
    608  Batch loss =  0.121067
    609  Batch loss =  0.129993
    610  Batch loss =  0.0938918
    611  Batch loss =  0.15977
    612  Batch loss =  0.114374
    613  Batch loss =  0.123663
    614  Batch loss =  0.117006
    615  Batch loss =  0.121783
    616  Batch loss =  0.103229
    617  Batch loss =  0.124419
    618  Batch loss =  0.145013
    619  Batch loss =  0.125523
    620  Batch loss =  0.0992337
    621  Batch loss =  0.134836
    622  Batch loss =  0.126253
    623  Batch loss =  0.116167
    624  Batch loss =  0.151147
    625  Batch loss =  0.111171
    626  Batch loss =  0.0976748
    627  Batch loss =  0.132431
    628  Batch loss =  0.150643
    629  Batch loss =  0.153246
    630  Batch loss =  0.108571
    631  Batch loss =  0.112045
    632  Batch loss =  0.129526
    633  Batch loss =  0.120309
    634  Batch loss =  0.117313
    635  Batch loss =  0.134069
    636  Batch loss =  0.141161
    637  Batch loss =  0.148494
    638  Batch loss =  0.0953374
    639  Batch loss =  0.139881
    640  Batch loss =  0.102183
    641  Batch loss =  0.11853
    642  Batch loss =  0.145306
    643  Batch loss =  0.127441
    644  Batch loss =  0.118127
    645  Batch loss =  0.123583
    646  Batch loss =  0.10889
    647  Batch loss =  0.124691
    648  Batch loss =  0.108535
    649  Batch loss =  0.0928406
    650  Batch loss =  0.161618
    651  Batch loss =  0.116947
    652  Batch loss =  0.154662
    653  Batch loss =  0.112339
    654  Batch loss =  0.0959338
    655  Batch loss =  0.137529
    656  Batch loss =  0.117595
    657  Batch loss =  0.1096
    658  Batch loss =  0.155953
    659  Batch loss =  0.137781
    660  Batch loss =  0.131631
    661  Batch loss =  0.124181
    662  Batch loss =  0.160863
    663  Batch loss =  0.0934755
    664  Batch loss =  0.117801
    665  Batch loss =  0.14269
    666  Batch loss =  0.103405
    667  Batch loss =  0.126547
    668  Batch loss =  0.133227
    669  Batch loss =  0.12121
    670  Batch loss =  0.135059
    671  Batch loss =  0.146138
    672  Batch loss =  0.132679
    673  Batch loss =  0.0977894
    674  Batch loss =  0.124736
    675  Batch loss =  0.117408
    676  Batch loss =  0.118809
    677  Batch loss =  0.106913
    678  Batch loss =  0.140864
    679  Batch loss =  0.111728
    680  Batch loss =  0.128604
    681  Batch loss =  0.136264
    682  Batch loss =  0.120844
    683  Batch loss =  0.126101
    684  Batch loss =  0.104018
    685  Batch loss =  0.128324
    686  Batch loss =  0.141697
    687  Batch loss =  0.11707
    688  Batch loss =  0.137672
    689  Batch loss =  0.157655
    690  Batch loss =  0.103028
    691  Batch loss =  0.135357
    692  Batch loss =  0.124407
    693  Batch loss =  0.1242
    694  Batch loss =  0.122844
    695  Batch loss =  0.127077
    696  Batch loss =  0.135292
    697  Batch loss =  0.12589
    698  Batch loss =  0.1021
    699  Batch loss =  0.13349
    700  Batch loss =  0.124232
    701  Batch loss =  0.110442
    702  Batch loss =  0.127624
    703  Batch loss =  0.112931
    704  Batch loss =  0.159223
    705  Batch loss =  0.123858
    706  Batch loss =  0.116963
    707  Batch loss =  0.123623
    708  Batch loss =  0.106567
    709  Batch loss =  0.107356
    710  Batch loss =  0.140845
    711  Batch loss =  0.0994645
    712  Batch loss =  0.153209
    713  Batch loss =  0.135956
    714  Batch loss =  0.153121
    715  Batch loss =  0.125964
    716  Batch loss =  0.120945
    717  Batch loss =  0.130099
    718  Batch loss =  0.134197
    719  Batch loss =  0.115131
    720  Batch loss =  0.117563
    721  Batch loss =  0.15125
    722  Batch loss =  0.10112
    723  Batch loss =  0.141766
    724  Batch loss =  0.115991
    725  Batch loss =  0.104312
    726  Batch loss =  0.102902
    727  Batch loss =  0.122753
    728  Batch loss =  0.112045
    729  Batch loss =  0.136423
    730  Batch loss =  0.146586
    731  Batch loss =  0.117338
    732  Batch loss =  0.121085
    733  Batch loss =  0.126671
    734  Batch loss =  0.120491
    735  Batch loss =  0.134299
    736  Batch loss =  0.100185
    737  Batch loss =  0.111402
    738  Batch loss =  0.150115
    739  Batch loss =  0.133377
    740  Batch loss =  0.131752
    741  Batch loss =  0.130118
    742  Batch loss =  0.104086
    743  Batch loss =  0.145275
    744  Batch loss =  0.112301
    745  Batch loss =  0.136678
    746  Batch loss =  0.109848
    747  Batch loss =  0.122122
    748  Batch loss =  0.136061
    749  Batch loss =  0.113049
    750  Batch loss =  0.100354
    751  Batch loss =  0.112537
    752  Batch loss =  0.134287
    753  Batch loss =  0.165058
    754  Batch loss =  0.127908
    755  Batch loss =  0.136512
    756  Batch loss =  0.113103
    757  Batch loss =  0.10749
    758  Batch loss =  0.179709
    759  Batch loss =  0.11229
    760  Batch loss =  0.0977964
    761  Batch loss =  0.12989
    762  Batch loss =  0.141293
    763  Batch loss =  0.132257
    764  Batch loss =  0.143588
    765  Batch loss =  0.122575
    766  Batch loss =  0.109019
    767  Batch loss =  0.138144
    768  Batch loss =  0.121499
    769  Batch loss =  0.151559
    770  Batch loss =  0.103134
    771  Batch loss =  0.12821
    772  Batch loss =  0.115848
    773  Batch loss =  0.108267
    774  Batch loss =  0.106636
    775  Batch loss =  0.114524
    776  Batch loss =  0.11681
    777  Batch loss =  0.11999
    778  Batch loss =  0.127318
    779  Batch loss =  0.114373
    780  Batch loss =  0.11751
    781  Batch loss =  0.146584
    782  Batch loss =  0.147678
    783  Batch loss =  0.145095
    784  Batch loss =  0.117913
    785  Batch loss =  0.121123
    786  Batch loss =  0.14746
    787  Batch loss =  0.134528
    788  Batch loss =  0.127266
    789  Batch loss =  0.104609
    790  Batch loss =  0.117543
    791  Batch loss =  0.123061
    792  Batch loss =  0.122993
    793  Batch loss =  0.118817
    794  Batch loss =  0.152748
    795  Batch loss =  0.112183
    796  Batch loss =  0.117411
    797  Batch loss =  0.134674
    798  Batch loss =  0.121195
    799  Batch loss =  0.102579
    800  Batch loss =  0.135547
    801  Batch loss =  0.122705
    802  Batch loss =  0.139448
    803  Batch loss =  0.134518
    804  Batch loss =  0.116449
    805  Batch loss =  0.13163
    806  Batch loss =  0.136807
    807  Batch loss =  0.137202
    808  Batch loss =  0.0993789
    809  Batch loss =  0.134912
    810  Batch loss =  0.123578
    811  Batch loss =  0.108421
    812  Batch loss =  0.138946
    813  Batch loss =  0.13331
    814  Batch loss =  0.133317
    815  Batch loss =  0.11123
    816  Batch loss =  0.102699
    817  Batch loss =  0.136123
    818  Batch loss =  0.126422
    819  Batch loss =  0.123767
    820  Batch loss =  0.121712
    821  Batch loss =  0.107875
    822  Batch loss =  0.122695
    823  Batch loss =  0.141233
    824  Batch loss =  0.155078
    825  Batch loss =  0.122713
    826  Batch loss =  0.130537
    827  Batch loss =  0.146802
    828  Batch loss =  0.125652
    829  Batch loss =  0.0971379
    830  Batch loss =  0.12223
    831  Batch loss =  0.108333
    832  Batch loss =  0.113645
    833  Batch loss =  0.14908
    834  Batch loss =  0.154718
    835  Batch loss =  0.107735
    836  Batch loss =  0.109319
    837  Batch loss =  0.115742
    838  Batch loss =  0.121041
    839  Batch loss =  0.101776
    840  Batch loss =  0.12694
    841  Batch loss =  0.135905
    842  Batch loss =  0.116701
    843  Batch loss =  0.130695
    844  Batch loss =  0.11039
    845  Batch loss =  0.143948
    846  Batch loss =  0.132511
    847  Batch loss =  0.125702
    848  Batch loss =  0.126499
    849  Batch loss =  0.109562
    850  Batch loss =  0.132429
    851  Batch loss =  0.14101
    852  Batch loss =  0.144993
    853  Batch loss =  0.0994568
    854  Batch loss =  0.148181
    855  Batch loss =  0.115597
    856  Batch loss =  0.113096
    857  Batch loss =  0.128137
    858  Batch loss =  0.138891
    859  Batch loss =  0.132381
    860  Batch loss =  0.112833
    861  Batch loss =  0.123612
    862  Batch loss =  0.129606
    863  Batch loss =  0.128441
    864  Batch loss =  0.0916404
    865  Batch loss =  0.130185
    866  Batch loss =  0.130556
    867  Batch loss =  0.134447
    868  Batch loss =  0.123109
    869  Batch loss =  0.131239
    870  Batch loss =  0.139196
    871  Batch loss =  0.117149
    872  Batch loss =  0.140178
    873  Batch loss =  0.121132
    874  Batch loss =  0.102684
    875  Batch loss =  0.129191
    876  Batch loss =  0.100893
    877  Batch loss =  0.144078
    878  Batch loss =  0.145183
    879  Batch loss =  0.136119
    880  Batch loss =  0.146273
    881  Batch loss =  0.123302
    882  Batch loss =  0.123344
    883  Batch loss =  0.103814
    884  Batch loss =  0.13528
    885  Batch loss =  0.104636
    886  Batch loss =  0.148563
    887  Batch loss =  0.136005
    888  Batch loss =  0.134034
    889  Batch loss =  0.116409
    890  Batch loss =  0.116402
    891  Batch loss =  0.138345
    892  Batch loss =  0.117315
    893  Batch loss =  0.121045
    894  Batch loss =  0.143357
    895  Batch loss =  0.109086
    896  Batch loss =  0.109073
    897  Batch loss =  0.159898
    898  Batch loss =  0.1307
    899  Batch loss =  0.146465
    900  Batch loss =  0.11035
    901  Batch loss =  0.134004
    902  Batch loss =  0.119279
    903  Batch loss =  0.122216
    904  Batch loss =  0.104356
    905  Batch loss =  0.127999
    906  Batch loss =  0.122812
    907  Batch loss =  0.142113
    908  Batch loss =  0.131432
    909  Batch loss =  0.16433
    910  Batch loss =  0.124741
    911  Batch loss =  0.099499
    912  Batch loss =  0.132599
    913  Batch loss =  0.12877
    914  Batch loss =  0.119361
    915  Batch loss =  0.128188
    916  Batch loss =  0.137614
    917  Batch loss =  0.14864
    918  Batch loss =  0.12354
    919  Batch loss =  0.116302
    920  Batch loss =  0.113023
    921  Batch loss =  0.10737
    922  Batch loss =  0.1479
    923  Batch loss =  0.131797
    924  Batch loss =  0.132418
    925  Batch loss =  0.127096
    926  Batch loss =  0.12648
    927  Batch loss =  0.107004
    928  Batch loss =  0.127657
    929  Batch loss =  0.140023
    930  Batch loss =  0.124686
    931  Batch loss =  0.122195
    932  Batch loss =  0.11318
    933  Batch loss =  0.163402
    934  Batch loss =  0.132384
    935  Batch loss =  0.147433
    936  Batch loss =  0.123892
    937  Batch loss =  0.105517
    938  Batch loss =  0.0936676
    939  Batch loss =  0.132323
    940  Batch loss =  0.117696
    941  Batch loss =  0.117053
    942  Batch loss =  0.108058
    943  Batch loss =  0.119136
    944  Batch loss =  0.111593
    945  Batch loss =  0.108617
    946  Batch loss =  0.0912777
    947  Batch loss =  0.130213
    948  Batch loss =  0.148123
    949  Batch loss =  0.113079
    950  Batch loss =  0.115588
    951  Batch loss =  0.126241
    952  Batch loss =  0.145386
    953  Batch loss =  0.145642
    954  Batch loss =  0.132092
    955  Batch loss =  0.12148
    956  Batch loss =  0.12129
    957  Batch loss =  0.113767
    958  Batch loss =  0.138301
    959  Batch loss =  0.124937
    960  Batch loss =  0.124683
    961  Batch loss =  0.124863
    962  Batch loss =  0.136533
    963  Batch loss =  0.115744
    964  Batch loss =  0.133331
    965  Batch loss =  0.130522
    966  Batch loss =  0.123349
    967  Batch loss =  0.121433
    968  Batch loss =  0.109522
    969  Batch loss =  0.130935
    970  Batch loss =  0.127243
    971  Batch loss =  0.13621
    972  Batch loss =  0.112938
    973  Batch loss =  0.122093
    974  Batch loss =  0.112599
    975  Batch loss =  0.125177
    976  Batch loss =  0.141618
    977  Batch loss =  0.0878865
    978  Batch loss =  0.114625
    979  Batch loss =  0.121106
    980  Batch loss =  0.14614
    981  Batch loss =  0.112149
    982  Batch loss =  0.130811
    983  Batch loss =  0.140706
    984  Batch loss =  0.130885
    985  Batch loss =  0.0941467
    986  Batch loss =  0.102346
    987  Batch loss =  0.128588
    988  Batch loss =  0.133996
    989  Batch loss =  0.164358
    990  Batch loss =  0.108792
    991  Batch loss =  0.117701
    992  Batch loss =  0.129106
    993  Batch loss =  0.107956
    994  Batch loss =  0.123358
    995  Batch loss =  0.13332
    996  Batch loss =  0.140159
    997  Batch loss =  0.150532
    998  Batch loss =  0.122733
    999  Batch loss =  0.127273
    1000  Batch loss =  0.103376
    1001  Batch loss =  0.156162
    1002  Batch loss =  0.126076
    1003  Batch loss =  0.109098
    1004  Batch loss =  0.124156
    1005  Batch loss =  0.14748
    1006  Batch loss =  0.11324
    1007  Batch loss =  0.114664
    1008  Batch loss =  0.151191
    1009  Batch loss =  0.122908
    1010  Batch loss =  0.124026
    1011  Batch loss =  0.111806
    1012  Batch loss =  0.142524
    1013  Batch loss =  0.129395
    1014  Batch loss =  0.129107
    1015  Batch loss =  0.134208
    1016  Batch loss =  0.109787
    1017  Batch loss =  0.131491
    1018  Batch loss =  0.133349
    1019  Batch loss =  0.131031
    1020  Batch loss =  0.117996
    1021  Batch loss =  0.123979
    1022  Batch loss =  0.123155
    1023  Batch loss =  0.136973
    1024  Batch loss =  0.142282
    1025  Batch loss =  0.117761
    1026  Batch loss =  0.134279
    1027  Batch loss =  0.0992254
    1028  Batch loss =  0.113339
    1029  Batch loss =  0.139486
    1030  Batch loss =  0.115133
    1031  Batch loss =  0.12817
    1032  Batch loss =  0.111982
    1033  Batch loss =  0.151535
    1034  Batch loss =  0.12424
    1035  Batch loss =  0.104517
    1036  Batch loss =  0.126206
    1037  Batch loss =  0.132489
    1038  Batch loss =  0.116947
    1039  Batch loss =  0.127138
    1040  Batch loss =  0.129629
    1041  Batch loss =  0.126971
    1042  Batch loss =  0.114887
    1043  Batch loss =  0.11543
    1044  Batch loss =  0.13688
    1045  Batch loss =  0.117979
    1046  Batch loss =  0.131137
    1047  Batch loss =  0.130753
    1048  Batch loss =  0.132116
    1049  Batch loss =  0.113736
    1050  Batch loss =  0.13418
    1051  Batch loss =  0.12952
    1052  Batch loss =  0.12085
    1053  Batch loss =  0.111934
    1054  Batch loss =  0.153226
    1055  Batch loss =  0.111516
    1056  Batch loss =  0.11389
    1057  Batch loss =  0.118361
    1058  Batch loss =  0.149821
    1059  Batch loss =  0.110027
    1060  Batch loss =  0.105749
    1061  Batch loss =  0.127485
    1062  Batch loss =  0.129901
    1063  Batch loss =  0.159113
    1064  Batch loss =  0.152973
    1065  Batch loss =  0.124924
    1066  Batch loss =  0.139838
    1067  Batch loss =  0.109716
    1068  Batch loss =  0.126091
    1069  Batch loss =  0.12011
    1070  Batch loss =  0.124768
    1071  Batch loss =  0.107845
    1072  Batch loss =  0.123461
    1073  Batch loss =  0.119669
    1074  Batch loss =  0.13038
    1075  Batch loss =  0.120749
    1076  Batch loss =  0.135213
    1077  Batch loss =  0.119742
    1078  Batch loss =  0.125696
    1079  Batch loss =  0.129915
    1080  Batch loss =  0.128066
    1081  Batch loss =  0.114124
    1082  Batch loss =  0.11187
    1083  Batch loss =  0.125722
    1084  Batch loss =  0.125651
    1085  Batch loss =  0.146891
    1086  Batch loss =  0.116651
    1087  Batch loss =  0.143916
    1088  Batch loss =  0.131204
    1089  Batch loss =  0.143488
    1090  Batch loss =  0.120333
    1091  Batch loss =  0.10761
    1092  Batch loss =  0.125318
    1093  Batch loss =  0.127205
    1094  Batch loss =  0.127719
    1095  Batch loss =  0.129321
    1096  Batch loss =  0.148207
    1097  Batch loss =  0.121959
    1098  Batch loss =  0.120339
    1099  Batch loss =  0.131649
    1100  Batch loss =  0.1017
    1101  Batch loss =  0.139194
    1102  Batch loss =  0.104348
    1103  Batch loss =  0.156051
    1104  Batch loss =  0.13743
    1105  Batch loss =  0.131709
    1106  Batch loss =  0.0896715
    1107  Batch loss =  0.105746
    1108  Batch loss =  0.13279
    1109  Batch loss =  0.0871025
    1110  Batch loss =  0.154338
    1111  Batch loss =  0.113424
    1112  Batch loss =  0.125681
    1113  Batch loss =  0.123899
    1114  Batch loss =  0.116408
    1115  Batch loss =  0.131383
    1116  Batch loss =  0.127905
    1117  Batch loss =  0.126985
    1118  Batch loss =  0.143396
    1119  Batch loss =  0.147879
    1120  Batch loss =  0.108583
    1121  Batch loss =  0.0985501
    1122  Batch loss =  0.13913
    1123  Batch loss =  0.138033
    1124  Batch loss =  0.116864
    1125  Batch loss =  0.123493
    1126  Batch loss =  0.145988
    1127  Batch loss =  0.140516
    1128  Batch loss =  0.135354
    1129  Batch loss =  0.127378
    1130  Batch loss =  0.106592
    1131  Batch loss =  0.114342
    1132  Batch loss =  0.144764
    1133  Batch loss =  0.113525
    1134  Batch loss =  0.127552
    1135  Batch loss =  0.144706
    1136  Batch loss =  0.134673
    1137  Batch loss =  0.142867
    1138  Batch loss =  0.114524
    1139  Batch loss =  0.11072
    1140  Batch loss =  0.123879
    1141  Batch loss =  0.156023
    1142  Batch loss =  0.113674
    1143  Batch loss =  0.10152
    1144  Batch loss =  0.140215
    1145  Batch loss =  0.117199
    1146  Batch loss =  0.0911841
    1147  Batch loss =  0.10558
    1148  Batch loss =  0.129329
    1149  Batch loss =  0.129591
    1150  Batch loss =  0.146507
    1151  Batch loss =  0.134231
    1152  Batch loss =  0.113714
    1153  Batch loss =  0.148937
    1154  Batch loss =  0.119082
    1155  Batch loss =  0.152841
    1156  Batch loss =  0.128942
    1157  Batch loss =  0.108118
    1158  Batch loss =  0.141023
    1159  Batch loss =  0.117279
    1160  Batch loss =  0.114753
    1161  Batch loss =  0.122892
    1162  Batch loss =  0.159004
    1163  Batch loss =  0.111902
    1164  Batch loss =  0.1338
    1165  Batch loss =  0.0975842
    1166  Batch loss =  0.118615
    1167  Batch loss =  0.125605
    1168  Batch loss =  0.107157
    1169  Batch loss =  0.11311
    1170  Batch loss =  0.121796
    1171  Batch loss =  0.128888
    1172  Batch loss =  0.141974
    1173  Batch loss =  0.138449
    1174  Batch loss =  0.128154
    1175  Batch loss =  0.114332
    1176  Batch loss =  0.137942
    1177  Batch loss =  0.124951
    1178  Batch loss =  0.122641
    1179  Batch loss =  0.123024
    1180  Batch loss =  0.139792
    1181  Batch loss =  0.116693
    1182  Batch loss =  0.121186
    1183  Batch loss =  0.129908
    1184  Batch loss =  0.129094
    1185  Batch loss =  0.135515
    1186  Batch loss =  0.140272
    1187  Batch loss =  0.157062
    1188  Batch loss =  0.116965
    1189  Batch loss =  0.0980259
    1190  Batch loss =  0.116394
    1191  Batch loss =  0.108457
    1192  Batch loss =  0.136498
    1193  Batch loss =  0.120644
    1194  Batch loss =  0.126268
    1195  Batch loss =  0.121419
    1196  Batch loss =  0.132125
    1197  Batch loss =  0.117163
    1198  Batch loss =  0.121174
    1199  Batch loss =  0.11306
    1200  Batch loss =  0.153882
    1201  Batch loss =  0.117788
    1202  Batch loss =  0.138523
    1203  Batch loss =  0.112469
    1204  Batch loss =  0.113376
    1205  Batch loss =  0.126253
    1206  Batch loss =  0.149412
    1207  Batch loss =  0.127695
    1208  Batch loss =  0.118033
    1209  Batch loss =  0.128296
    1210  Batch loss =  0.130748
    1211  Batch loss =  0.114076
    1212  Batch loss =  0.14278
    1213  Batch loss =  0.136626
    1214  Batch loss =  0.123107
    1215  Batch loss =  0.118138
    1216  Batch loss =  0.112047
    1217  Batch loss =  0.137694
    1218  Batch loss =  0.118023
    1219  Batch loss =  0.112592
    1220  Batch loss =  0.145659
    1221  Batch loss =  0.0967591
    1222  Batch loss =  0.136389
    1223  Batch loss =  0.125523
    1224  Batch loss =  0.144446
    1225  Batch loss =  0.135231
    1226  Batch loss =  0.114462
    1227  Batch loss =  0.137951
    1228  Batch loss =  0.109686
    1229  Batch loss =  0.102769
    1230  Batch loss =  0.129465
    1231  Batch loss =  0.0914182
    1232  Batch loss =  0.120251
    1233  Batch loss =  0.145812
    1234  Batch loss =  0.129803
    1235  Batch loss =  0.142494
    1236  Batch loss =  0.13012
    1237  Batch loss =  0.102967
    1238  Batch loss =  0.102132
    1239  Batch loss =  0.149486
    1240  Batch loss =  0.138355
    1241  Batch loss =  0.105865
    1242  Batch loss =  0.138895
    1243  Batch loss =  0.134371
    1244  Batch loss =  0.123548
    1245  Batch loss =  0.134653
    1246  Batch loss =  0.129745
    1247  Batch loss =  0.124018
    1248  Batch loss =  0.117903
    1249  Batch loss =  0.121735
    1250  Batch loss =  0.107051
    1251  Batch loss =  0.102546
    1252  Batch loss =  0.135664
    1253  Batch loss =  0.158507
    1254  Batch loss =  0.123715
    1255  Batch loss =  0.113335
    1256  Batch loss =  0.11754
    1257  Batch loss =  0.12515
    1258  Batch loss =  0.153371
    1259  Batch loss =  0.114377
    1260  Batch loss =  0.137387
    1261  Batch loss =  0.148806
    1262  Batch loss =  0.10781
    1263  Batch loss =  0.0991298
    1264  Batch loss =  0.133042
    1265  Batch loss =  0.135908
    1266  Batch loss =  0.153238
    1267  Batch loss =  0.143132
    1268  Batch loss =  0.109145
    1269  Batch loss =  0.106872
    1270  Batch loss =  0.115967
    1271  Batch loss =  0.128877
    1272  Batch loss =  0.103612
    1273  Batch loss =  0.126188
    1274  Batch loss =  0.112925
    1275  Batch loss =  0.157271
    1276  Batch loss =  0.134872
    1277  Batch loss =  0.133175
    1278  Batch loss =  0.0874534
    1279  Batch loss =  0.125374
    1280  Batch loss =  0.146105
    1281  Batch loss =  0.129373
    1282  Batch loss =  0.100387
    1283  Batch loss =  0.145982
    1284  Batch loss =  0.1179
    1285  Batch loss =  0.125246
    1286  Batch loss =  0.136294
    1287  Batch loss =  0.11121
    1288  Batch loss =  0.118586
    1289  Batch loss =  0.123084
    Epoch_end = 0 , avg_loss =  0.125743936039  nn =  176044
    


```python
import copy
from sklearn.metrics import roc_curve, auc, average_precision_score

all_preds = []
all_labels = []
all_losses = []
for prot_data in train_data:
     temp_data = copy.deepcopy(prot_data)
     n = prot_data['label'].shape[0] #no of labels for this protein molecule.
     #split the labels into chunks of minibatch_size.
     batch_split_points = np.arange(0,n,minibatch_size)[1:]
     batches = np.array_split(prot_data['label'],batch_split_points)
     for a_batch in batches:
        temp_data['label'] = a_batch
        feed_dict = build_feed_dict(temp_data)
        res = sess.run([loss,preds,labels], feed_dict=feed_dict)
        pred_v = np.squeeze(res[1])
        if len(pred_v.shape)==0:
           pred_v = [pred_v]
           all_preds += pred_v
        else:
           pred_v = pred_v.tolist()
           all_preds += pred_v
        all_labels += res[2].tolist()
        all_losses += [res[0]]

fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
print('mean loss = ',np.mean(all_losses))
print('roc_auc = ',roc_auc)
```

    mean loss =  0.121488
    roc_auc =  0.569184936411
    


```python

```
