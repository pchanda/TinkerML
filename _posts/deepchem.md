
This writeup shows how to use graph convolutions for a regression like problem using the [DeepChem](https://deepchem.io/) library. 

The input data will be a csv file containing the [Smiles](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) representation of molecules and solubility properties 'cLogP' and 'cLogS'. We will use graph convolutions from deepchem. The file looks like:

```python
ID,Molweight,cLogP,cLogS,Smiles
788216,357.429,3.3344,-4.043,COc(c1ccccc11)ccc1S(NCCOc1ccccc1)(=O)=O
788246,226.255,0.5955,-2.606,CCOc(ccc(S(N)(=O)=O)c1)c1C#N
788256,428.726,5.249,-7.72,Cc(ccc(Cl)c1)c1NC(CSc1nnc(-c(ccc(Cl)c2)c2Cl)o1)=O
788266,296.414,2.2175,-2.319,Cc1nc(ccc(S(N2CCCCC2)(=O)=O)c2)c2s1
788286,277.387,3.5128,-3.775,CC(C)CCNS(c1cc2ccccc2cc1)(=O)=O
788326,296.393,3.1664,-4.284,N#CCc(cccc1)c1C(NCCSc1ccccc1)=O
788366,291.825,4.7289,-5.617,Clc1c(CSc2nc(cccc3)c3s2)cccc1
788386,382.458,3.5813,-4.326,CC(C)(C)NC(c(cccc1)c1NC(/C=C/c(cc1)cc(OC)c1OC)=O)=O
...
```
We will be predicting the molecular properties 'cLogP' and 'cLogS' using only the Smiles represenation of the molecules. 


Start with some python imports and a function definition to read the input data:


```python
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, L2Loss, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
import numpy as np
import tensorflow as tf
import os


num_epochs = 50
batch_size = 200
pad_batches = True

tg = TensorGraph(batch_size=batch_size,learning_rate=0.0005,use_queue=False)
prediction_tasks = ['cLogP','cLogS']

def read_data(input_file_path):
    featurizer = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(tasks=prediction_tasks, smiles_field="Smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    # Initialize transformers
    transformer = dc.trans.NormalizationTransformer(transform_w=True, dataset=dataset)
    print("About to transform data")
    dataset = transformer.transform(dataset)
    #Randomly split into training and testing : 80-20 split.
    splitter = dc.splits.splitters.RandomSplitter()
    trainset,testset = splitter.train_test_split(dataset,frac_train=0.8)
    return trainset,testset
```

Next we define the deep learning model for training and testing. There will be two layers each with graph convolutions followed by normalization and graph pooling.


```python
print('About to define models')

# placeholder for a feature vector of length 75 for each atom
atom_features = Feature(shape=(None, 75))
# an indexing convenience that makes it easy to locate atoms from all molecules with a given degree
degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
# placeholder that determines the membership of atoms in molecules (atom i belongs to molecule membership[i])
membership = Feature(shape=(None,), dtype=tf.int32)
# list that contains adjacency lists grouped by atom degree
deg_adjs = []
for i in range(0, 10 + 1):
   deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32) # placeholder for adj list of all nodes with i neighbors
   deg_adjs.append(deg_adj)

# layer 1
gc1 = GraphConv(64, activation_fn=tf.nn.relu, in_layers=[atom_features, degree_slice, membership]+deg_adjs )
batch_norm1 = BatchNorm(in_layers=[gc1])
gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)

# Layer 2
gc2 = GraphConv(64,activation_fn=tf.nn.relu,in_layers=[gp1, degree_slice, membership] + deg_adjs)
batch_norm2 = BatchNorm(in_layers=[gc2])
gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)

# fully connected layer
dense = Dense(out_channels=512, activation_fn=tf.nn.relu, in_layers=[gp2])
batch_norm3 = BatchNorm(in_layers=[dense])
readout = GraphGather( batch_size=batch_size, activation_fn=tf.nn.tanh, in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)
```

As there are two regression tasks (one for each of 'clogP' and 'cLogS'), add a final dense layer for each task. Add L2 loss as the loss function.


```python
costs = []
labels = []
for task in range(len(prediction_tasks)): # for each regression task
    regression = Dense( out_channels=1, activation_fn=None, in_layers=[readout])
    tg.add_output(regression)
    label = Label(shape=(None, 1))
    labels.append(label)
    cost = L2Loss(in_layers=[label, regression])
    costs.append(cost)

all_cost = Stack(in_layers=costs, axis=1)
weights = Weights(shape=(None, len(tox21_tasks)))
loss = WeightedError(in_layers=[all_cost, weights])
tg.set_loss(loss)
```

Add a data generator to generate batches of data for feeding the tensorflow graph.


```python
def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
  for epoch in range(epochs):
    if not predict:
        print('Starting epoch %i' % epoch)
    data_iterator_batch = dataset.iterbatches(batch_size, pad_batches=pad_batches, deterministic=True)
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(data_iterator_batch):
      d = {} #sort of feed_dict
      for index, label in enumerate(labels):
        d[label] = np.expand_dims(y_b[:, index],1)
      d[weights] = w_b
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      d[atom_features] = multiConvMol.get_atom_features()
      d[degree_slice] = multiConvMol.deg_slice
      d[membership] = multiConvMol.membership
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
      yield d
```

TensorGraph.Predict returns a list of arrays, one for each output. So add a utility function to reshape the output values from the two predictions into a single column. We also have to remove the padding on the last batch.


```python
def reshape_y_pred(y_true, y_pred):
    """
    TensorGraph.Predict returns a list of arrays, one for each output
    We also have to remove the padding on the last batch
    Metrics taks results of shape (samples, n_task, prob_of_class)
    """
    n_samples = len(y_true)
    retval = np.stack(y_pred, axis=1)
    return retval[:n_samples]
```

Now, we can train the model using TensorGraph.fit_generator(generator) which will use the generator we’ve defined to train the model.


```python
train_dataset,test_dataset = read_data("Sample_train.csv")
tg.fit_generator(data_generator(train_dataset, epochs=num_epochs))

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean,  mode = "regression")

#check performance on training data itself
print("Evaluating on train data")
train_predictions = tg.predict_on_generator(data_generator(train_dataset, predict=True))
train_predictions = reshape_y_pred(train_dataset.y, train_predictions)
train_scores = metric.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
print("Train Correlation Score: %f" % train_scores)

#check performance on the test data
print("Evaluating on test data")
test_predictions = tg.predict_on_generator(data_generator(test_dataset, predict=True))
test_predictions = reshape_y_pred(test_dataset.y, test_predictions)
test_scores = metric.compute_metric(test_dataset.y, test_predictions, test_dataset.w)
print("Test Correlation Score: %f" % test_scores)
```

Running the code gives fairly good results:
