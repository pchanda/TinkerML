---
layout: post
title: Encoding a set of Graphs using Neural Message Passing
categories: ['Pytorch','DeepLearning','Graphs','Message Passing Networks']
---

You will need a gpu and cuda. [Data used in the code](https://github.com/pchanda/pchanda.github.io/tree/master/data/data_graph_encoding/)

## Model parameters and input configurations

```python
'''
We will read 2 graphs, store them in netwrokx objects.
Then we will encode each graph to a vector.
'''

import networkx as nx
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def check_if_gpu_is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())

CUDA=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)

check_if_gpu_is_available()
```
## Read the graphs. 

```python
#Create and empty graph and populate it by reading the input files.
G = nx.Graph() #create an empty graph.
filepath = './data_graph_encoding/'

#list of edges.
data_edge_list = np.loadtxt(os.path.join(filepath,'edges.txt'), delimiter=',').astype(int) 

#features for each node
data_node_features = np.loadtxt(os.path.join(filepath,'node_features.txt'), delimiter=',')

#labels for each node
data_node_labels = np.loadtxt(os.path.join(filepath,'node_labels.txt'), delimiter=',').astype(int)

#features for each edge
data_edge_features = np.loadtxt(os.path.join(filepath,'edge_features.txt'), delimiter=',')

#1 line = a label for each graph. 
data_graph_labels = np.loadtxt(os.path.join(filepath,'graph_labels.txt'), delimiter=',').astype(int)

#which nodes belong to which graph.
data_graph_indicator = np.loadtxt(os.path.join(filepath,'graph_indicators.txt'), delimiter=',').astype(int)

data_tuple = list(map(tuple, data_edge_list)) #convert to (node1,node2) tuples.
# and add edges to the graph
G.add_edges_from(data_tuple)


NODE_FDIM = data_node_features.shape[1]
# update the nodes by adding node attributes
for i in range(data_node_labels.shape[0]):
    G.add_node(i, feature = data_node_features[i])
    G.add_node(i, label = data_node_labels[i])

EDGE_FDIM = data_edge_features.shape[1]
# update the edges by adding edge attributes
for i in range(data_edge_features.shape[0]):
    u,v = data_tuple[i]
    G.add_edge(u,v, feature = data_edge_features[i])
    
G.remove_nodes_from(list(nx.isolates(G)))

# split into each graphs
graph_num = data_graph_indicator.max()+1
node_list = np.arange(data_graph_indicator.shape[0])

all_graphs = []
for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator==i]
    #print('nodes = ',nodes)
    G_sub = G.subgraph(nodes)
    G_sub.graph['label'] = data_graph_labels[i]
    all_graphs.append(G_sub)

#print each graph
print('Printing the graphs read ------------------>')
for i,gr in enumerate(all_graphs):
    print('graph ',i,', no of nodes:',gr.number_of_nodes())
    print('adjacency_list:')
    for n, nbrdict in gr.adjacency():
        print(n,'=',nbrdict)
    print('adjacency_matrix = \n',nx.to_numpy_matrix(gr)) #adjacency matrix
    node_features = nx.get_node_attributes(gr, 'feature')
    node_labels = nx.get_node_attributes(gr, 'label')

    for nd in gr.nodes():
        print('node',nd,'features=',node_features[nd],'label=',node_labels[nd])

    edge_features = nx.get_edge_attributes(gr,'feature') 
    for k,ee in enumerate(gr.edges()):
        print('edge',ee,'features=',edge_features[ee])
    print('----------------------------------------')
```

## Create a torch Dataset object using the graphs read.


```python
from torch.utils.data import Dataset
#Create a torch DataLoader object using the graph data read above.
class GraphSet(Dataset):
    
    def __init__(self,all_graphs):
        self.all_graphs = all_graphs

    def __len__(self):
        return len(self.all_graphs)
    
    def __getitem__(self,idx):
        return self.all_graphs[idx]
```


## Define the helper functions for neural message passing for encoding each batch of graphs.

```python

def index_select_ND(message, dim, index_matrix):
    # say message is of shape [a,c] 89,5
    # say index_matrix is of shape [a,b] 89,6
    # to select the entries from message indexed by index_matrix entries.
    index_size = index_matrix.size()  # say index_size = [a,b]  
    suffix_dim = message.size()[1:]   # suffix_dim = [c]
    final_size = index_size + suffix_dim # final_size = [a,b,c] = 89 6 5
    index_matrix_flat = index_matrix.view(-1) # flatten the index to 1-dim tensor of shape = [ab]. 89*6
    # use index_matrix_flat to index into message, i.e., select ab entries from [a,c]
    # this is possible as the indices are repeated.
    selected = message.index_select(dim, index_matrix_flat) # selected has shape [ab,c]  
    selected_reshaped = selected.view(final_size) #reshape tensor to [a,b,c]
    return selected_reshaped


def process_all_graphs(all_graphs, node_fdim, edge_fdim):
    
    padding = torch.zeros(node_fdim + edge_fdim)
    fnodes = []
    fedges = [padding] #Ensure edges are 1-indexed, i.e entry 0 is dummy [000...0]

    edge_indices = []
    all_edges = [(-1,-1)] #Ensure edges are 1-indexed, entry 0 is dummay [(-1,-1)]
 
    scope = [] # start and no. of nodes of each graph in all_graphs
    total_nodes = 0
    MAX_NBR = 0

    for i,gr in enumerate(all_graphs):

        #get the node and edge features for this graph.
        node_features = nx.get_node_attributes(gr, 'feature')
        edge_features = nx.get_edge_attributes(gr,'feature') 
        num_nodes = gr.number_of_nodes()
        #print('graph',i,' has #nodes = ',num_nodes)
        
        for a_node in gr.nodes():
            num_nbr = len(gr[a_node])
            MAX_NBR = num_nbr if num_nbr>MAX_NBR else MAX_NBR        
            nf = torch.Tensor(node_features[a_node]) 
            fnodes.append(nf) #one-hot encoded node features.
            edge_indices.append([])

        for ne, an_edge in enumerate(gr.edges):
            x,y = an_edge
            #print('EDGE : ',x,y)
            bf = torch.Tensor(edge_features[an_edge])

            b = len(all_edges)
            all_edges.append((x,y))
            fedges.append( torch.cat([fnodes[x], bf], 0) )
            edge_indices[y].append(b)

            b = len(all_edges)
            all_edges.append((y,x))
            fedges.append( torch.cat([fnodes[y], bf], 0) )
            edge_indices[x].append(b)
            
        scope.append((total_nodes,num_nodes))
        total_nodes += num_nodes

    total_edges = len(all_edges)
    fnodes = torch.stack(fnodes, 0)
    fedges = torch.stack(fedges, 0)
    nodes_graph = torch.zeros(total_nodes,MAX_NBR).long()
    edges_graph = torch.zeros(total_edges,MAX_NBR).long()

    # nodes_graph[y,:] : for each node y : indices of all edges (z,y) in all_edges
    for a in range(total_nodes):
        for i,b in enumerate(edge_indices[a]):
            nodes_graph[a,i] = b

    # say all_edges[i] holds edge (x,y)
    # then edge_indices[i] holds indices for edges (_,x)
    # then, edges_graph[i] : indices of all edges (_,x) excluding (y,x) 
    for b1 in range(1, total_edges):
        x,y = all_edges[b1]
        for i,b2 in enumerate(edge_indices[x]):
            if all_edges[b2][0] != y:
                edges_graph[b1,i] = b2

    return fnodes,fedges,nodes_graph,edges_graph,scope

```

## Define the Graph Encoder class 

```python
class Graph_Encoder(nn.Module):

    def __init__(self, hidden_size, depth):
        super(Graph_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
      
        #Trainable parameters for the encoding network.
        self.W_i = nn.Linear(NODE_FDIM + EDGE_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(NODE_FDIM + hidden_size, hidden_size)

        
    def forward(self, GRAPHS):
        fnodes,fedges,nodes_graph,edges_graph,scope = GRAPHS
        fnodes = Variable(fnodes,requires_grad=False).cuda()
        fedges = Variable(fedges,requires_grad=False).cuda()
        nodes_graph = Variable(nodes_graph,requires_grad=False).cuda()
        edges_graph = Variable(edges_graph,requires_grad=False).cuda()

        binput = self.W_i(fedges) # no_edges x hidden_size
        message = nn.ReLU()(binput) # no_edges x hidden_size

        #Starting to loop, is this the loopy belief propagation ?
        for i in range(self.depth - 1):
            #get the message vectors for each edge in a no_edges x MAX_NBR x hidden_size tensor.
            nei_message = index_select_ND(message, 0, edges_graph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = nn.ReLU()(binput + nei_message)

        nei_message = index_select_ND(message, 0, nodes_graph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fnodes, nei_message], dim=1)
        #hidden states for each node, size = no_of_nodes x hidden_size
        nodes_hidden_states = nn.ReLU()(self.W_o(ainput))

        graph_vecs = []
        #scope = (start,len)
        for start,length in scope:
            #select the hidden states of all nodes for each graph and get mean hidden state
            node_mean_vec = nodes_hidden_states.narrow(0, start, length).sum(dim=0) / length
            graph_vecs.append(node_mean_vec)

        graph_vecs = torch.stack(graph_vecs, dim=0)
        return graph_vecs
```

## Create Graph Encoder model and encode a simple batch of graphs. 

```python
hidden_size = 64
depth = 8
model_graph_encoder = Graph_Encoder(hidden_size, depth).cuda() #the Graph_Encoder model
graph_dataset =  GraphSet(all_graphs) #dataset

#prepare a dummy batch to see how a single batch of graph encodings is generated.
batch = []
batch.append(graph_dataset[0])
batch.append(graph_dataset[1]) # batch_size = 2

#convert the batch of graphs to [fnodes,fedges,nodes_graph,edges_graph,scope] for passing thru Graph_Encoder.
GRAPHS = process_all_graphs(batch, NODE_FDIM,EDGE_FDIM)
graph_vec = model_graph_encoder(GRAPHS)
print('Got encoded graph_vec = \n',graph_vec.data.shape,'\n',graph_vec.data)
```
## Add loss, optimizers and code to train the above model
TBD
