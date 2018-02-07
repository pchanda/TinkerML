---
layout: post
title: Simple Named entity Recognition (NER) with tensorflow
---

Given a peice of text, NER seeks to identify named entities in text and classify them  into various categories such as names of persons, organizations, locations, expressions of times, quantities, percentages, etc. Here we just want to build a model to predict $$N_c = $$ 5 classes for every word in a sentence: PER (person), ORG (organization), LOC (location), MISC (miscellaneous) and O(null class, not a NER).  

## Reading the training data
For training, use the file 'train.conll'. A snapshot of the file looks like: 

```
-DOCSTART-      O

EU      ORG
rejects O
German  MISC
call    O
to      O
boycott O
British MISC
lamb    O
.       O

The     O
European        ORG
Commission      ORG
said    O
on      O
Thursday        O
it      O
disagreed       O
with    O
German  MISC
```
Each line has a word and the associated class (one of the 5 described above). A blank line separates two sentences. The following code implements a reader for conll files : 

```python
def read_conll_file(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    ret = []
    current_toks, current_lbls = [], []
    for line in fstream:
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
            current_toks, current_lbls = [], []
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
            tok, lbl = line.split("\t")
            current_toks.append(tok)
            current_lbls.append(lbl)
    if len(current_toks) > 0:
        assert len(current_toks) == len(current_lbls)
        ret.append((current_toks, current_lbls))
    return ret
```
For every sentence, this will return a list of [ words, labels] where words is a list of words in the sentence and labels contains the NER labels. For example element 0 in the list returned is
```
(['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],['ORG', 'O', 'MISC', 'O', 'O', 'O', 'MISC', 'O', 'O'])
```

### Tokenize the training data, create sliding windows etc.
Next we will convert each word to a unique integer id. First we add some global definitions:

```python
START_TOKEN = "<s>"
END_TOKEN = "</s>"
NUM = "nnnummm"
UNK = "uuunkkk"
LBLS = ["PER","ORG","LOC","MISC","O"]
WINDOW_SIZE = 2
```

Next is the code to convert words to integer ids. Note a word dictionarty word_dict will be built in the process if word_dict is empty, otherwise new {word,id} pairs will get added to word_dict

```python
def words_to_ids(data,word_dict):
    # Preprocess data to construct an embedding
    if len(word_dict)==0:
       offset = 0
       print('word_dict offset = ',offset)
    else:
       offset = max(word_dict.values())

    for sentence, _ in data :
       for word in sentence :
           if word.isdigit(): word = NUM
           else: word = word.lower()
           index = word_dict.setdefault(word,offset)
           offset = offset if index<offset else (offset+1)

    offset = i+1
    for i,word in enumerate([START_TOKEN, END_TOKEN, UNK],offset):
        word_dict.setdefault(word,i)

    sentences_ = []
    labels_ = []

    for sentence, label in data:
       s = []
       k = 0
       for word in sentence:
           if word.isdigit(): word = NUM
           else: word = word.lower()
           s += [word_dict.get(word, word_dict[UNK])]
           k += 1
       sentences_ += [s]
       labels_ += [[LBLS.index(l) for l in label]]
       
    return (sentences_,labels_)
```

Next we define the function to create sliding windows of size 'window_size'. The input has already the words and labels converted to integer ids:

```python
def make_windowed_data(data, start, end, window_size=1):
    # ensure data has both sentences and labels
    assert len(data)==2, 'data should be a tuple = (list of sentences, list of labels)'
    sentence_list = data[0] # sentence as a list of tokens e.g. [1,2,3,4]
    label_list = data[1]    # labels as a list of tokens   e.g. [0,1,1,4]

    orig_len = len(sentence_list)

    #extend the sentence_list with start and end tokens
    sentence_list = window_size*[start] + sentence_list + window_size*[end]

    output_list = []
    for i in range(window_size,window_size+orig_len):
      sentence_slice = sentence_list[i-window_size:i+window_size+1]
      label = label_list[i-window_size]
      tuple = (sentence_slice,label)
      output_list.append(tuple)

    return output_list
```

Finally the following functions puts all the elements described above, together. 'to_string' function just strips square brackets and curly braces from a list. So if the input string is '([87, 11, 0, 1, 2], 4)', the output will be `87, 11, 0, 1, 2; 4`. Note that ';' serves as delimiter between the word tokens and label token. 

```python
def to_string(s):
    # input example : s = ([87, 11, 0, 1, 2], 4)
    return str(s).strip('()').strip('[').replace('],',';')

def process_sentences_and_labels(data,window_size,word_dict=None):
    if word_dict is None:
      word_dict = {}
    data = words_to_ids(data,word_dict)
  
    start_token = word_dict[START_TOKEN]
    end_token = word_dict[END_TOKEN]

    sentences_ = data[0] # list of tokenized sentences e.g. [[1,2,3,4],[5,6,7,8,9],[3,4,5],....]
    labels_ = data[1]    # list of tokenized labels    e.g. [[0,1,1,4],[0,0,3,3,4],[4,4,2],....]
    windowed_data = []
    for s,l in zip(sentences_,labels_):
       list_of_windows = make_windowed_data((s,l), start_token, end_token, window_size)
       # each window in list_of_windows is a tuple
       windowed_data += list_of_windows

    windowed_data_string = map(to_string,windowed_data)
    return (word_dict,windowed_data_string)
```

You can quickly test the code as:

```python
# main
vocab_fstream = open('sample.conll','r')
data = read_conll_file(vocab_fstream)
vocab_fstream.close()
word_dict,windowed_data = process_sentences_and_labels(data,WINDOW_SIZE)
print windowed_data
print word_dict
```

### Tensorflow queues for creating input pipeline to read the training data
Next we will use tensorflow queues to read the training data. This will take care of batching and shuffling. First lets add the necessary definitions:

```python
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import cPickle as pickle

base_path = '/home/pchanda/deep_learning/NLP/word_classification_1/'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,"Number of windows to process in a batch.")
tf.app.flags.DEFINE_string('data_dir', os.path.join(base_path,'data/'),"""Path to data directory""")
tf.app.flags.DEFINE_string('model_dir', os.path.join(base_path,'saved_models/'),"""Path to saved models during training""")
tf.app.flags.DEFINE_integer('num_examples_train', 204000,"Total number of pairs per epoch for training")
tf.app.flags.DEFINE_integer('num_examples_test', 51362,"Total number of pairs per epoch for testing")
tf.app.flags.DEFINE_integer('vocabulary_size',50000,"vocabulary size")
tf.app.flags.DEFINE_integer('embedding_dimension', 50,"dimension of the word vectors")
tf.app.flags.DEFINE_integer('hidden_dim', 200,"dimension of hidden layers")
tf.app.flags.DEFINE_integer('number_of_classes', 5,"no of NER classes (including null or O)")
tf.app.flags.DEFINE_integer('max_to_keep', 50,"max models to retain")
tf.app.flags.DEFINE_integer('learning_rate', 0.001,"optimizer learning rate")

```

Note that you will need to make a specific directory structure for the above code. Also change the 'base_path' to your preferred directory and create subdirectories ./data and ./saved_models to hold the data files and models during training. The code below will read the data using tensorflow queue and the functions defined above. It will also shuffle them to create batches of size 'batch_size'. Note, the pre-processing (tokenizing, window creation etc) are done by calls to functions defined above. 

```python
FLAGS = tf.app.flags.FLAGS

def _generate_batch(part_0,part_1, min_queue_examples,batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    line_value_batch = tf.train.shuffle_batch(
        [part_0,part_1],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    line_value_batch = tf.train.batch(
        [part_0,part_1],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  return line_value_batch



def get_input(data_filename, batch_size, num_examples, shuffle, word_dict=None):
  # word_dict should be populated during testing to use the prebuilt dictionary made during testing.
  vocab_fstream = open(data_filename,'rb')
  data = process_vocabulary.read_conll_file(vocab_fstream)
  vocab_fstream.close()
  word_dict,windowed_data_string = process_vocabulary.process_sentences_and_labels(data,WINDOW_SIZE,word_dict)

  windowed_data_tensor = tf.convert_to_tensor(windowed_data_string, dtype=tf.string)
  input_queue = tf.train.slice_input_producer([windowed_data_tensor],shuffle=shuffle)

  line_value = input_queue[0]
  line_value_parts = tf.decode_csv(line_value, record_defaults=[['NA']]*2,field_delim=";")
  part_0 = tf.decode_csv(line_value_parts[0], record_defaults=[['.']]*(2*WINDOW_SIZE+1),field_delim=",")
  part_1 = line_value_parts[1]

  part_0 = tf.string_to_number(part_0,out_type=tf.int64)
  part_1 = tf.string_to_number(part_1,out_type=tf.int64)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples * min_fraction_of_examples_in_queue)

  # Generate a batch by building up a queue of examples.
  batch =  _generate_batch(part_0,part_1,min_queue_examples, batch_size,shuffle=shuffle)
  return batch,word_dict
```

## Define the model for training

Now with the reading and input queues taken care of, lets define our model and train. Input for training is a batch of windows 'batch' in the return value of the function 'get_input(...)' above. Lets add the ops to read the data.

### Definitions

```python
FLAGS = tf.app.flags.FLAGS

input_filename = os.path.join(FLAGS.data_dir,'train.conll')
# read the vocabulary file and pre-trained embedding file on some of the words.
vocab_file = os.path.join(FLAGS.data_dir,'vocab.txt')
# read the embedding matrix, contains pre-trained word vectors for each word in the vocabulary 
embedding_file = os.path.join(FLAGS.data_dir,'wordVectors.txt')
# add ops to read data
line_batch,word_dict = get_input(input_filename,FLAGS.batch_size, FLAGS.num_examples_train, True)

#save the dictionary to use during testing
with open(os.path.join(FLAGS.data_dir,'word_dict_train.pkl'),'wb') as fptr:
   pickle.dump(word_dict,fptr)
   
```

### Read pre-trained word vectors

Each word (and integer) will be one-hot encoded and used to pull out the corresponding word vector from the embedding matrix $$Word\_Vectors$$ (or the matrix of word vectors (shape: vocabulary_size x embedding_dimension)). The function 'load_embedding_matrix(...)' will read the pre-trained embedding matrix of the words present in our vocabulary. The pre-trained embedding matrix will serve to initialize the $$Word\_Vectors$$ matrix, which will also get updated during our training, more onto this later. 

```python
def load_word_embeddings(word_dict,vocab_file,embedding_file):
   #read the embeddings
   embeddings = np.array(np.random.randn(FLAGS.vocabulary_size, FLAGS.embedding_dimension), dtype=np.float32)
   fstream_1 = open(vocab_file,'rb')
   fstream_2 = open(embedding_file,'rb')
   for word,vector in zip(fstream_1,fstream_2):
      word = word.strip()
      if word.isdigit(): word = NUM
      else: word = word.lower()
      vector = np.array(list(map(float,vector.strip().split(" "))))
      if word in word_dict:
          embeddings[word_dict[word]] = vector
   fstream_1.close()
   fstream_2.close()
   print 'load_word_embeddings : vocabulary size = ',FLAGS.vocabulary_size, ' embedding_shape = ',embeddings.shape
   return embeddings

embeddings = utils.load_word_embeddings(word_dict,vocab_file,embedding_file) # read pre-trained word embeddings.
```

### Create the model

Now comes the model creation. Note that '$$\cdot$$' indicates matrix multiplication. Breifly, given a input window $$ \textbf{x} = [ x^{(t-w)}, x^{(t)}, ... ; x^{(t+w)}]$$ (center word is $$x^{(t)}$$) model is :

Chanda

$$
\begin{align}
& word\_embeddings(t) = \textbf{x} \cdot Word\_Vectors \\\\
& h(t) = ReLU( word\_embeddings(t) \cdot W\_matrix + b_1) \\\\
& \hat{y}^{(t)} = softmax(h(t) \cdot U\_matrix + b_2) \\\\
& Loss(t) = CrossEntropy(y^{(t)}, \hat{y}(t))
\end{align}
$$

Each $$x^{t}$$ is one-hot encoded (vector of dimension = vocabulary_size). The $$\textbf{x}$$ vector is a concatenation of all the vectors $$x^{(t-w)}, x^{(t)}, ... ; x^{(t+w)}$$. So $$\textbf{x}$$ has dimension $$ \tilde d = $$ (2 x WINDOW_SIZE + 1) x vocabulary_size, as each window has (2 x WINDOW_SIZE + 1) words.

The embeddings read above are used to initialize the $$Word\_Vectors$$ matrix (dimensions vocabulary_size x embedding_dimension). The $$W\_matrix$$ ($$ \tilde d \times d_h $$) are the weights from the embedding layer (line 1 in model above) to the hidden layer (line 2 in model above) with $$d_h$$ neurons. The $$U\_matrix$$ ($$ d_h \times N_c $$) are the weights from the hidden layer (2. in model above) to the output layer (line 3 in model above).   

Define utility functions to create weight and bias tensors:
```python
def get_weights(name, shape, stddev=0.02, dtype=tf.float32, initializer=None):
    if initializer is None:
      initializer = tf.truncated_normal_initializer(stddev = stddev, dtype=dtype)
    return tf.get_variable(name, shape, initializer = initializer, dtype=dtype)

def get_biases(name, shape, val=0.0, dtype=tf.float32, initializer=None):
    if initializer is None:
       initializer = tf.constant_initializer(val)
    return tf.get_variable(name, shape, initializer = initializer, dtype=dtype)
```

### Code for the model
```python
window_size = WINDOW_SIZE
no_of_words_per_window =  2*window_size+1
stacked_window_dim = no_of_words_per_window*FLAGS.embedding_dimension

Word_Vectors = tf.Variable(embeddings,name='Word_Embedding_Matrix')
W_matrix = utils.get_weights(name='W_Matrix',shape=[stacked_window_dim,FLAGS.hidden_dim])
W_biases = utils.get_weights('hidden_biases',shape=[FLAGS.batch_size,FLAGS.hidden_dim])

U_matrix = utils.get_weights('U_Matrix',shape=[FLAGS.hidden_dim,FLAGS.number_of_classes])
U_biases = utils.get_weights('output_biases',shape=[FLAGS.batch_size,FLAGS.number_of_classes])

window = line_batch[0]
labels = line_batch[1]

one_hot_words = tf.one_hot(window,depth=FLAGS.vocabulary_size,dtype=tf.float32)
words_reshaped = tf.reshape(one_hot_words,[-1,FLAGS.vocabulary_size])

word_embeddings = tf.matmul(words_reshaped,Word_Vectors)
word_embeddings_stacked_per_window = tf.reshape(word_embeddings,shape=[-1,stacked_window_dim]) # shape = [batch_size x stacked_window_dim]

hidden_layer = tf.nn.relu(tf.matmul(word_embeddings_stacked_per_window,W_matrix) + W_biases,name='hidden_layer')
logits = tf.nn.softmax(tf.matmul(hidden_layer,U_matrix) + U_biases,name='logits')  # should be of shape = [batch_size x no.of.classes]

#loss
loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")))
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

#predicted NER labels
predicted_labels = tf.argmax(logits,axis=1)
```

### Precision-Recall 
Precision is calculated as the ratio of correct non-null labels predicted to the total number of non-null labels predicted. Recall is calculated as the ratio of correct non-null labels predicted to the total number of correct non-null labels. $$F_1$$ is the harmonic mean of the two.

```python
def precision_recall(predicted,actual):
   no_correct_nonnull_predictions = np.sum([x==y and x!=defs.LBLS.index('O') for (x,y) in zip(predicted,actual)])
   no_nonnull_predictions = np.sum([x!=defs.LBLS.index('O') for x in predicted])
   no_nonnull_predictions = (no_nonnull_predictions+1) if no_nonnull_predictions==0 else no_nonnull_predictions
   no_nonnull_labels = np.sum([y!=defs.LBLS.index('O') for y in actual])
   no_nonnull_labels = (no_nonnull_labels+1) if no_nonnull_labels==0 else no_nonnull_labels
   precision = no_correct_nonnull_predictions/no_nonnull_predictions
   recall = no_correct_nonnull_predictions/no_nonnull_labels
   return precision,recall
```

### Create session for training

Now lets fire up the tensorflow training session with the following code: 

```python
saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep) #to save the models duing training.
with tf.Session() as sess:
     init = tf.initialize_all_variables()
     sess = tf.Session()
     sess.run(init)
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)

     for step in range(10000):
        _,loss_v,predictions,actual = sess.run([train_step,loss,predicted_labels,labels])
        pr,re = utils.precision_recall(predictions,actual)
        f1 = 2*pr*re/(pr + re + 1e-10)
        print step,' loss=',loss_v,' pr=',pr,' re=',re,' F1=',f1
        #print zip(predictions,actual)
        print '----------------------------------------'

        if step % 500 == 0 and step>=500:  #save model every 500 steps
           #save the model
           ckptfile = FLAGS.model_dir+'model_'+str(step)+'.ckpt'
           ckptfile = saver.save(sess, ckptfile)

     coord.request_stop()
     coord.join(threads)
```

## Code for testing
The model should train with precision and recall of more the 0.9 each and $$F_1$$ more than 0.8. Now onto testing. The code below follows in the line of the training code above to define variables and create the model. But we must read the word_dict created during training (saved in file 'word_dict_train.pkl') to get the correct integer ids for each word during testing.


```python
from __future__ import division
import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import pandas as pd

FLAGS = tf.app.flags.FLAGS

#read file to test
input_filename = os.path.join(FLAGS.data_dir,'dev.conll')
#read word_dict saved during training
with open(os.path.join(FLAGS.data_dir,'word_dict_train.pkl'),'rb') as fptr:
   word_dict = pickle.load(fptr)

line_batch,word_dict = get_input(input_filename,FLAGS.batch_size, FLAGS.num_examples_test, False, word_dict)

window_size = defs.WINDOW_SIZE
no_of_words_per_window =  2*window_size+1
stacked_window_dim = no_of_words_per_window*FLAGS.embedding_dimension

Word_Vectors = tf.get_variable(name='Word_Embedding_Matrix',shape=[FLAGS.vocabulary_size,FLAGS.embedding_dimension])
W_matrix = utils.get_weights(name='W_Matrix',shape=[stacked_window_dim,FLAGS.hidden_dim])
W_biases = utils.get_weights('hidden_biases',shape=[FLAGS.batch_size,FLAGS.hidden_dim])

U_matrix = tf.get_variable('U_Matrix',shape=[FLAGS.hidden_dim,FLAGS.number_of_classes])
U_biases = tf.get_variable('output_biases',shape=[FLAGS.batch_size,FLAGS.number_of_classes])

window = line_batch[0]
labels = line_batch[1]

one_hot_words = tf.one_hot(window,depth=FLAGS.vocabulary_size,dtype=tf.float32)
words_reshaped = tf.reshape(one_hot_words,[-1,FLAGS.vocabulary_size])

word_embeddings = tf.matmul(words_reshaped,Word_Vectors)
word_embeddings_stacked_per_window = tf.reshape(word_embeddings,shape=[-1,stacked_window_dim]) # shape = [batch_size x stacked_window_dim]

hidden_layer = tf.nn.relu(tf.matmul(word_embeddings_stacked_per_window,W_matrix) + W_biases,name='hidden_layer')
#loss
loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")))

#predicted NER labels
predicted_labels = tf.argmax(logits,axis=1)
```

### Create session for testing
Run the tensorflow testing session with the following code. Do not forget to do 'saver.restore' which will load the tensors defined above with values saved during training from the model file:

```python
saver = tf.train.Saver()
with tf.Session() as sess:
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
     saver.restore(sess, FLAGS.model_dir+'model_9000.ckpt')

     pr_labels = []
     act_labels = []
     for step in range(5000):
        loss_v,predictions,actual = sess.run([loss,predicted_labels,labels])
        pr,re = utils.precision_recall(predictions,actual)
        f1 = 2*pr*re/(pr + re + 1e-10)
        pr_labels += list(predictions)
        act_labels +=  list(actual)
        print step,' loss=',loss_v,' pr=',pr,' re=',re,' F1=',f1
        #print zip(predictions,actual)
        print '----------------------------------------'

     pr,re = utils.precision_recall(predictions,actual)
     f1 = 2*pr*re/(pr + re + 1e-10)
     cf = pd.crosstab(pd.Series(act_labels), pd.Series(pr_labels), rownames=['True'], colnames=['Predicted'], margins=True)
     print 'precision = ',pr,' recall = ',re, ' f1=',f1
     print 'Confusion Matrix = ',cf
     coord.request_stop()
     coord.join(threads)
```

This should result in precision of about 0.83, recall of 0.83 and $$F_1$ score of about 0.75 (Not bad for the simple model!!!).
