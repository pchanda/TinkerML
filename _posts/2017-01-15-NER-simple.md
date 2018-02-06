---
layout: post
title: Simple Named entity Recognition (NER) with tensorflow
---

Given a peice of text, NER seeks to identify named entities in text and classify them  into various categories such as names of persons, organizations, locations, expressions of times, quantities, percentages, etc. Here we just want to build a model to predict 5 classes for every word in a sentence: PER (person), ORG (organization), LOC (location), MISC (miscellaneous) and O(null class, not a NER).  

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

Next we will convert each word to a unique integer id. First we add some global definitions:

```python
START_TOKEN = "<s>"
END_TOKEN = "</s>"
NUM = "nnnummm"
UNK = "uuunkkk"
LBLS = ["PER","ORG","LOC","MISC","O"]
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
           if word.isdigit(): word = defs.NUM
           else: word = word.lower()
           index = word_dict.setdefault(word,offset)
           offset = offset if index<offset else (offset+1)

    offset = i+1
    for i,word in enumerate([defs.START_TOKEN, defs.END_TOKEN, defs.UNK],offset):
        word_dict.setdefault(word,i)

    sentences_ = []
    labels_ = []

    for sentence, label in data:
       s = []
       k = 0
       for word in sentence:
           if word.isdigit(): word = defs.NUM
           else: word = word.lower()
           #print(word,word_dict.get(word,defs.UNK),label[k])
           #sentences_ += [[word_dict.get(word, word_dict[UNK]), word_dict[P_CASE + casing(word)]]]
           s += [word_dict.get(word, word_dict[defs.UNK])]
           k += 1
       sentences_ += [s]
       labels_ += [[defs.LBLS.index(l) for l in label]]
       
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

Finally the following functions puts all together. 'to_string' function just strips square brackets and curly braces from a list. So if the input string is '([8711, 8711, 0, 1, 2], 4)', the output will be `8711, 8711, 0, 1, 2; 4`. Note that ';' serves as delimiter between the word tokens and label token. 

```python
def to_string(s):
    # s = ([8711, 8711, 0, 1, 2], 4)
    return str(s).strip('()').strip('[').replace('],',';')

def process_sentences_and_labels(data,window_size,word_dict=None):
    if word_dict is None:
      word_dict = {}
    data = words_to_ids(data,word_dict)
    #print 'tokenized data : ',data

    #start_token = [word_dict[START_TOKEN],word_dict[P_CASE + "aa"]]
    #end_token = [word_dict[END_TOKEN], word_dict[P_CASE + "aa"]]
    start_token = word_dict[defs.START_TOKEN]
    end_token = word_dict[defs.END_TOKEN]

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
