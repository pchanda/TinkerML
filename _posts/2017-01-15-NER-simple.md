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
