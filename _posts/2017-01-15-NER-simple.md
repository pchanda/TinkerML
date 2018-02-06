---
layout: post
title: Simple Named entity Recognition (NER) with tensorflow
---

Given a peice of text, NER seeks to identify named entities in text and classify them  into various categories such as names of persons, organizations, locations, expressions of times, quantities, percentages, etc. Here we just want to build a model to predict 5 classes for every word in a sentence: PER (person), ORG (organization), LOC (location), MISC (miscellaneous) and O(null class, not a NER).  

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

The following code implements a reader for conll files : 

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
