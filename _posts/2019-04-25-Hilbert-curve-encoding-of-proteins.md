---
layout: post
title: Encode a protein to a image using Hilbert curves. 
categories: ['Deep Learning']
---

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import itertools

# generate elements of the sequence
def element(seq_list):
    list_ = []
    for s in seq_list:
        if s not in list_:
            list_.append(s)
    return list_

# generate mapping dictionary
def combination(elements, seq_length):
    keys = map(''.join, itertools.product(elements, repeat=seq_length))
    n_word = len(keys)
    array_word = np.eye(n_word)
    mapping_dic = {}
    for i in range(n_word):
        mapping_dic[keys[i]] = array_word[i,:]
    return mapping_dic

def hilbert_curve(n):
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n // 2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


#one hot encoding
def one_hot(sequence, sub_len, mapping_dic):
    n_ = len(sequence)
    sub_list = []
    for i in range(n_ - sub_len + 1):
        sub_list.append(sequence[i:i + sub_len])
    res_ = []
    for sub in sub_list:
        res_.append(mapping_dic[sub])
    return np.array(res_)

#assign each pixel a one-hot encoding.
def plot_hb_dna(seq, H_curve, sub_length,map_dic):
    r, c = H_curve.shape
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = np.zeros((r, c, 20 ** sub_length))
    for i in range(len(num_A)):
        x, y = np.where(H_curve == i)
        H_dna[x, y, :] = num_A[i, :]
    return H_dna

#assign each pixel a 1 to check if all 1-mers are present in the image.
def plot_hb_dna_check(seq, H_curve, sub_length,map_dic):
    r, c = H_curve.shape
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = np.zeros((r,c))
    for i in range(len(num_A)):
        x, y = np.where(H_curve == i)
        H_dna[x,y] = 1
    return H_dna

#input protein.
protein_seq = 'MANMNNTKLNARALPSFIDYFNGIYGFATGIKDIMNMIFKTDTGGNLTLDEILKNQQLLNMMNNPPAARRYYFFEISGKLDGVNGSLNDLIAQGNLNTELSKEILKIANEQNQVLNDVNNKLDAINTMLHIYLPKITSMLSDVMKQNYALSLQVEYLSKQLKEISDKLDVINVNVLINSTLTEITPAYQRIKYVNEKFEELTFATETTLKVKKDSSPADILDELTELTELARSVTRNDMESFEFYIKTFHDVMIGNNLFSRSALKTASELIAKENIHTRGSEIGNVYTFMIVLTSLQAKAFLTLTTCRKLLGLADIDYTQIMNENLDREKEEFRLNILPTLSNDFSNPNYTETLGSDLVDPIVTLEAEPGYALIGFEILNDPLPVLKVFQAKLKQNYQVDKESIMENIYGNIHKLLCPKQREQKYYIKDITFPEGYVITKIVFEKKLNLLGYEVTANLYDPFTGSIDLNKTILESWKEDCCEEDCCEEDCCEENCCEEDYIKLMPLGVISETFLTPIYSFKLIIDKKTKKISLAGKSYLRESLLATDLVNKETNLIPSPNGFISSIVQTWHITSDNIEPWEANNKNAYVDKTDTMVGFSSLYTHKDGEFLQFIGAKLKPKTEYVIQYTVKGKPSIHLKDENTGYILYEDTNNDLEDFQTITKRFTTGTDLMRVYLILKSQSGHEAWGDNFTILEIKPAEALVSPELINPNSWITTQGASISGDKLFISLGTNGTFRQNLSLNSYSTYSISFTASGPFNVTVRNSREVLYERNNLMSSTSHISGEFKTESNNTGLYVELSRRSGGAGHISFENISIK'
print('protein len = ',len(protein_seq))

#20 amino acids
elements= ['M', 'A', 'N', 'T', 'K', 'L', 'R', 'P', 'S', 'F', 'I', 'D', 'Y', 'G', 'E', 'Q', 'V', 'H', 'C', 'W']

#choose kmer size = 1. So each one-hot encoding will be of len 20.  
sub_length = 1 #k-mer size.
mapping_dic = combination(elements, sub_length)

print('\nMapping dictionary:')
#each 1-mer and its encodings
for k,v in mapping_dic.iteritems():
    print(k,v)
```

## Output

```python
protein len =  816

Mapping dictionary:
A [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
C [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
E [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
D [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
G [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
F [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
I [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
H [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
K [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
M [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
L [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
N [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Q [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
P [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
S [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
R [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
T [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
W [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
V [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
Y [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
```

```python
# The Hilbert curve yields a square image of size 2^n × 2^n = 2^(2n), where n is the order of the curve.
# Choose order of Hilbert curve to accommodate all the k-mers in the image. 
# The count of kmers is roughly the length of the protein sequence.
# So for protein of length 816, we choose order = 5. Then Hilber image size = 2^(10) = 1024 >= 800
# Each pixel in the Hilber image will be one-hot encoding of a k-mer.
# In our case, k=1, so each pixel is one-hot encoded of len = 20. 
# So final image size : 32 x 32 x 20.

order = 32

H = hilbert_curve(order)

print('Hilbert matrix=\n',H)

X = plot_hb_dna_check(seq=protein_seq, H_curve=H, sub_length=sub_length, map_dic=mapping_dic)

seq_image = plot_hb_dna(seq=protein_seq, H_curve=H, sub_length=sub_length, map_dic=mapping_dic)
```
## Output

```python
Hilbert matrix=
 [[   0    1   14 ...  339  340  341]
 [   3    2   13 ...  338  343  342]
 [   4    7    8 ...  349  344  345]
 ...
 [1019 1016 1015 ...  674  679  678]
 [1020 1021 1010 ...  685  680  681]
 [1023 1022 1009 ...  684  683  682]]
 ```

```python
#check if all 1-mers in the sequence are assigned to the image.
z = 0
for r in range(len(X)):
    z += sum(X[r,:])
    print(','.join([str(int(x))for x in X[r,:]]))
print("Total no of 1's =",z) #should be same as the protein seq len to ensure all 1-mers are in the image.
```

## Output

```python
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
Total no of 1's = 816.0
```

```python
print('Final image shape:')
print(seq_image.shape)
```
## Output

```python
Final image shape:
(32, 32, 20)
```

## Full notebook

[Notebook](https://github.com/pchanda/pchanda.github.io/blob/master/data/Encode_protein_hilbert_curve.ipynb)

