---
layout: post
title: Markov Random Fields (MRF)
---

A short intro to MRFs. Let \\( \{X_1,X_2,...,X_n\} \\) be a family of random variables defined on a set \\( S=\{1,2,...,n\} \\) of sites. 
As an example, \\( S \\) can represent the pixel positions of an image in a \\( m \times m \\) 2-D lattice \\( \{(i,j) | 1 \leq i,j \leq m\} \\) where the double indexing can be recoded to univariate indexing by \\( (i,j) \rightarrow (i-1)m+j \\) so that \\( S=\{1,2,...,m^2\} \\). The family \\( X \\) is called a \textbf{Random Field}.
