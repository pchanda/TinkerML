---
layout: post
title: Gibbs Random Field (GRF)
---

Lets see what is a GRF and how it is connected to the MRF. If you are new to my blogs, please visit my blog on [MRF](https://pchanda.github.io/MarkovRandomFields/) to get familiar with the notations. 
A GRF can be thought of as a graphical representation of a family of random variables \\( \textbf{X} =\\{ X_1,X_2,...,X_n \\} \\) on set \\( S=\\{1,2,...,n\\} \\) of sites.The relationship between the random variables is defined using a neighborhood system. A GRF obey's Gibb's distribution given by,

$$ P( \textbf{x} ) = P(x_1,x_2,...,x_n) = \dfrac{1}{Z}\exp\{ \dfrac{1}{T} U(\textbf{x}) \} $$

Here, $$Z$$ is a normalizing constant,also called a partition function whose job is to ensure $$ P( \textbf{x} ) $$ is between 0 and 1, obtained by summing all possible configurations of value assignment to the $$n$$ random variables,

$$ Z = \Sigma \limits_{\textbf{x}} \exp\{ \dfrac{1}{T} U(\textbf{x}) \} $$.

Let \\( \textbf{X} =\\{ X_1,X_2,...,X_n \\} \\) be a family of random variables defined on a set \\( S=\\{1,2,...,n\\} \\) of sites. 
As an example, \\( S \\) can represent the pixel positions of an \\( m \times m \\) image in a  2-D lattice \\( \\{(i,j) | 1 \leq i,j \leq m\\} \\) where the double indexing can be recoded to univariate indexing by \\( (i,j) \rightarrow (i-1)m+j \\) so that \\( S=\\{ 1,2,...,m^2 \\} \\). 
