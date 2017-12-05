---
layout: post
title: Gibbs Random Field (GRF)
---

Lets see what is a GRF and how it is connected to the MRF. If you are new to my blogs, please visit my blog on [MRF](https://pchanda.github.io/MarkovRandomFields/) to get familiar with the notations. 
A GRF can be thought of as a graphical representation of a family of random variables \\( \textbf{X} =\\{ X_1,X_2,...,X_n \\} \\) on set \\( S=\\{1,2,...,n\\} \\) of sites.The relationship between the random variables is defined using a neighborhood system. A GRF obey's Gibb's distribution given by,

$$ P( \textbf{x} ) = P(x_1,x_2,...,x_n) = \dfrac{1}{Z}\exp\{ -\dfrac{1}{T} U(\textbf{x}) \} $$

Here, $$Z$$ is a normalizing constant,also called a partition function whose job is to ensure $$ P( \textbf{x} ) $$ is between 0 and 1, obtained by summing all possible configurations of value assignment to the $$n$$ random variables,

$$ Z = \sum\limits_{\textbf{x}} \exp\{ -\dfrac{1}{T} U(\textbf{x}) \} $$.

$$T$$ is the temperature, we will not worry about it here, just set it to 1. 

$$U(\textbf{x})$$ is called the energy function defined using cliques that obey the neighborhood system. Note that smaller the energy $$U(\textbf{x})$$, higher is the probability and vice-versa. We will see that one of the goals will be to find parameters and values to be assigned to the random variables that will lower the energy function.

A clique is just a set of nodes (random variables) in a graph that are all connected (neighbors) with each other. Let $$C_i$$ be the set of all cliques of size $$i$$. 

So $$ C_1 = \{(i) \vert i \in S\} $$;  

$$ C_2 = \{(i,j) \vert i \in S, j \in N_i \} $$, $$N_i=\{$$ all neighbors of $$ i\} $$;

$$ C_3 = \{(i,j,k) \vert i,j,k \in S, i,j,k $$ are all  neighbors of each other $$ \} $$;

And so on ...

Thus $$ C = \{ C_1 \cup C_2 \cup C_3 ... $$ higher order cliques $$\}$$. 

Now define something called a potential function $$\phi(...) $$ for each of the cliques. So the energy function $$ U(\textbf{x})$$ is represented as a sum of clique potentials over cliques of many sizes,

$$ U(\textbf{x}) = \sum\limits_{c \in C} \phi_{c}(\textbf{x}) $$
 = (sum of potentials of all cliques of size 1) + (sum of potentials of all cliques of size 2) + ... 

Therefore, 
$$
\begin{align}
P( \textbf{x} ) & = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \} \\\\
& = \dfrac{1}{Z}\exp \{ -\sum\limits_{c \in C} \phi_{c}(\textbf{x}) \} \\\\
& = \dfrac{1}{Z} \prod\limits_{c \in C} \exp \{ -\phi_{c}(\textbf{x}) \} 
\end{align}
$$

Each term \\( \exp \{ -\phi_{c}(.) \} \\) must be positive, but may not sum to 1, hence we need the partition function $$Z$$ make this a valid probability distribution. 

Also, for any Gibbs Field, there is a subset $$\hat C$$ of $$C$$ consisting of only maximal cliques which are not proper subsets of any other cliques. We can write a clique potential for each maximal clique that is the product of the exponentials of potentials of all its sub-cliques. In this way, we can write the joint probability using only the potentials of the maximal cliques:

$$ P( \textbf{x} ) = \dfrac{1}{Z} \prod\limits_{c \in \hat C} \exp \{ -\phi_{c}(\textbf{x}) \} $$

with the normalizing factor Z defined as,

$$ Z = \sum\limits_{\textbf{x}} \prod\limits_{c \in \hat C} \exp \{ -\phi_{c}(\textbf{x}) \}  $$.

Note that this is defined over all maximal cliques, not just the single largest one. We usually take these potentials to be only functions over the maximal cliques.

Ok, the GRF math is pretty boring, whats its connection with a MRF ? It turns out that GRF and MRF are connected by the Hammersley-Clifford theorem which states that the set of random variable $$\textbf{X}$$ is an MRF wrt a neighborhood system iff $$\textbf{X}$$ is a GRF wrt the same neighborhood system. 

Bye, the rest of it comes tomorrow :-)

