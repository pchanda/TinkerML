---
layout: post
title: Ising Model
---
Please refer to my earlier posts on [MRF](https://pchanda.github.io/MarkovRandomFields/) and [GRF](https://pchanda.github.io/GibbsRandomField/) for getting used to the notations. If we consider clique potentials for sizes of upto 2, he energy function is the sum of clique potentials of all cliques of size 1 and size 2,

$$ U( \textbf{x} ) = \sum_{\{i\}\in C_1} \phi_1(x_i) + \sum_{\{i,j\}\in C_2} \phi_2(x_i,x_j) $$

Also assume that our label set $$ L $$ is binary, $$L \in \{-1,1\}$$. To further simplify things, assume that $$\phi_1(x_i)=\alpha_i x_i $$ and $$ \phi_2(x_i,x_j) = \beta_{i,j} \delta(x_i,x_j) $$. 

Here $$\beta_{i,j}$$ can be thought of as strengths of pairwise interactions, while $$\alpha_i$$ reflects the bias of $$x_i$$ towards a particular state in preference to the other. 

Further simplify, $$\alpha_i = \alpha, \forall i $$, $$\beta_{i,j} = \beta, \forall i,j $$, and $$\delta(x_i,x_j) = -x_i x_j $$. This gives and example of an [Ising Model](https://en.wikipedia.org/wiki/Ising_model). Then,

$$
\begin{align}
P( \textbf{x} ) & = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \} \\\\
& = \dfrac{1}{Z}\exp \{ -\alpha \sum_{\{i\}\in C_1} x_i  + \beta \sum_{\{i,j\}\in C_2} x_i x_j  \} \\\\
& = \dfrac{1}{Z}  \exp \{ -\alpha \sum_{\{i\}\in C_1} x_i\}  \exp\{ \beta \sum_{\{i,j\}\in C_2} x_i x_j \}   
\end{align}
$$

Drawing samples from $$P(\textbf{x})$$ is tricky, as we do not have a way to compute the partition function $$Z$$. 

## Sampling by Metropolis-Hastings method
1. Start with some configuration $$\textbf{x} = \{x_1,x_2,...x_n\}$$.
Repeat the following steps:
2. Select a random $$x_k$$. 
3. Produce a new sample $$\hat{\textbf{x}} = \{x_1,x_2,...,-x_k,...,x_n \}$$ by flipping $$x_k$$.
4. Compute the acceptance probability of the new sample $$ \hat {\textbf{x}} $$ as $$A(\hat{\textbf{x}} \vert \textbf{x})$$ = min $$\{ 1, \dfrac{ \pi (\hat{\textbf{x}})  g(\textbf{x} \vert \hat{\textbf{x}} )}{ \pi (\textbf{x}) g(\hat{\textbf{x}} \vert \textbf{x} ) } \} $$. 

Here, $$\pi (\textbf{x}) = P(\textbf{x})$$. Similarly, $$\pi (\hat{\textbf{x}}) = P(\hat{\textbf{x}})$$ where the only difference is thay $$x_k$$ has been flipped to $$-x_k$$ in $$\hat{\textbf{x}}$$. The paritition function $$Z$$ is unchanged and cancels out in ratio $$\dfrac{\pi(\hat{\textbf{x}})}{\pi(\textbf{x})}$$. The function $$g(\textbf{x} \prime \vert \textbf{x} )$$ is the probability of the move $$\textbf{x} \rightarrow \hat{\textbf{x}}$$, it is assumed to be the same as $$g(\hat{\textbf{x}} \vert \textbf{x} )$$; i.e, the proposal distribution $$g(. \vert .)$$ is symmetric. So $$A(\hat{\textbf{x}} \vert \textbf{x})$$ is simply, $$A(\hat{\textbf{x}} \vert \textbf{x})$$ = min $$\{ 1, \dfrac{ \pi (\hat{\textbf{x}})}{ \pi (\textbf{x})} \} $$.

 Clearly when $$\pi (\hat{\textbf{x}}) \geqslant \pi (\textbf{x})$$, the ratio $$\dfrac{ \pi (\hat{\textbf{x}})}{ \pi (\textbf{x})} \geqslant 1 $$, so that $$A(\hat{\textbf{x}} \vert \textbf{x}) = 1 $$ and the state will transition to $$\hat{\textbf{x}}$$ with certainty. If $$\pi (\hat{\textbf{x}}) < \pi (\textbf{x})$$, accept the new sample $$\hat{\textbf{x}}$$ with some probability. That is, generate a random number $$u \in Uniform(0,1)$$, accept $$\hat{\textbf{x}}$$ if $$ u < \dfrac{ \pi (\hat{\textbf{x}})}{ \pi (\textbf{x})} $$, otherwise keep $$\textbf{x}$$.


## Sampling by Gibb's method
Repeat the following steps:
1. Start with some configuration $$\textbf{x} = \{x_1,x_2,...x_n\}$$.
2. Repeat:
- $$\hat{x}_1 \leftarrow $$ sample from $$P(x_1 \vert x_2,x_3,...,X_n)$$. Replace $$x_1$$ with $$\hat{x}_1$$ in $$\textbf{x}$$.
- $$\hat{x}_2 \leftarrow $$ sample from $$P(x_2 \vert \hat{x_1},x_3,...,x_n)$$. Replace $$x_2$$ with $$\hat{x}_2$$ in $$\textbf{x}$$.
- $$\vdots$$
- $$\hat{x}_n \leftarrow $$ sample from $$P(x_n \vert \hat{x_1},\hat{x_2},...,\hat{x_{n-1}})$$. Replace $$x_n$$ with $$\hat{x}_n$$ in $$\textbf{x}$$.

Now how to compute $$P(x_i \vert x_1,x_2...,x_n)$$ ?. We know that $$P(\textbf{x}) = \dfrac{1}{Z}  \exp \{ -\alpha \sum_{\{i\}\in C_1} x_i  +  \beta \sum_{\{i,j\}\in C_2} x_i x_j \}$$. Consider pixel $$x_i$$ (random variable for site $$i$$) conditioned on all other pixels $$x_1,x_2,...,x_{i-1},x_{i+1},...,x_n$$, i.e,

$$P(x_i \vert x_1,...,x_{i-1},x_{i+1},...,x_n) = \dfrac{P(x_1,...,x_{i-1},x_i,x_{i+1},...,x_n)}{\sum_{x_i \in L} P(x_1,...,x_{i-1},x_i,x_{i+1},...,x_n)}$$.

But we can split $$P(x_1,...,x_{i-1},x_i,x_{i+1},...,x_n}$$ into two groups - one that depends on $$x_i$$ and the other that does not. The group that depends on $$x_i$$ would be $$ \exp \{ -\alpha  x_i + \beta \sum_{x_j\in N(x_i)} x_i x_j \} $$. And the other goup would be $$ \exp \{ -\alpha \sum_{ \substack{ \{k\} \in C_1 \\ k \ne i }} x_k + \beta \sum_{ \substack{ \{i,j\} \in C_2 \\ k\ne i,k\ne j}} x_k x_j \} $$ 
