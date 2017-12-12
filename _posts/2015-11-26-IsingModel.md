Please refer to my earlier posts on [MRF](https://pchanda.github.io/MarkovRandomFields/) and [GRF](https://pchanda.github.io/GibbsRandomField/) for getting used to the notations. If we consider clique potentials for sizes of upto 2, he energy function is the sum of clique potentials of all cliques of size 1 and size 2,

$$ U( \textbf{x} ) = \sum_{\{i\}\in C_1} \phi_1(x_i) + \sum_{\{i,j\}\in C_2} \phi_2(x_i,x_j) $$

Also assume that our label set $$ L $$ is binary, $$L \in \{-1,1\}$$. To further simplify things, assume that $$\phi_1(x_i)=\alpha_i x_i $$ and $$ \phi_2(x_i,x_j) = \beta_{i,j} \delta(x_i,x_j) $$. 

Here $$\beta_{i,j}$$ can be thought of as strengths of pairwise interactions, while $$\alpha_i$$ reflects the bias of $$x_i$$ towards a particular state in preference to the other. 

Further simplify, $$\alpha_i = \alpha, \forall i $$, $$\beta_{i,j} = \beta, \forall i,j $$, and $$\delta(x_i,x_j) = -x_i x_j $$. This gives and example of an [Ising Model](https://en.wikipedia.org/wiki/Ising_model). Then,

Hello3

$$\alpha$$

$$\dfrac{1}{Z}\exp \{ -\alpha \sum_{\{i\}\in C_1} x_i  + \beta \sum_{\{i,j\}\in C_2} x_i x_j  \}$$

$$P( \textbf{x} ) = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \}$$

$$U( \textbf{x} ) = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \}$$


$$
\begin{align}
U( \textbf{x} ) & = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \} \\\\
& = \dfrac{1}{Z}\exp \{ -\sum\limits_{c \in C} \phi_{c}(\textbf{x}) \} \\\\
& = \dfrac{1}{Z} \prod\limits_{c \in C} \exp \{ -\phi_{c}(\textbf{x}) \} 
\end{align}
$$


$$
\begin{align}
P( \textbf{x} ) & = \dfrac{1}{Z} \exp \{ -U(\textbf{x}) \} \\\\
& = \dfrac{1}{Z}\exp \{ -\alpha \sum_{\{i\}\in C_1} x_i  + \beta \sum_{\{i,j\}\in C_2} x_i x_j  \} \\\\
& = \dfrac{1}{Z}  \exp \{ -\alpha \sum_{\{i\}\in C_1} x_i\}  \exp\{ \beta \sum_{\{i,j\}\in C_2} x_i x_j \}   
\end{align}
$$

