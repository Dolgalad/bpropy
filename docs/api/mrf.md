# Markov Random Fields
Markov Random Fields (or Undirected Graphical Models) are families of joint distributions having the Markov property with regard to an undirected graph $G=(V,E)$. Given $\mathcal{C}$ the set of maximal cliques of $G$, members of the Markov Random Field defined over $G$ are distribution functions $p(\cdot)$ that can be factorized as the product of *potentials* over the cliques of $G$

$$ p(x) = \frac{1}{Z} \prod\limits_{c\in \mathcal{C}} \psi_c(x_c) $$
where the *potentials* $\psi_c : \mathcal{X}_c \rightarrow \mathbb{R}^+$ attribute importance to factor assignments and $Z$ is the **partition funciton** that ensures that $p(\cdot)$ is a distribution.


## Factor Graphs
A *factor graph* is a bipartite graph $H = (V\cup F, \mathcal{E})$ that makes the decomposition of $p(\cdot)$ explicit over the set of *factors* $F$. 

![Simple factor graph](../../assets/factor_graph_house.svg)
In this example the distribution factorizes as $p(x) = \frac{1}{Z} \psi_{01}(x_0,x_1) \psi_{03}(x_0,x_3) \psi_{23}(x_2,x_3) \psi_{124}(x_1,x_2,x_4)$.

## Exponential Family
The exponential family is a particular class of MRFs where the potential functions take the form 

$$\psi_c (x_c) = \exp \langle \theta_c, \phi_c(x_c) \rangle$$

with $\phi(x) = \left( \phi_c(x_c) \right)_{c\in\mathcal{C}, x_c \in \mathcal{X}_c}$ the *sufficient statistics* and serve as a "representation" of an assignment $x$. $\theta = \left( \theta_c(x_c) \right)_{c\in\mathcal{C}, x_c\in\mathcal{X}_c}$ the set of parameters. 

### Canonical Over-Complete Sufficient Statistics
The canonical over-complete representation 

$$\phi_{\alpha}(x_{\alpha}) = (\delta \left[ x_{\alpha} = y \right])_{y\in\mathcal{X}_{\alpha}}$$

When all the variables are binary the number of sufficient statistics corresponding to factor $\alpha\in F$ is $2^{|F|}$, one for each possible partial assignment of $x_{\alpha}$. 

In this package we have implemented a synchronous version of the *Belief Propagation* algorithm, a message passing scheme for approximating the factor marginal distributions. We deal in particular with marginal inference for members of the exponential family.

## GPU Friendly Belief Propagation
