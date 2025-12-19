# Maximum Independent Set
Given an undirected graph $G=(V,E)$, with $n$ vertices ($n=|V|$ and $m$ edges $m=|E|$, an *independent set* $S$ is a subset of vertices ($S\subseteq V$) such that no two vertices $i,j \in S$ are neighbors in $G$. The *Maximum Independent Set problem* is the problem of finding the largest of such sets. 

Introducing for each vertex $i\in V$ a binary decision variables $x_i \in \{0,1\}$, taking value $1$ if $i\in S$ and $0$ otherwise the problem can be written as a linear program 
$$
\max_{x\in \\{0,1\\}^n}  \sum_{i\in V} x_i  \quad \text{s.t.} \quad  x_i + x_j \leq 1 \quad \forall (i,j) \in E
$$

A stronger form of the linear program can be obtained by introducing the set $\mathcal{C}$ of *maximal cliques* (cliques that cannot be extended by adding a vertex)
$$
\max_{x\in \\{0,1\\}^n}  \sum_{i\in V} x_i  \quad \text{s.t.} \quad  \sum_{i\in c} x_i \leq 1 \quad \forall c \in \mathcal{C}
$$

![Simple MIS and solution](../../assets/mis_solution_house_example.svg)

::: bpropy.co.mis
