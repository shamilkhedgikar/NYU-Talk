# Clustering and Other Applications

This module turns the Laplacian spectrum into a clustering object. Once RelWeights has produced a contextual graph on the base units, the next question is whether that graph contains relatively coherent regimes, blocks, or weakly connected bridges. Spectral decomposition gives one way to read that structure directly from the operator [@chung1997spectral; @merris1994laplacian].

## Spectral decomposition of the Laplacian

For any symmetric RelWeights matrix $R$, the associated Laplacian is

$$
L_R = D_R - R.
$$

Because $L_R$ is symmetric and positive semidefinite, it admits an orthogonal eigendecomposition

$$
L_R = U \Lambda U^{\top},
$$

where:

- $U = [u_1,\dots,u_n]$ collects orthonormal eigenvectors
- $\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)$ collects ordered eigenvalues with

$$
0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_n.
$$

The first eigenvector is the constant vector on each connected component. The remaining eigenvectors describe increasingly rough modes of variation over the graph. Small eigenvalues correspond to smooth, low-energy patterns under the RelWeights structure; larger eigenvalues correspond to faster oscillation across contextual ties.

For RelWeights, this matters because the supports can create contextual neighborhoods that do not align with simple first-order geographic adjacency. The spectrum therefore captures structure induced by inherited support, not just by contiguity.

## The second eigenvalue and algebraic connectivity

The second-smallest eigenvalue $\lambda_2$ plays a special role. In graph theory it is the **algebraic connectivity** of the graph [@fiedler1973]. Intuitively:

- if $\lambda_2$ is close to zero, the graph has a weak bridge, near-separation, or clearly distinguishable regimes
- if $\lambda_2$ is larger, the graph is more tightly tied together under the chosen support structure

So in a RelWeights setting, a small $\lambda_2$ is often the first spectral sign that inherited supports have partitioned the base geography into groups that interact strongly within themselves and only weakly across groups.

## Fiedler vectors

An eigenvector associated with $\lambda_2$ is called a **Fiedler vector** [@fiedler1973]. This is the first nontrivial smooth mode of the Laplacian. It is useful because its entries assign each base unit a scalar coordinate that reflects how that unit sits relative to the graph's weakest separation.

For a RelWeights graph, the Fiedler vector often behaves like a one-dimensional contextual ordering:

- units with similar Fiedler scores tend to lie in the same inherited regime
- units with opposite-signed or widely separated scores tend to sit on different sides of a weak contextual cut
- sharp transitions in the vector often indicate bridges or boundaries between support-defined blocks

This is why the Fiedler vector is so often used in graph partitioning. It is not an arbitrary embedding; it is the smoothest non-constant direction allowed by the graph.

## Using the second eigenvector for clustering

The simplest spectral clustering step is to use the sign or magnitude of the Fiedler vector entries to split the graph into groups.

For a two-way cut, a common rule is:

$$
\mathcal C_1 = \{i : u_{2,i} \le 0\}, \qquad
\mathcal C_2 = \{i : u_{2,i} > 0\},
$$

where $u_2$ is a Fiedler vector. More generally, one can sort the entries of $u_2$ and choose a threshold that produces a substantively meaningful split.

The interpretation in this lab is important:

- the clustering is occurring on the **base units**
- but the similarity structure comes from the **inherited supports**
- so the resulting groups are contextual regimes induced by RelWeights, not simply contiguous polygons

That is exactly the behavior suggested by the eigenvalue grouping you saw in Lab 1. When support structure creates repeated or near-repeated contextual patterns, the low end of the spectrum often reflects those regimes before we ever run an explicit clustering algorithm.

## Why this is useful for RelWeights

This spectral view is useful beyond graph cuts.

- It provides a low-dimensional summary of contextual structure on the base geography.
- It helps identify whether inherited supports are creating coherent regimes rather than diffuse ties everywhere.
- It supplies features that can later be used in clustering, interpolation, smoothing, or spatial econometric diagnostics.

The key conceptual move is simple: RelWeights gives the graph, the Laplacian gives the operator, and the eigenvectors give coordinates adapted to that operator. Clustering is then one downstream use of those coordinates.

## What This Module Sets Up

The next practical step is to compute the low-order eigenpairs of $L_R$ and use them for:

- graph partitioning with the Fiedler vector
- regime detection on the base geography
- comparing geographic clusters with support-induced clusters
- extending from a single split to multi-way clustering using several low-frequency eigenvectors

That is where the clustering workflow begins: not with an arbitrary distance matrix, but with a spectral basis generated by the relational support structure itself.
