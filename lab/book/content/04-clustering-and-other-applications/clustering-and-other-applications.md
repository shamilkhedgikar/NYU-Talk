# Clustering and Other Applications

In Lab 1, the main object was the symmetric contextual affinity matrix

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top}),
$$

where $B$ is the cross-layer incidence matrix mapping base units into inherited supports. In Lab 2, that graph was used to define contextual autocorrelation through Laplacian energy and Geary-style statistics. The next step is to ask a more structural question: once RelWeights has induced a graph on the base layer, what is the right operator for diffusion, for clustering, and for spectral geometry?

There is a real problem motivating this move. If the inherited support layer is much broader than the base analysis layer, then RelWeights can connect almost everything to everything else. Relative to a first-order geographic baseline such as queen contiguity, the projection may add many extra nonlocal edges simply because two base units happen to co-occur inside a large support. Once that happens, the graph begins to lose locality and contrast, and clustering becomes harder rather than easier: the spectrum starts to reflect a blurred, over-connected system instead of sharp support-induced regimes.

So the graph-theoretic question is not just whether RelWeights adds linkages beyond queen weights, but how those additional linkages should be treated.

*Should they all count equally? Should they be normalized, downweighted, or otherwise corrected so that the graph remains useful for clustering?*

The answer depends on the modeling goal, and in this module we focus on operator choice. Raw affinities, row-normalized operators, symmetrically normalized operators, and symmetrized random-walk operators all preserve the same support pattern but encode different notions of movement, similarity, and smoothing [@chung1997spectral; @merris1994laplacian; @fiedler1973].

## 1. Conceptual setup

Let $R \in \mathbb{R}^{n \times n}$ be a symmetric relational affinity matrix with

$$
R_{ij} \ge 0,
\qquad
R_{ij} = R_{ji},
\qquad
R_{ii} = 0.
$$

Define the degree matrix

$$
D = \operatorname{diag}(d_1,\dots,d_n),
\qquad
d_i = \sum_j R_{ij}.
$$

The degree $d_i$ measures the total inherited support mass attached to node $i$. Once $D$ is available, three closely related operators become natural.

### Random-walk normalization

The row-normalized operator is

$$
P = D^{-1}R.
$$

Its entries are

$$
P_{ij} = \frac{R_{ij}}{d_i}.
$$

For every non-island node,

$$
\sum_j P_{ij}
=
\sum_j \frac{R_{ij}}{d_i}
=
\frac{1}{d_i}\sum_j R_{ij}
= 1.
$$

So $P$ is row-stochastic: it defines a random walk on the RelWeights graph. The $i$th row is a probability transition vector telling us how mass, information, or influence leaves node $i$ and diffuses across inherited supports.

````{note}
:class: dropdown
Working example carried forward from *Defining RelWeights*

For the four-unit example,

$$
R =
\begin{bmatrix}
0 & 2 & 1 & 0 \\
2 & 0 & 2 & 1 \\
1 & 2 & 0 & 2 \\
0 & 1 & 2 & 0
\end{bmatrix},
\qquad
D =
\begin{bmatrix}
3 & 0 & 0 & 0 \\
0 & 5 & 0 & 0 \\
0 & 0 & 5 & 0 \\
0 & 0 & 0 & 3
\end{bmatrix}.
$$

The random-walk operator is therefore

$$
P = D^{-1}R =
\begin{bmatrix}
0 & 2/3 & 1/3 & 0 \\
2/5 & 0 & 2/5 & 1/5 \\
1/5 & 2/5 & 0 & 2/5 \\
0 & 1/3 & 2/3 & 0
\end{bmatrix}.
$$

Each row now sums to one, so the operator can be read as movement probabilities across the contextual graph.
````

### Symmetric normalization

The symmetric normalized similarity operator is

$$
S = D^{-1/2} R D^{-1/2}.
$$

Since $R$ is symmetric and $D^{-1/2}$ is diagonal,

$$
S^{\top}
=
\left(D^{-1/2} R D^{-1/2}\right)^{\top}
=
D^{-1/2} R D^{-1/2}
=
S.
$$

Its entries are

$$
S_{ij} = \frac{R_{ij}}{\sqrt{d_i d_j}}.
$$

This rescales each edge by both endpoints, which reduces hub domination and makes the operator better suited for spectral analysis, embedding, and graph partitioning.

````{note}
:class: dropdown
Working example carried forward from *Defining RelWeights*

Using the same $R$ and $D$,

$$
D^{-1/2}
=
\begin{bmatrix}
1/\sqrt{3} & 0 & 0 & 0 \\
0 & 1/\sqrt{5} & 0 & 0 \\
0 & 0 & 1/\sqrt{5} & 0 \\
0 & 0 & 0 & 1/\sqrt{3}
\end{bmatrix}.
$$

Hence

$$
S = D^{-1/2} R D^{-1/2}
=
\begin{bmatrix}
0 & 2/\sqrt{15} & 1/\sqrt{15} & 0 \\
2/\sqrt{15} & 0 & 2/5 & 1/\sqrt{15} \\
1/\sqrt{15} & 2/5 & 0 & 2/\sqrt{15} \\
0 & 1/\sqrt{15} & 2/\sqrt{15} & 0
\end{bmatrix}.
$$

Relative to the raw $R$, ties touching high-degree nodes are now degree-corrected rather than taken at face value.
````

### Symmetric summing after row normalization

A third operator restores symmetry after directional normalization:

$$
S_{\mathrm{ss}}
=
\frac{1}{2}\left(D^{-1}R + R D^{-1}\right)
=
\frac{1}{2}\left(P + P^{\top}\right).
$$

This keeps the random-walk normalization idea in view, but averages the forward and backward directions so that reciprocity is restored. Unlike $P$, it is symmetric. Unlike $S$, it is built directly from the row-normalized operator.

````{note}
:class: dropdown
Working example carried forward from *Defining RelWeights*

For the toy graph,

$$
S_{\mathrm{ss}}
=
\frac{1}{2}
\left(
\begin{bmatrix}
0 & 2/3 & 1/3 & 0 \\
2/5 & 0 & 2/5 & 1/5 \\
1/5 & 2/5 & 0 & 2/5 \\
0 & 1/3 & 2/3 & 0
\end{bmatrix}
+
\begin{bmatrix}
0 & 2/5 & 1/5 & 0 \\
2/3 & 0 & 2/5 & 1/3 \\
1/3 & 2/5 & 0 & 2/3 \\
0 & 1/5 & 2/5 & 0
\end{bmatrix}
\right).
$$

So

$$
S_{\mathrm{ss}}
=
\begin{bmatrix}
0 & 8/15 & 4/15 & 0 \\
8/15 & 0 & 2/5 & 4/15 \\
4/15 & 2/5 & 0 & 8/15 \\
0 & 4/15 & 8/15 & 0
\end{bmatrix}.
$$

This restores symmetry after row-normalization while retaining a direct connection to random-walk scaling.
````

### Normalized Laplacians

Two normalized Laplacians follow immediately:

$$
L_{\mathrm{rw}} = I - D^{-1}R = I - P
$$

and

$$
L_{\mathrm{sym}} = I - D^{-1/2} R D^{-1/2} = I - S.
$$

These operators correspond to two different readings of the same RelWeights graph:

- $L_{\mathrm{rw}}$ is the diffusion or random-walk Laplacian
- $L_{\mathrm{sym}}$ is the symmetric spectral-geometric Laplacian

They are closely related:

$$
L_{\mathrm{sym}} = D^{1/2} L_{\mathrm{rw}} D^{-1/2},
$$

so they encode the same connectivity structure but in different coordinate systems.

````{note}
:class: dropdown
Working example carried forward from *Defining RelWeights*

For the same four-unit graph,

$$
L_{\mathrm{rw}} = I - P
=
\begin{bmatrix}
1 & -2/3 & -1/3 & 0 \\
-2/5 & 1 & -2/5 & -1/5 \\
-1/5 & -2/5 & 1 & -2/5 \\
0 & -1/3 & -2/3 & 1
\end{bmatrix},
$$

while

$$
L_{\mathrm{sym}} = I - S.
$$

Both are normalized analogues of the raw Laplacian

$$
L_R = D - R,
$$

but they emphasize different structures: $L_{\mathrm{rw}}$ is aligned with diffusion, whereas $L_{\mathrm{sym}}$ is aligned with spectral geometry and graph cuts.
````

## 2. When projection becomes too dense

One of the main practical risks with RelWeights appears when the inherited support layer is much coarser, broader, or more overlapping than the analysis layer. In that case the projection

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

can become nearly dense. Relative to a queen graph, this means that RelWeights is no longer just adding a few meaningful nonlocal ties; it may be adding a huge number of weak and potentially unhelpful extra linkages.

The essential pathology is simple:

- overlay-based relation is too easy to create
- large support units induce many weak co-memberships
- long-range ties proliferate even when they are not especially informative

Mathematically, the projection entry is

$$
(B B^{\top})_{ij}
=
\sum_k B_{ik} B_{jk}.
$$

So $R_{ij}$ is positive whenever base units $i$ and $j$ co-participate in at least one inherited support. If the support units are broad, then many pairs share at least one support, and the graph begins to resemble a blurred complete graph rather than a sparse neighborhood system. For clustering, that is exactly the wrong direction: instead of exposing regime structure, the spectrum can wash it out.

## 3. Operator comparison

The practical question is therefore not whether to use an operator, but which operator is appropriate for which downstream goal.

| Feature | Raw \(R\) | Random-walk \(P = D^{-1}R\) | Symmetric normalization \(S = D^{-1/2} R D^{-1/2}\) | Symmetric sum \(S_{\mathrm{ss}}\) | Normalized Laplacians |
| --- | --- | --- | --- | --- | --- |
| symmetry | yes | no | yes | yes | \(L_{\mathrm{sym}}\): yes, \(L_{\mathrm{rw}}\): generally no |
| row stochasticity | no | yes | no | no | no |
| preserves zero pattern | yes | yes | yes | yes | yes |
| diffusion interpretation | weak | strong | moderate | moderate | \(L_{\mathrm{rw}}\): strong |
| spectral clustering readiness | moderate | weak | strong | moderate | \(L_{\mathrm{sym}}\): strong |
| sensitivity to hubs | high | reduced | reduced strongly | reduced moderately | depends on normalization |
| key diagnostic | degree distribution, density | row sums, transition structure, repeated multiplication | eigenvalues, eigengaps, eigenvectors | symmetry after normalization | spectrum, multiplicity of zero, spectral gap |
| best use | raw overlap intensity | spatial lag, random walks, propagation | graph cuts, embeddings, regime detection | reciprocity-preserving compromise | diffusion vs spectral geometry |
| limitation | may become too dense | can still diffuse too globally | may still inherit a dense support pattern | compromise operator, less canonical | interpretation depends on operator choice upstream |

Two facts are especially important.

First, if the degree correction does not change the support pattern, then all of these operators preserve the same connected-component structure. What changes is not whether nodes are reachable, but how strongly different parts of the graph are coupled.

Second, the spectral gap can change substantially across operators even when the zero pattern stays fixed. That is why the clustering behavior of a RelWeights graph is not a property of $R$ alone; it is also a property of the normalization.

## 4. Identifying clusters in RelWeights graphs

The reason to care about these operators is not purely algebraic. Once the graph has been normalized in a sensible way, the next question is whether the inherited support structure has created coherent regimes on the base layer. This is where the clustering spectrum enters.

### The clustering spectrum

For a symmetric normalized Laplacian such as

$$
L_{\mathrm{sym}} = I - D^{-1/2} R D^{-1/2},
$$

the ordered eigenvalues

$$
0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_n \le 2
$$

measure how costly it is for a signal to vary across the graph. Small nonzero eigenvalues correspond to low-energy modes. If several such eigenvalues cluster near zero, that is often a sign that the graph contains several relatively coherent blocks or weakly connected regimes. In a RelWeights context, those regimes are not necessarily geographic neighborhoods; they are support-induced groups.

### Fiedler vectors

The first nontrivial eigenvector, associated with $\lambda_2$, is the Fiedler vector [@fiedler1973]. It is the smoothest non-constant direction on the graph and therefore the first place to look for a weak cut. In practice:

- the sign pattern of the Fiedler vector suggests a two-way partition
- large positive and negative values indicate the two sides of the cut
- values near zero often identify bridge units or transition zones

So if RelWeights has created contextual regimes, the Fiedler vector is the first spectral object likely to reveal them.

### Recursive bi-partitioning

One strategy is to use the Fiedler vector recursively. Split the graph once using the sign or threshold of the Fiedler vector, then repeat the procedure within each resulting subgraph. This is the logic of recursive spectral bipartitioning [@hagen1992]. For RelWeights, this is attractive when the goal is an interpretable hierarchy of support-induced regimes rather than a single flat clustering.

### Clustering with multiple eigenvectors

A second strategy is to use several low-order eigenvectors jointly. Instead of clustering from $u_2$ alone, one embeds each node into the low-dimensional spectral space spanned by

$$
u_2, u_3, \dots, u_k
$$

and then clusters in that space. This is the normalized-cuts logic developed by Shi and Malik [@shi2000]. In RelWeights terms, it is useful when the support structure is not well described by a single split but instead contains several interacting modes of contextual organization.

### Spectral clustering in GeoDa

In practice, software such as GeoDa implements spectral clustering by first constructing an adjacency or similarity graph and then applying one of these low-eigenvector strategies. The results can be highly sensitive to the parameters used to define that graph. For example, when the adjacency is based on $k$-nearest neighbors, the choice of $k$ is critical. A common heuristic in the literature is to choose $k$ on the order of $\log(n)$ for large $n$ [@vonluxburg2007].

Similarly, when a Gaussian kernel is used, the bandwidth parameter $\sigma$ strongly affects the resulting graph. One suggestion is to set $\sigma$ using the mean distance to the $k$ nearest neighbors, or to use a rule of the order $\log(n)+1$ [@vonluxburg2007]. In practice, these values are not fixed truths. They are starting points.

So the lesson for RelWeights-based clustering is the same as in other spectral workflows: parameter settings should be treated as substantive choices, and careful sensitivity analysis is essential. For a contextual graph, the way support overlap is normalized or sparsified may matter as much as the clustering algorithm itself.

## 5. Variable or embedding transfer while preserving sums

Clustering is one downstream use of RelWeights. Another is transfer: taking information defined on one support layer and moving it to another while respecting the relational graph.

Suppose an embedding or multivariate representation is observed on a source support $\beta$ and we want to transfer it onto a target support $\alpha$. A first-pass overlap transfer is

$$
E_{\alpha}^{(0)} = \bar{B} E_{\beta},
$$

where $\bar{B}$ is a row-normalized overlap operator from source support $\beta$ to target support $\alpha$.

This first-pass transfer may be noisy or structurally inconsistent. A natural RelWeights regularization is therefore

$$
\widehat{E}_{\alpha}
=
\arg\min_X
\left\{
\|X - E_{\alpha}^{(0)}\|_F^2
+
\lambda \operatorname{tr}(X^{\top} L_R X)
\right\}.
$$

The two terms play different roles:

- $\|X - E_{\alpha}^{(0)}\|_F^2$ says: stay close to the overlap-based transfer
- $\lambda \operatorname{tr}(X^{\top} L_R X)$ says: smooth the embedding across the RelWeights graph

So the optimization says: among all possible target embeddings, choose the one that stays near the transferred embedding while also respecting the support-induced relational geometry.

### Closed-form solution

Because the objective is quadratic, the minimizer is explicit. Differentiating with respect to $X$ gives

$$
2(X - E_{\alpha}^{(0)}) + 2 \lambda L_R X = 0,
$$

so

$$
(I + \lambda L_R)X = E_{\alpha}^{(0)}.
$$

Hence

$$
\widehat{E}_{\alpha}
=
(I + \lambda L_R)^{-1} E_{\alpha}^{(0)}.
$$

This is not an arbitrary smoothing rule. It is the exact minimizer of the fidelity-smoothness tradeoff.

### Columnwise interpretation

If $X$ has columns $x_1,\dots,x_d$, then the optimization decomposes into

$$
\widehat{x}_j
=
\arg\min_x
\left\{
\|x - e_j^{(0)}\|_2^2 + \lambda x^{\top} L_R x
\right\},
$$

one problem for each embedding coordinate $j$. Each dimension is smoothed over the same RelWeights graph. The geometry acts on the support layer, not on the embedding dimension.

### Why this is graph denoising

Using the Laplacian identity,

$$
\operatorname{tr}(X^{\top} L_R X)
=
\frac{1}{2}
\sum_{i,k} R_{ik}\|X_{i\cdot} - X_{k\cdot}\|_2^2.
$$

So the optimization can be read as

$$
\widehat{E}_{\alpha}
=
\arg\min_X
\left\{
\sum_i \|X_{i\cdot} - E_{\alpha,i\cdot}^{(0)}\|_2^2
+
\frac{\lambda}{2}\sum_{i,k} R_{ik}\|X_{i\cdot} - X_{k\cdot}\|_2^2
\right\}.
$$

This is vector-valued denoising on the RelWeights graph:

- do not move any target embedding too far from its baseline transferred value
- but if two target units share strong inherited structure, their embedding vectors should be similar

### What is conserved

The conservation issue is subtle. The smoother does **not** automatically preserve the totals of the original source embedding $E_{\beta}$. What it preserves are the column sums of the baseline transferred embedding $E_{\alpha}^{(0)}$.

The reason is that graph Laplacians satisfy

$$
L_R \mathbf{1} = 0.
$$

Therefore

$$
(I + \lambda L_R)\mathbf{1} = \mathbf{1},
\qquad
(I + \lambda L_R)^{-1}\mathbf{1} = \mathbf{1}.
$$

Since $L_R$ is symmetric,

$$
\mathbf{1}^{\top}(I + \lambda L_R)^{-1} = \mathbf{1}^{\top}.
$$

Now take any column $e^{(0)}$ of $E_{\alpha}^{(0)}$. Its smoothed version is

$$
\widehat{x} = (I + \lambda L_R)^{-1} e^{(0)}.
$$

Then

$$
\mathbf{1}^{\top}\widehat{x}
=
\mathbf{1}^{\top}(I + \lambda L_R)^{-1} e^{(0)}
=
\mathbf{1}^{\top}e^{(0)}.
$$

So each column sum, and therefore each column mean, is preserved on the target layer.

The distinction is important:

- transfer decides what totals or means arrive on the target layer
- smoothing redistributes within the target layer while preserving those target-layer sums

If exact source-target consistency is required, then a constrained optimization must be added explicitly.

### Working example

Use the same toy transfer setup as before. Let

$$
\bar{B}
=
\begin{bmatrix}
1/2 & 1/3 & 0 & 0 \\
1/2 & 1/3 & 1/3 & 0 \\
0 & 1/3 & 1/3 & 1/2 \\
0 & 0 & 1/3 & 1/2
\end{bmatrix}
$$

and let the source embedding be

$$
E_{\beta}
=
\begin{bmatrix}
1 & 0 \\
2 & 1 \\
5 & 4 \\
6 & 5
\end{bmatrix}.
$$

The overlap-based transfer is

$$
E_{\alpha}^{(0)}
=
\bar{B}E_{\beta}
=
\begin{bmatrix}
1.5 & 0.5 \\
2.6667 & 1.6667 \\
4.3333 & 3.3333 \\
5.5 & 4.5
\end{bmatrix}.
$$

The column sums are

$$
1.5 + 2.6667 + 4.3333 + 5.5 = 14,
$$

and

$$
0.5 + 1.6667 + 3.3333 + 4.5 = 10.
$$

Now use the same RelWeights Laplacian from the four-unit example,

$$
L_R =
\begin{bmatrix}
3 & -2 & -1 & 0 \\
-2 & 5 & -2 & -1 \\
-1 & -2 & 5 & -2 \\
0 & -1 & -2 & 3
\end{bmatrix},
$$

with $\lambda = 0.2$. Then

$$
A := I + \lambda L_R
=
\begin{bmatrix}
1.6 & -0.4 & -0.2 & 0 \\
-0.4 & 2.0 & -0.4 & -0.2 \\
-0.2 & -0.4 & 2.0 & -0.4 \\
0 & -0.2 & -0.4 & 1.6
\end{bmatrix}.
$$

The smoothed embedding is

$$
\widehat{E}_{\alpha} = A^{-1}E_{\alpha}^{(0)}
\approx
\begin{bmatrix}
2.1786 & 1.1786 \\
2.7857 & 1.7857 \\
4.2143 & 3.2143 \\
4.8214 & 3.8214
\end{bmatrix}.
$$

The new values are redistributed, but the column sums are preserved:

$$
2.1786 + 2.7857 + 4.2143 + 4.8214 = 14,
$$

and

$$
1.1786 + 1.7857 + 3.2143 + 3.8214 = 10.
$$

So the smoother acts like a conservative diffusion or balancing operator on the target layer:

- high values are pulled down
- low values are pulled up
- total target-layer column mass stays fixed

## What This Module Sets Up

This chapter establishes the main normalization choices available once RelWeights has generated a contextual graph:

- raw affinity for direct overlap intensity
- row normalization for diffusion and lag interpretation
- symmetric normalization for spectral clustering and graph geometry
- symmetric summing for a reciprocity-preserving compromise
- spectral clustering strategies based on one or several low-order eigenvectors
- graph-based embedding transfer that preserves target-layer column sums

That is the theoretical basis for the next stage of work: using normalized RelWeights operators to detect contextual regimes, compare alternative clusterings, and carry relational structure into interpolation and spatial econometric applications.
