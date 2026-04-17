---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Defining RelWeights

RelWeights is easiest to understand as an extension of the spatial weights tradition. Before constructing a relational operator from overlays, it helps to understand more formally what a spatial weights matrix is, what assumptions it carries, and why it is useful.

## Spatial Weights Matrix

Let $S = \{1, \ldots, n\}$ denote a set of spatial units. A spatial weights matrix is an $n \times n$ matrix

$$
W = [w_{jk}]
$$

whose entries encode the relative spatial influence of unit $k$ on unit $j$. In the most common formulation, the weights satisfy $w_{jk} \ge 0$, and many applications row-standardize the matrix so that

$$
\sum_{k=1}^{m} w_{jk} = 1
$$

for each row $j$. Row-standardization is useful because it makes different weighting schemes more directly comparable, but it is a convention rather than a universal requirement. In practice, spatial weights may be binary, row-standardized, distance-decayed, kernel-smoothed, or left unstandardized until a later modeling stage.

$W$ captures some structural assumptions about how spatial dependence propagates. Choosing $W$ means choosing what counts as a neighbor, how strongly neighbors matter, and whether influence is local, directional, group-based, or layered across multiple supports.

## Properties and Types

Several implications follow from this broader definition:

- weights need not be symmetric, since influence from $j$ to $k$ may differ from influence from $k$ to $j$
- weights need not be restricted to contiguous units, since relevant spillovers may travel through distance, networks, institutions, or shared membership
- diagonal entries may be set to zero when self-influence is excluded, but they may also be non-zero when the analyst wants to represent inertia, persistence, or within-unit autonomy
- **most useful spatial weights are sparse, since only a small fraction of all possible unit pairs are typically treated as connected**

For this lab, that last point matters a great deal: RelWeights does not abandon the operator logic of $W$. It changes the rule by which neighborhoods are induced, moving from simple geometric adjacency to inherited relational support through overlays.

### Contiguity

Contiguity weights are the classical starting point for polygon data. Rook contiguity links units that share an edge; queen contiguity links units that share either an edge or a vertex; bishop contiguity isolates the corner-sharing case and is less common in applied econometrics but useful for clarifying how permissive different neighbor rules are. The substantive consequence is immediate: queen weights are denser than rook weights, and the choice changes the graph over which lags, diagnostics, and spillovers are computed.

```{figure} ../../assets/images/weights-1.jpg
:width: 100%
:name: fig-contiguity-weights

Common contiguity structures for areal data: rook, bishop, and queen criteria induce different adjacency graphs and therefore different spatial weights.

```

The broader geographic data science literature emphasizes several practical properties of these matrices.

- First, the same spatial configuration can be represented either as a sparse neighbor list or as a dense matrix, but sparse storage is usually preferable because most entries are zero.
- Second, density matters: some weighting rules create few links while others create many, and that affects both interpretation and computation.
- Third, disconnected observations or "islands" deserve attention because many downstream statistics assume every unit is connected to at least part of the graph.

### Data Types

Beyond contiguity, several other families of weights can be defined based on variation of underlying spatial processes and the type of spatial data:

```{figure} ../../assets/images/weights-2.jpg
:width: 100%
:name: fig-distance-block-kernel-weights

Beyond contiguity, spatial weights can also be induced by distance-decay, shared group membership, and continuous kernel functions.

```

- distance-based weights define neighbors as a function of separation in space rather than shared boundaries
- $k$-nearest-neighbor weights force each unit to have at least $k$ neighbors and are often useful when contiguity would leave islands
- distance-band weights connect all units within a threshold distance, either with binary links or truncated continuous decay; *threshold of coverage*
- kernel weights impose smooth distance decay, making nearby units more influential than distant ones within a chosen bandwidth; *processes of friction or decay*
- block weights connect units that share membership in the same category, such as school districts, counties, aquifers, governance zones, or policy regimes; *processes dependent on institutional membership*
- hybrid weights combine multiple rules, such as queen contiguity within counties or distance-based links constrained by institutional boundaries.

See [@bavaud1998models] for a more advanced treatment of constructing spatial weights.

```{figure} ../../assets/images/weights-3.jpg
:width: 100%
:name: fig-datatype-weights

Spatial weights choice also changes based on the type of data being analyzed. 
```

### Graph Objects

**Spatial weights matrices can be interpreted as graph operators!**

Each observation is a node, and each non-zero weight is an edge or tie in the graph. The matrix is always indexed by the observational units, but the visual ordering of rows and columns is arbitrary. What matters is the mapping between units and indices, not the particular row order printed on the page. Because most pairs are not neighbors, weights are usually stored sparsely by recording only the non-zero links.

**Even when a weights matrix appears to be a simple technical choice, it remains a substantive modeling decision.**

We start from that broader view of spatial weights and then ask how to define a more flexible operator from overlays. We frame this problem as one of **converting layered spatial intersections into a reusable operator**, so the same operator can support diagnostics, smoothing, decomposition(s), and policy interpretation. In that framing, the geometry that matters is inherited support in $\beta$ in addition to the shared boundaries in $\alpha$.

## RelWeights

Let $\alpha = \{\alpha_1, \ldots, \alpha_n\}$ denote the analysis layer and let $\beta = \{\beta_1, \ldots, \beta_m\}$ denote the contextual or inherited support layer. Typical examples are:

- districts over hydrological basins
- tracts over school catchments
- neighborhoods over policy zones

The broader associativeness we seek from RelWeights is that several distinct contextual layers can be inherited onto the same analysis layer without first forcing them into a common geography. In that sense, overlays become a way of accumulating relational support: different supports can all contribute structure on $\alpha$, even when they do not share the same boundaries or scale.

The key design choice is that $\beta$ defines the support pattern that will later induce similarity among the $\alpha$ units.

```{figure} ../../assets/images/config-1.png
:width: 40%
:name: fig-alpha-beta-config

$\alpha$ (Red) and $\beta$ (Blue) represent two different units of analysis (extendable to *n* layers). We are interested in transferring the similarities.  
```

### Building the incidence matrix

We can think of two ways to capture the intersection relationship:

1.Binary support:

$$
B_{\mathrm{bin}}(i,k) =
\begin{cases}
1 & \text{if } |\alpha_i \cap \beta_k| > 0 \\
0 & \text{otherwise}
\end{cases}
$$

2.Area-overlap support:

$$
B_{\mathrm{area}}(i,k) = \frac{|\alpha_i \cap \beta_k|}{|\alpha_i|}
$$

For the area-based construction, each row is normalized over the support carried by $\alpha_i$:

$$
\sum_k B_{\mathrm{area}}(i,k) = 1
$$

The binary matrix is just the support pattern of the area-overlap matrix:

$$
B_{\mathrm{bin}}(i,k) = \mathbf{1}\!\left(B_{\mathrm{area}}(i,k) > 0\right)
$$

:::{note} Note: Notation
:class: dropdown

Here $i$ indexes a unit in the analysis layer $\alpha$, and $k$ indexes a unit in the inherited support layer $\beta$.

- $B_{\mathrm{area}}(i,k)$ is the share of $\alpha_i$ covered by $\beta_k$
- $\mathbf{1}(\cdot)$ is the indicator function: it equals **1** when the condition inside is true and **0** otherwise
- We use (i,k) indices to differentiate our cross-layer incidence matrix from $\mathrm{W_{ij}}$

So essentially:

"Set the binary incidence entry to `1` whenever the area-overlap entry is positive."

In other words, $B_{\mathrm{bin}}$ forgets *how much* overlap there is and keeps only whether any overlap exists at all. It is therefore the support pattern of $B_{\mathrm{area}}$: a yes/no version of the same overlay relationship.
:::

:::{important} Note: Why the incidence matrix matters
:class: dropdown

The dimensionality of the incidence matrix is part of the construction. If $\alpha$ has $n$ units and $\beta$ has $m$ units, then $B \in \mathbb{R}^{n \times m}$ is a cross-layer object: its rows index units in $\alpha$ and its columns index units in $\beta$. That is exactly what lets it record which $\alpha_i$ inherits support from which $\beta_k$.

By contrast, adjacency matrices live within layers. $A_{\alpha} \in \mathbb{R}^{n \times n}$ only tells us which $\alpha$ units touch other $\alpha$ units, and $A_{\beta} \in \mathbb{R}^{m \times m}$ only tells us which $\beta$ units touch other $\beta$ units. Those square graphs do not contain the overlay information needed to build a cross-layer similarity on $\alpha$. That missing information lives in $B$, which is why RelWeights is built from $B B^{\top}$ rather than from $A_{\alpha}$ and $A_{\beta}$ alone.
:::

### Form the RelWeights kernel

The relational similarity matrix is built from shared support:

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

Entrywise,

$$
R_{ij} = \sum_k B_{ik} B_{jk}
$$

before diagonal removal. In words, $R_{ij}$ measures how much contextual support units $\alpha_i$ and $\alpha_j$ inherit in common through $\beta$.

There are some very interesting properties 

- $R_{ij} \ge 0$
- $R$ is symmetric
- $\operatorname{rank}(R) \le \operatorname{rank}(B)$
- $B B^{\top}$ is positive semidefinite before the diagonal is removed

## The Laplacian Operator

Define the relational degree matrix from the row sums of $R$:

$$
(D_R)_{ii} = \sum_j R_{ij}
$$

Then define the relational Laplacian:

$$
L_R = D_R - R
$$

Entrywise,

$$
(L_R)_{ij} = \delta_{ij} \sum_k R_{ik} - R_{ij}
$$

This is the operator form that the rest of the lab will reuse.

## Quadratic Form of the Laplacian

For any signal $x \in \mathbb{R}^n$, the core identity is

$$
x^{\top} L_R x = \frac{1}{2} \sum_{i,j} R_{ij} (x_i - x_j)^2
$$

This is the most useful interpretation to keep in mind while building the lab. The quadratic form is an energy functional: it is small when values vary smoothly across units that share inherited support, and large when strongly related units diverge.

## Step 6: Use the spectrum

Because $(x_i - x_j)^2 \ge 0$ and $R_{ij} \ge 0$, the reference note shows that

$$
x^{\top} L_R x \ge 0
$$

so $L_R$ is positive semidefinite and all eigenvalues satisfy $\lambda \ge 0$. It also shows that the constant vector lies in the nullspace:

$$
L_R \mathbf{1} = 0
$$

That gives the standard spectral picture:

- the zero mode is constant over connected relational support
- small eigenvalues correspond to smooth relational variation
- larger eigenvalues capture oscillatory or high-friction variation
