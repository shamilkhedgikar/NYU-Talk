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

Thus, we the binary incidence entry to **1** whenever the area-overlap entry is positive. It is therefore the support pattern of $B_{\mathrm{area}}$: a yes/no version of the same overlay relationship leading to an adjacency matrix for the system.

:::

:::{note} The Incidence Matrix
:class: dropdown

The dimensionality of the incidence matrix is part of the construction. If $\alpha$ has $n$ units and $\beta$ has $m$ units, then $B \in \mathbb{R}^{n \times m}$ is a cross-layer object: its rows index units in $\alpha$ and its columns index units in $\beta$. That is exactly what lets it record which $\alpha_i$ inherits support from which $\beta_k$.

By contrast, adjacency matrices live within layers. $A_{\alpha} \in \mathbb{R}^{n \times n}$ only tells us which $\alpha$ units touch other $\alpha$ units, and $A_{\beta} \in \mathbb{R}^{m \times m}$ only tells us which $\beta$ units touch other $\beta$ units. Those square graphs do not contain the overlay information needed to build a cross-layer similarity on $\alpha$. That missing information lives in $B$, which is why RelWeights are built from $B B^{\top}$ rather than from $A_{\alpha}$ and $A_{\beta}$ alone.

**In fact it can be proved that the incidence matrix cannot be derived simply from the independent adjacency.**
:::

### The RelWeights kernel

The RelWeights matrix is built from shared support by using the Gram Matrix $B B^{\top}$ and is :

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

Entrywise,

$$
R_{ij} = \sum_k B_{ik} B_{jk}
$$

before diagonal removal. In words, $R_{ij}$ measures how much contextual support units $\alpha_i$ and $\alpha_j$ inherit in common through $\beta$.

$R_{ij}$ has some very nice properties

- $R_{ij} \ge 0$
- $R$ is symmetric
- $\operatorname{rank}(B B^{\top}) = \operatorname{rank}(B)$ before diagonal removal
- $B B^{\top}$ is positive semidefinite before the diagonal is removed

:::{note} Note: Why these properties hold
:class: dropdown

- Each entry of $B$ is nonnegative, so each product $B_{ik} B_{jk}$ is nonnegative as well. Summing over $k$ gives $(B B^{\top})_{ij} \ge 0$, and subtracting the diagonal only replaces $R_{ii}$ by zero, so the off-diagonal entries of $R$ remain nonnegative.

- Symmetry follows from the Gram form: $(B B^{\top})^{\top} = B B^{\top}$. Removing the diagonal preserves symmetry, so $R_{ij} = R_{ji}$.

- The rank statement is cleanest before diagonal removal. For any matrix $B$, the matrices $B$ and $B B^{\top}$ have the same nullspace on the left-hand side, so the Gram matrix $B B^{\top}$ has the same rank as $B$. Diagonal removal can change rank, which is why the equality is stated for $B B^{\top}$ rather than for $R$ itself.

- Positive semidefiniteness is immediate from the quadratic form. For any $x \in \mathbb{R}^n$,

$$
x^{\top} B B^{\top} x = (B^{\top} x)^{\top}(B^{\top} x) = \|B^{\top} x\|_2^2 \ge 0,
$$

so $B B^{\top}$ is positive semidefinite.
:::

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

Here $\delta_{ij}$ is the Kronecker delta: it equals $1$ when $i=j$ and $0$ otherwise. So the first term only contributes on the diagonal, which means $(L_R)_{ii}$ stores the relational degree of unit $\alpha_i$, while the off-diagonal entries are just $-(R_{ij})$.

This is the operator form that the rest of the lab will mostly rely on.

:::{note} Note: Comparing $B B^{\top}$ and $L_R$
:class: dropdown

Both $B B^{\top}$ and $L_R$ are symmetric positive semidefinite, but they have different interpretations.

- $B B^{\top}$ is a Gram matrix. It accumulates shared inherited support, so large entries mean that two $\alpha$ units overlap many of the same $\beta$ supports. Its diagonal records self-similarity.
- $L_R = D_R - R$ is a Laplacian. It converts similarity into a roughness operator that penalizes differences across related units. Its rows sum to zero, and the constant vector lies in its nullspace.

So $B B^{\top}$ is the similarity object, whereas $L_R$ is the smoothing or variation-penalty object built from that similarity structure.
:::

## Linear Form of the Laplacian

Applying the operator to a vector $x \in \mathbb{R}^n$ gives

$$
(L_R x)_i = \sum_j R_{ij}(x_i - x_j).
$$

This is the **linear form** of the Laplacian: a node-level expression that tells us how far unit $i$ sits above or below its contextual neighborhood under the RelWeights graph. Expanding it back into degree-minus-similarity form gives

$$
(L_R x)_i = \left(\sum_j R_{ij}\right)x_i - \sum_j R_{ij}x_j,
$$

so it compares the value at unit $i$ to the weighted values carried by the units to which it is relationally tied.

The contrast with ordinary spatial lags is useful:

| Operator | Formula | Interpretation |
| ----- | ----- | ----- |
| $Wx$ | $(Wx)_i = \sum_j W_{ij}x_j$ | Standard spatial lag. If $W$ is row-standardized, this is the neighborhood average under the chosen spatial weights matrix; otherwise it is a weighted neighborhood exposure. |
| $Rx$ | $(Rx)_i = \sum_j R_{ij}x_j$ | Contextual spatial lag induced by shared inherited support. If $R$ is normalized, it is a contextual neighborhood average; otherwise it accumulates weighted support from related units. |
| $L_Rx$ | $(L_R x)_i = \sum_j R_{ij}(x_i - x_j)$ | Contextual deviation from neighbors. This is not an average of surrounding values, but the signed local imbalance of unit $i$ relative to its contextual neighborhood. |

- $Wx$ and $Rx$ are **lag operators**: they push neighboring values onto unit $i$.
- $L_Rx$ is a **difference operator**: it measures disagreement between unit $i$ and the values around it.
- Constant vectors pass through lags but are annihilated by the Laplacian, since $L_R \mathbf{1}=0$.

So the ordinary lag asks, "what do my neighbors look like?", while the Laplacian asks, "how far am I from what my neighbors imply?"

## Quadratic Form of the Laplacian

For any vector $x \in \mathbb{R}^n$, the scalar

$$
E_R(x) := x^{\top} L_R x
$$

is the **Relational Laplacian Energy** (a physical framing) or **Roughness** of that vector across the combined supports of the base and overlay layers. It is an aggregate object - a scalar. So rather than describing one pair of units at a time, it collapses the entire pattern of weighted disagreement into a single number.

This scalar should be interpreted as a measure over the combined (base + inherited) support structure. It is small when nearby values are smooth over strong relational ties, and it becomes large when units that share substantial contextual support take very different values. In combined supports where the vector values are constant, the value vanishes.

Similar to the linear form, the aggregate can be written as a weighted sum of squared pairwise differences.

For any vector $x \in \mathbb{R}^n$, the core identity is

$$
x^{\top} L_R x = \frac{1}{2} \sum_{i,j} R_{ij} (x_i - x_j)^2
$$

The right hand side is very similar to local form of **Geary's C** (without variance normalization), a measure of global spatial autocorrelation based on squared differences across all weighted neighbor pairs. In fact $L_R$ based on our RelWeights can be interpreted as a contextual autocorrelation statistic with appropriate variance normalization. We will explore this idea in Lab 2: Simulating Structural Variations.

:::{note} Note: Proof of the quadratic identity
:class: dropdown

It helps to start from the ordinary Euclidean case: $\|x\|_2^2 = x^{\top} I x$. More generally, a symmetric positive semidefinite matrix $M$ defines a weighted quadratic size

$$
\|x\|_M^2 := x^{\top} M x.
$$

If $M$ is positive definite, this is a genuine norm. If $M$ is only positive semidefinite, it is a seminorm. Laplacians fall in the second category, since constant vectors represent zero *roughness*. Thus $x^{\top} L_R x$ should be read as a relational roughness seminorm: it measures variation in $x$ across the support encoded by $R$.

To prove the identity, expand from $L_R = D_R - R$:

$$
x^{\top} L_R x = x^{\top} D_R x - x^{\top} R x
= \sum_i \left(\sum_j R_{ij}\right)x_i^2 - \sum_{i,j} R_{ij} x_i x_j.
$$

Using symmetry of $R$, rewrite the first term symmetrically:

$$
\sum_i \left(\sum_j R_{ij}\right)x_i^2
= \frac{1}{2}\sum_{i,j} R_{ij}(x_i^2 + x_j^2).
$$

Substituting gives

$$
x^{\top} L_R x
= \frac{1}{2}\sum_{i,j} R_{ij}(x_i^2 + x_j^2 - 2x_i x_j)
= \frac{1}{2}\sum_{i,j} R_{ij}(x_i - x_j)^2.
$$

That is why the Laplacian quadratic form is a measure of roughness: it aggregates squared pairwise differences, weighted by relational similarity.
:::

Because $(x_i - x_j)^2 \ge 0$ and $R_{ij} \ge 0$,

$$
x^{\top} L_R x \ge 0
$$

so $L_R$ is positive semidefinite and all eigenvalues satisfy $\lambda \ge 0$. It also shows that a constant vector lies in the nullspace:

$$
L_R \mathbf{1} = 0
$$

That gives the standard spectral picture:

- the zero mode is constant over connected relational support
- small eigenvalues correspond to smooth relational variation
- larger eigenvalues capture oscillatory or high-friction variation

## Working Example

To make the construction concrete, us work through the example we started with:

```{figure} ../../assets/images/config-1.png
:width: 40%
:name: fig-alpha-beta-config-2

$\alpha$ (Red) and $\beta$ (Blue) represent two different units of analysis (extendable to *n* layers). We are interested in transferring the similarities.  
```

$$
\alpha = \{A, B, C, D\}
$$

$$
\beta = \{X, Y, Z, W\}.
$$

Suppose the overlay relationships are:

- $A$ intersects $X, Y$
- $B$ intersects $X, Y, Z$
- $C$ intersects $W, Y, Z$
- $D$ intersects $W, Z$

This is the simplest binary-support version of RelWeights: each district either overlaps a regime or it does not.

### Step 1: Choose the two layers

Here $\alpha$ is the layer on which we want to model dependence, and $\beta$ is the contextual layer whose support structure will be inherited onto $\alpha$. The point of the construction is that districts need not be similar because they share borders alone; they can also be similar because they intersect the same contextual regimes.

### Step 2: Build the incidence matrix

Using the notation of this note, the intersection structure is recorded in the incidence matrix $B$ rather than a conventional within-layer weights matrix. Rows correspond to districts and columns correspond to contextual regimes ordered as $(X, Y, Z, W)$:

$$
B =
\begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1
\end{bmatrix}.
$$

Entry $B_{ik}=1$ means district $\alpha_i$ intersects regime $\beta_k$. This is the cross-layer object that carries the inherited support pattern.

### Step 3: Form the RelWeights kernel

The first object we build from $B$ is the Gram matrix

$$
B B^{\top}.
$$

For this example,

$$
B B^{\top} =
\begin{bmatrix}
2 & 2 & 1 & 0 \\
2 & 3 & 2 & 1 \\
1 & 2 & 3 & 2 \\
0 & 1 & 2 & 2
\end{bmatrix}.
$$

Each entry $(B B^{\top})_{ij}$ counts how many contextual regimes districts $i$ and $j$ share. Thus $A$ and $B$ share two regimes, $A$ and $C$ share one, and $A$ and $D$ share none.

Removing the diagonal produces the RelWeights matrix:

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

so that

$$
R =
\begin{bmatrix}
0 & 2 & 1 & 0 \\
2 & 0 & 2 & 1 \\
1 & 2 & 0 & 2 \\
0 & 1 & 2 & 0
\end{bmatrix}.
$$

This is now a same-layer relational weights matrix on $\alpha$: it tells us how strongly the districts are connected after inheriting support from $\beta$.

### Step 4: Lift similarity into a Laplacian

The degree matrix comes from the row sums of $R$:

$$
D_R =
\begin{bmatrix}
3 & 0 & 0 & 0 \\
0 & 5 & 0 & 0 \\
0 & 0 & 5 & 0 \\
0 & 0 & 0 & 3
\end{bmatrix}.
$$

The relational Laplacian is therefore

$$
L_R = D_R - R
$$

with

$$
L_R =
\begin{bmatrix}
3 & -2 & -1 & 0 \\
-2 & 5 & -2 & -1 \\
-1 & -2 & 5 & -2 \\
0 & -1 & -2 & 3
\end{bmatrix}.
$$

This operator is the object that the rest of the lab will use for smoothing, decomposition, and diagnostics.

### Step 5: Read the quadratic form as roughness

For any vector $x \in \mathbb{R}^4$,

$$
x^{\top} L_R x = \frac{1}{2} \sum_{i,j} R_{ij}(x_i - x_j)^2.
$$

In this example, the penalty is strongest when districts connected by larger inherited support diverge in value. So deviations between $B$ and $C$ or between $A$ and $B$ matter more than deviations between $A$ and $C$, and there is no direct penalty between $A$ and $D$ because they share no contextual regime.

### Step 6: Calculate the aggregate Laplacian Energy or Roughness

In this four-district example, the quadratic identity becomes

$$
x^{\top} L_R x
= 2(x_A - x_B)^2 + (x_A - x_C)^2 + 2(x_B - x_C)^2 + (x_B - x_D)^2 + 2(x_C - x_D)^2.
$$

This is the expanded form: a single scalar that sums every weighted disagreement implied by $R$.

For a vector, take

$$
x =
\begin{bmatrix}
1 \\ 2 \\ 3 \\ 4
\end{bmatrix}.
$$

Using the matrix form,

$$
L_R x =
\begin{bmatrix}
-4 \\ -2 \\ 2 \\ 4
\end{bmatrix}
\quad\text{so}\quad
x^{\top} L_R x = 14.
$$

Independently, using the pairwise expansion,

$$
2(1-2)^2 + (1-3)^2 + 2(2-3)^2 + (2-4)^2 + 2(3-4)^2 = 14.
$$

Both routes produce the same scalar, the total weighted friction carried by the vector over the relational graph.

### Step 7: Use the spectrum

Because the relational graph induced by $R$ is connected, the constant vector lies in the nullspace and there is a single zero eigenvalue. The spectrum of this example is approximately

$$
\lambda(L_R) \approx \{0,\; 2.764,\; 6.000,\; 7.236\}.
$$

The zero mode corresponds to a constant vector over all four districts. The positive eigenvalues describe increasingly rough modes of variation over the inherited relational structure.

## Interpretation notes for RelWeights

The objects in this note are easiest to read as graph operators on the analysis layer $\alpha$, where ties are induced by shared contextual supports in $\beta$. That changes the interpretation in an important way: smoothness, clustering, and imbalance are no longer defined by physical contiguity alone, but by common inherited regimes. This is the key move from overlay logic to operator logic [@anselin1988; @bavaud1998models].

### 1. Laplacian Energy or Roughness

It is small when districts that inherit the same supports also carry similar values, and it is large when strongly related districts diverge. In policy terms, this can be read as a mismatch score: how much the observed outcome departs from the contextual structure we believe should organize the space. If the signal is approximately constant over each relationally coherent region, the roughness is low; if sharp differences cut across shared supports, the roughness rises.

### 2. Eigenvalues and eigenvectors

The eigenvectors of $L_R$ are orthogonal modes of variation over the inherited support graph, and the eigenvalues tell us how rough each mode is. The zero eigenvalue corresponds to a constant component on each connected relational component. Small positive eigenvalues describe broad, low-frequency patterns that change gradually across the graph. Large eigenvalues describe high-frequency or localized contrasts, where values flip sharply across strong ties. In our setting, these are not merely geometric modes of adjacency; they are overlay-defined modes of similarity and separation [@chung1997spectral; @griffith2000linear; @tiefelsdorf2007semiparametric].

Equivalently, the symmetric Laplacian admits a spectral decomposition $L_R = U \Lambda U^{\top}$, where the columns of $U$ are orthonormal eigenvectors and $\Lambda$ collects the eigenvalues on the diagonal. That means any signal on $\alpha$ can be expanded in this basis and read as a mixture of smooth and rough relational modes, which is exactly why the Laplacian is useful for filtering, approximation, and decomposition [@chung1997spectral; @merris1994laplacian].

### 3. Graph-theoretic meaning of the discrete Laplacian

Node by node, the operator can be written as

$$
(L_R x)_i = \sum_j R_{ij}(x_i - x_j).
$$

This is a local imbalance measure. If $(L_Rx)_i$ is positive, district $i$ sits above the weighted average of the districts to which it is relationally tied. If it is negative, it sits below that relational neighborhood average. If it is near zero, the district is locally consistent with the values around it under the RelWeights graph. This is why the discrete Laplacian behaves like a second-order difference operator on a graph: it measures how far a node departs from the level implied by its weighted neighbors [@chung1997spectral].

### 4. Why this matters for spatial econometrics

In classical spatial econometrics, the spectrum of a weights operator already carries substantive information about spatial dependence, filtering, and admissible map patterns. Eigenvector spatial filtering makes this especially clear by using eigenvectors of a transformed connectivity operator as regressors that absorb structured dependence [@griffith2000linear; @tiefelsdorf2007semiparametric]. What RelWeights adds is a way to build that operator from cross-layer overlays rather than from contiguity or distance alone. Once the operator is in Laplacian form, the same spectral logic can be used for smoothing, basis construction, decomposition, and low-rank approximation on large (geospatial) systems [@mahdi2019efficient].
