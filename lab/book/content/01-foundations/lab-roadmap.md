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

RelWeights is easiest to understand as an extension of the spatial weights tradition. Before constructing a relational operator from overlays, it helps to state more formally what a spatial weights matrix is, what assumptions it carries, and how different neighborhood rules encode different theories of interaction.

## Spatial Weights Matrix: Definition

Let $S = \{1, \ldots, n\}$ denote a set of spatial units. A spatial weights matrix is an $n \times n$ matrix

$$
W = [w_{jk}]
$$

whose entries encode the relative spatial influence of unit $k$ on unit $j$. In the most common formulation, the weights satisfy $w_{jk} \ge 0$, and many applications row-standardize the matrix so that

$$
\sum_{k=1}^{m} w_{jk} = 1
$$

for each row $j$. Row-standardization is useful because it makes different weighting schemes more directly comparable, but it is a convention rather than a universal requirement. In practice, spatial weights may be binary, row-standardized, distance-decayed, kernel-smoothed, or left unstandardized until a later modeling stage.

$W$ captures the structural assumption about how spatial dependence propagates. Choosing $W$ means choosing what counts as a neighbor, how strongly neighbors matter, and whether influence is local, directional, group-based, or layered across multiple supports.

## Properties and Types of Spatial Weights Matrices

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
:name: fig-contiguity-weights

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


```

### Graph Objects

**Spatial weights matrices can be interpreted as graph operators!**

Each observation is a node, and each non-zero weight is an edge or tie in the graph. The matrix is always indexed by the observational units, but the visual ordering of rows and columns is arbitrary. What matters is the mapping between units and indices, not the particular row order printed on the page. Because most pairs are not neighbors, weights are usually stored sparsely by recording only the non-zero links.

Thus, even when a weights matrix appears to be a simple technical choice, it remains a substantive modeling decision.

We start from that broader view of spatial weights and then ask how to define a more flexible operator from overlays. The proposal frames the method as a way to convert layered spatial intersections into reusable operators, so the same construction can support diagnostics, smoothing, spectral analysis, and policy interpretation. In that framing, the geometry that matters is inherited support in $\beta$, not only shared boundaries in $\alpha$.

## RelWeights
Now we ask the question of what happens when we consider the idea that overlays interacting with each other. More formally, 

Let $\alpha = \{\alpha_1, \ldots, \alpha_n\}$ denote the analysis layer and let $\beta = \{\beta_1, \ldots, \beta_m\}$ denote the contextual or inherited support layer. Typical examples are:

- districts over hydrological basins
- tracts over school catchments
- neighborhoods over policy zones

The key design choice is that $\beta$ defines the support pattern that will later induce similarity among the $\alpha$ units.

```{figure} ../../assets/images/config-1.jpg
:width: 100%
:name: fig-datatype-weights

Here, $\alpha$ can be thought of 
```

### Step 1: Build the incidence matrix

The reference note starts with two versions of the incidence matrix $B$.

Binary support:

$$
B_{\mathrm{bin}}(i,k) =
\begin{cases}
1 & \text{if } |\alpha_i \cap \beta_k| > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Area-overlap support:

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

This is the point where an overlay becomes algebraic. Once the intersection structure is in $B$, the rest of the operator stack is matrix construction.

### Step 2: Form the RelWeights kernel

The relational similarity matrix is built from shared support:

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

Entrywise,

$$
R_{ij} = \sum_k B_{ik} B_{jk}
$$

before diagonal removal. In words, $R_{ij}$ measures how much contextual support units $\alpha_i$ and $\alpha_j$ inherit in common through $\beta$.

The reference note emphasizes the main properties immediately:

- $R_{ij} \ge 0$
- $R$ is symmetric
- $\operatorname{rank}(R) \le \operatorname{rank}(B)$
- $B B^{\top}$ is positive semidefinite before the diagonal is removed

## Step 4: Lift similarity into a Laplacian

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

## Step 5: Read the quadratic form as roughness

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

## Worked proposal example

The proposal gives a compact $4 \times 4$ binary example with districts $\alpha = \{A, B, C, D\}$ and contextual regimes $\beta = \{X, Y, Z, W\}$:

- $A$ intersects $X, Y$
- $B$ intersects $X, Y, Z$
- $C$ intersects $W, Y, Z$
- $D$ intersects $W, Z$

That yields

$$
B =
\begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1
\end{bmatrix}
$$

and therefore

$$
R =
\begin{bmatrix}
0 & 2 & 1 & 0 \\
2 & 0 & 2 & 1 \\
1 & 2 & 0 & 2 \\
0 & 1 & 2 & 0
\end{bmatrix},
\qquad
D_R =
\begin{bmatrix}
3 & 0 & 0 & 0 \\
0 & 5 & 0 & 0 \\
0 & 0 & 5 & 0 \\
0 & 0 & 0 & 3
\end{bmatrix}
$$

and

$$
L_R =
\begin{bmatrix}
3 & -2 & -1 & 0 \\
-2 & 5 & -2 & -1 \\
-1 & -2 & 5 & -2 \\
0 & -1 & -2 & 3
\end{bmatrix}
$$

```{code-cell}
import numpy as np

B = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ],
    dtype=float,
)

gram = B @ B.T
R = gram - np.diag(np.diag(gram))
D_R = np.diag(R.sum(axis=1))
L_R = D_R - R

B, R, L_R
```

## What this page sets up

This six-step construction is the backbone of the rest of the lab:

- `build` will construct $B$, $R$, and $L_R$ from real overlays
- `export` will move those operators into PySAL and related formats
- `spectral` will study the eigenstructure of $L_R$
- `diagnostics` and `externalities` will use the same operator to test and interpret dependence

The companion notebook now turns the same derivation into executable matrix checks.
