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

# Construction Roadmap

```{note} Read / Run Split
:class: col-page-right
Use this page for the theory arc and the companion notebook for runnable matrix construction.

- Companion notebook: [lab-roadmap-notebook.ipynb](./lab-roadmap-notebook.ipynb)
- Formal derivation page: [Operator State](./operator-state.md)
- Source note: [RelWeights Laplacian reference](../../assets/papers/laplacians_spatial_operators_relweights.pdf)
- Context note: [NYU proposal](../../assets/papers/nyu_proposal_final.pdf)
```

RelWeights starts from an overlay problem rather than a border problem. The proposal frames the method as a way to convert layered spatial intersections into reusable operators, so the same construction can support diagnostics, smoothing, spectral analysis, and policy interpretation. In that framing, the geometry that matters is inherited support in $\beta$, not only shared boundaries in $\alpha$.

## Step 1: Choose the two layers

Let $\alpha = \{\alpha_1, \ldots, \alpha_n\}$ denote the analysis layer and let $\beta = \{\beta_1, \ldots, \beta_m\}$ denote the contextual or inherited support layer. Typical examples are:

- districts over hydrological basins
- tracts over school catchments
- neighborhoods over policy zones

The key design choice is that $\beta$ defines the support pattern that will later induce similarity among the $\alpha$ units.

## Step 2: Build the incidence matrix

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

## Step 3: Form the RelWeights kernel

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
