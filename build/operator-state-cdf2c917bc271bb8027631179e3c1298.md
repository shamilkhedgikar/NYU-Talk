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

# Operator State

The RelWeights operator state is the minimal reusable object that the rest of the lab needs:

$$
B \longrightarrow R \longrightarrow L_R
$$

Everything downstream in the lab is a reuse of this pipeline, whether the goal is smoothing, diagnostics, spectral decomposition, or operator export.

```{note}
This page follows Sections 1 through 6 of the reference note and uses the proposal's notation of an analysis layer $\alpha$ and an inherited support layer $\beta$.
```

## 1. Incidence matrices

Let $B \in \mathbb{R}^{n \times m}$ encode how each analysis unit $\alpha_i$ intersects inherited supports $\beta_k$.

Binary incidence:

$$
B_{\mathrm{bin}}(i,k) =
\begin{cases}
1 & \text{if } |\alpha_i \cap \beta_k| > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Area-overlap incidence:

$$
B_{\mathrm{area}}(i,k) = \frac{|\alpha_i \cap \beta_k|}{|\alpha_i|}
$$

with the row-standardization property

$$
\sum_k B_{\mathrm{area}}(i,k) = 1
$$

and support relation

$$
B_{\mathrm{bin}}(i,k) = \mathbf{1}\!\left(B_{\mathrm{area}}(i,k) > 0\right)
$$

So the first modeling choice is not the Laplacian yet. It is the definition of support: presence/absence with $B_{\mathrm{bin}}$ or proportional inheritance with $B_{\mathrm{area}}$.

## 2. RelWeights matrix

Once $B$ is fixed, relational similarity is induced by shared support:

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top})
$$

Entrywise,

$$
R_{ij} = \sum_k B_{ik} B_{jk}
$$

before diagonal removal. The interpretation is direct: two $\alpha$-units are close in $R$ when they inherit the same contextual supports in $\beta$.

Main properties:

- $R_{ij} \ge 0$
- $R$ is symmetric
- $\operatorname{rank}(R) \le \operatorname{rank}(B)$
- $B B^{\top}$ is a valid positive semidefinite Gram matrix

That last point matters because it shows the construction is not ad hoc. It is a kernel induced by the incidence embedding.

## 3. Laplacian matrix

Define the degree matrix from the row sums of $R$:

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

where $\delta_{ij}$ is the Kronecker delta.

At this point the overlay construction has become an operator. $R$ is similarity, while $L_R$ is the discrete curvature or friction operator that will drive most of the analyses in the lab.

## 4. Quadratic form identity

For any $x \in \mathbb{R}^n$,

$$
x^{\top} L_R x = \frac{1}{2} \sum_{i,j} R_{ij}(x_i - x_j)^2
$$

This identity is the most useful interpretive bridge between geometry and analysis. It says that $L_R$ measures relational roughness: variation is penalized when it occurs across units with strong shared support.

That is why the same matrix can serve:

- regularization in interpolation
- smoothness diagnostics
- spectral embeddings
- diffusion and propagation operators

## 5. Positive semidefiniteness

Because each term $(x_i - x_j)^2$ is nonnegative and each $R_{ij}$ is nonnegative,

$$
x^{\top} L_R x \ge 0
$$

for every $x$. Therefore $L_R$ is positive semidefinite, and its eigenvalues satisfy

$$
\lambda_k \ge 0
$$

This is the structural reason the operator behaves like a graph Laplacian even though the graph is induced by overlays rather than by border adjacency.

## 6. Nullspace property

The constant vector lies in the nullspace:

$$
L_R \mathbf{1} = 0
$$

So constants have zero relational curvature. If the relational graph is connected, the constant mode spans the entire zero eigenspace. If the graph is disconnected, the multiplicity of the zero eigenvalue tracks the number of connected components.

For the lab, the operational interpretation is simple:

- zero mode: baseline level
- low-frequency modes: smooth contextual gradients
- high-frequency modes: sharp disagreement across shared supports

## Minimal construction code

```{code-cell}
import numpy as np


def build_relweights(B):
    B = np.asarray(B, dtype=float)
    gram = B @ B.T
    R = gram - np.diag(np.diag(gram))
    D_R = np.diag(R.sum(axis=1))
    L_R = D_R - R
    return {"B": B, "gram": gram, "R": R, "D_R": D_R, "L_R": L_R}


toy = build_relweights(
    [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]
)

toy["R"], toy["L_R"]
```

## Source thread

The formal statements on this page come from Sections 1 through 6 of the local reference note, while the interpretation of overlays as reusable policy operators follows the proposal's worked framing of layered spatial systems.
