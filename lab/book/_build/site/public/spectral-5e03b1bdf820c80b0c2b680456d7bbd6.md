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

# Spectral

Study the eigenstructure of weights and Laplacians.

## Module question

What kinds of smoothness, clustering, and filtering become visible under RelWeights operators?

## Objects

- `W`
- `R`
- `L = D - W`
- `L_R = D_R - R`
- centered Moran operator where relevant

## Theory block

Roughness energy for a signal $x$:

$$
x^{\top} L_R x
$$

Low-energy modes should correspond to support-mediated smooth variation.

## Notebook anchor

```{code-cell}
spectral_tasks = [
    "eigenvalue_scree",
    "mapped_eigenvectors",
    "mode_ordering",
    "signal_decomposition",
    "roughness_energy",
]

spectral_tasks
```

## Assets to add

- scree plot template
- eigenvector map gallery
- comparison panel for adjacency versus RelWeights roughness
