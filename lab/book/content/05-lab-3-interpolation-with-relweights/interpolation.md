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

# Interpolation

Test whether RelWeights improves change-of-support estimation.

## Module question

When latent variation is governed by inherited support, does RelWeights reduce interpolation error?

## Modes

- synthetic
- semi-synthetic
- uploaded real data

## Methods

- areal weighting
- dasymetric
- pycnophylactic
- adjacency Laplacian smoothing
- RelWeights Laplacian smoothing
- hybrid dasymetric plus RelWeights

## Outputs

- reconstructed fine field
- estimated target totals
- RMSE, MAE, bias, correlation
- mass-preservation error
- hotspot or gradient preservation

## Notebook anchor

```{code-cell}
interpolation_methods = [
    "areal_weighting",
    "dasymetric",
    "pycnophylactic",
    "adjacency_laplacian",
    "relweights_laplacian",
    "hybrid_dasymetric_relweights",
]

interpolation_methods
```

## Assets to add

- DGP selector visuals
- reconstructed surface comparison
- error table grouped by DGP family

