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

# Build

Construct and inspect RelWeights from overlay logic.

## Module question

What does the induced neighborhood look like when support overlap replaces plain contiguity?

## Inputs

- analysis layer `alpha`
- inherited or support layer `beta`
- overlap rule
- diagonal handling
- normalization rule

## Outputs

- `W_{alpha beta}`
- `R`
- `L_R`
- degree distribution
- connected components
- sparsity statistics
- matrix and map comparisons against Queen and Rook

## Theory block

$$
R = W_{\alpha \beta} W_{\alpha \beta}^{\top}
$$

The build module is where the claim of an irregular but non-arbitrary lag becomes visible.

## Notebook anchor

```{code-cell}
build_state = {
    "module": "build",
    "required_inputs": ["alpha", "beta", "overlap_rule", "diagonal", "normalization"],
    "planned_views": ["heatmap", "map", "network", "sparsity_summary"],
}

build_state
```

## Assets to add

- overlay diagram for `alpha` and `beta`
- heatmap for `R`
- side-by-side network maps for Queen, Rook, and RelWeights
