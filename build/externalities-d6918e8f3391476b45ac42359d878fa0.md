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

# Externalities

Extend spatial externalities from proximity-mediated interaction to support-mediated interaction.

## Module question

How should spillovers be interpreted when interaction comes from inherited support rather than adjacency?

## Model families

- SAR
- SEM
- SLX / Durbin

## Theory block

Standard SAR form:

$$
y = \rho W y + X \beta + \varepsilon
$$

RelWeights reframes the interaction operator. The interpretation shifts from geometric proximity to support-mediated exposure.

## Notebook anchor

```{code-cell}
model_families = ["sar", "sem", "slx", "durbin"]
comparison_frame = ["queen_rook", "relweights"]

model_families, comparison_frame
```

## Assets to add

- plain-language interpretation cards
- map view of induced neighborhoods
- one worked example of differing spillover narratives under `W` versus `R`
