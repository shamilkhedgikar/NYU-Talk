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

# Simulate

Generate outcomes under alternative dependence structures and test model recovery.

## Module question

Under what regimes should RelWeights outperform standard alternatives?

## Capabilities

- generate SAR, SEM, and SLX processes
- choose Queen, Rook, or RelWeights as the DGP operator
- fit models under correct and incorrect weights
- compare bias, power, residual dependence, and parameter recovery

## Notebook anchor

```{code-cell}
simulation_design = {
    "dgp_operators": ["queen", "rook", "relweights"],
    "model_families": ["sar", "sem", "slx"],
    "evaluation": ["bias", "power", "residual_dependence", "recovery"],
}

simulation_design
```

## Assets to add

- regime map for expected wins and losses
- bias and power comparison charts
- residual diagnostics after misspecification

