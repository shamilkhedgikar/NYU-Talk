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

# Diagnostics

Compare how competing weight specifications explain residual dependence.

## Module question

Does RelWeights capture dependence that standard contiguity misses?

## Candidate weights

- Queen
- Rook
- distance
- RelWeights
- uploaded GAL

## Diagnostics

- Moran's $I$
- LM-lag
- robust LM-lag
- LM-error
- robust LM-error

## Notebook anchor

```{code-cell}
diagnostic_suite = {
    "global": ["morans_i"],
    "lm": ["lm_lag", "rlm_lag", "lm_error", "rlm_error"],
}

diagnostic_suite
```

## Assets to add

- side-by-side diagnostics table
- interpretation cards for each weight family
- residual map paired with operator choice
