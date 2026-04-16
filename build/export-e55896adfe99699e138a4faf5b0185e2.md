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

# Export

Turn constructed operators into reusable artifacts.

## Module question

How do we make a RelWeights specification portable across GeoDa, PySAL, R, and custom workflows?

## Planned outputs

- `.gal`
- `.gwt`
- sparse matrix export
- CSV edge list
- metadata JSON recording construction choices

## Design note

Export is not an afterthought. The lab should make every operator specification reproducible and transferable.

## Notebook anchor

```{code-cell}
export_targets = [".gal", ".gwt", ".npz", ".csv", ".json"]
export_targets
```

## Assets to add

- example metadata schema
- one worked export/import round-trip
- minimal compatibility notes for GeoDa, `libpysal`, and `spdep`

