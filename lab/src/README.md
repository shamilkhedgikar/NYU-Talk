# Source Layout

Use this area for reusable Python code that should not live inline inside the book pages.

- `operators/`: RelWeights builders, normalization, Laplacians, sparse helpers
- `analysis/`: diagnostics, interpolation, simulation, model wrappers
- `io/`: data loading, layer alignment, export helpers
- `viz/`: matrix plots, maps, network views, comparison panels

The book pages should stay readable and thin. Heavy logic should move here.

