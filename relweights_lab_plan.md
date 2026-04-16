# RelWeights Lab — Web App Plan

## Core idea
Build a modular research web app centered on one reusable operator state:

- `R` = RelWeights matrix
- `W_R` = normalized RelWeights operator
- `L_R` = relational graph Laplacian

Every tab answers the master question:

> What changes when standard spatial weights are replaced by RelWeights?

---

## Main tabs

### 1. Build
Purpose: construct and inspect RelWeights.

Inputs:
- Giving basic background on W and it's applications (finance, econ, urban planning, USP, CS)
- analysis layer `alpha`
- inherited/support layer `beta`
- overlap rule (binary, area share, row-normalized, symmetric)
- diagonal handling
- normalization rule

Outputs:
- `W_{alpha beta}`
- `R = W_{alpha beta} W_{alpha beta}^T`
- `L_R = D_R - R`
- degree distribution
- connected components
- sparsity statistics
- matrix heatmap
- network/map comparison versus Queen and Rook

Why it matters:
- makes the "irregular but non-arbitrary lag" claim visible
- creates the operator used by all other tabs

---

### 2. Interpolation
Purpose: test whether RelWeights improves change-of-support estimation.

Modes:
- synthetic
- semi-synthetic
- uploaded real data

DGP families:
- smooth under adjacency
- smooth under RelWeights
- mixture
- piecewise constant
- adversarial / misaligned

Methods:
- areal weighting
- dasymetric
- pycnophylactic
- adjacency Laplacian smoothing
- RelWeights Laplacian smoothing
- hybrid dasymetric + RelWeights

Outputs:
- reconstructed fine field
- estimated target totals
- RMSE, MAE, bias, correlation
- mass-preservation error
- gradient/hotspot preservation
- performance by DGP class

Why it matters:
- shows where RelWeights wins, ties, or loses
- gives the main empirical validation framework

---

### 3. Externalities
Purpose: extend Anselin-style spatial externalities taxonomy.

Model families:
- SAR
- SEM
- SLX / Durbin

Main comparison:
- Queen/Rook: proximity-mediated interaction
- RelWeights: support-mediated interaction

Outputs:
- model equations
- plain-language interpretation cards
- spillover interpretation under each weights scheme
- map view of induced neighborhoods

Why it matters:
- formalizes RelWeights as a new interaction mechanism, not just another matrix

---

### 4. Diagnostics
Purpose: compare how different weight specifications explain residual dependence.

Diagnostics:
- Moran's I on residuals
- LM-lag
- robust LM-lag
- LM-error
- robust LM-error

Candidate weights:
- Queen
- Rook
- distance
- RelWeights
- uploaded GAL

Outputs:
- side-by-side diagnostics table
- interpretation panel for each weights choice

Why it matters:
- tests whether RelWeights captures dependence missed by standard contiguity

---

### 5. Spectral
Purpose: study eigenstructure of weights and Laplacians.

Objects:
- `W`
- `R`
- `L = D - W`
- `L_R = D_R - R`
- centered Moran operator where relevant

Features:
- eigenvalue scree plots
- mapped eigenvectors
- smooth-to-rough mode ordering
- decomposition of variables into eigenmodes
- roughness energy `x^T L x`

Why it matters:
- makes smoothness, diffusion, clustering, and filtering visible
- connects RelWeights to Moran/LISA/eigenvector spatial filtering ideas

---

### 6. Export
Purpose: generate usable weights artifacts.

Outputs:
- `.gal`
- `.gwt`
- sparse matrix
- CSV edge list
- metadata JSON documenting construction choices

Why it matters:
- makes RelWeights usable in GeoDa, PySAL, R `spdep`, and custom pipelines

---

### 7. Simulate
Purpose: simulate outcomes under alternative dependence structures.

Capabilities:
- generate SAR / SEM / SLX processes
- choose Queen / Rook / RelWeights as DGP operator
- fit models under correct and incorrect weights
- compare bias, power, residual dependence, and recovery

Why it matters:
- supports both pedagogy and theory-testing
- helps identify regimes where RelWeights should outperform alternatives

---

## Shared architecture

### Layer 1: Data
- upload layers and tabular data
- define source, target, and inherited layers
- compute intersections and joins

### Layer 2: Operators
- Queen / Rook / distance builders
- RelWeights builder
- normalization and thresholding
- Laplacian construction
- sparse eigensolvers

### Layer 3: Analysis modules
- interpolation
- externalities
- diagnostics
- spectral analysis
- export
- simulation

Design rule:
- `R`, `W_R`, and `L_R` are built once and reused everywhere

---

## MVP order

### Phase 1
- Build
- Export
- Spectral

### Phase 2
- Diagnostics
- Externalities

### Phase 3
- Interpolation

### Phase 4
- Simulation

---

## Strongest research claims supported by the app

### Claim 1
RelWeights generates irregular but non-arbitrary spatial lags.

Supported by:
- Build
- Export
- map/network comparisons

### Claim 2
RelWeights extends spatial externalities from proximity-mediated to support-mediated interaction.

Supported by:
- Externalities
- Diagnostics

### Claim 3
RelWeights can improve estimation when the inherited layer governs latent variation.

Supported by:
- Interpolation
- Simulation

### Claim 4
The Laplacian built from RelWeights reveals support-mediated smoothness, clustering, and spectral structure.

Supported by:
- Spectral

---

## Recommended implementation stack

Front end:
- React
- tab-based layout
- Plotly or ECharts for charts and matrix displays
- MapLibre or ArcGIS JS API for spatial interaction

Back end:
- Python
- GeoPandas
- Shapely
- SciPy sparse
- libpysal / esda / spreg
- networkx

Key implementation principles:
- sparse matrices from day one
- reusable operator state across tabs
- side-by-side comparison views wherever possible
- exportability of every operator specification
