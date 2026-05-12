---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python (relweights_lab)
  language: python
  name: relweights_lab
---

# Lab 3: Interpolation with RelWeights and PDFM Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamilkhedgikar/NYU-Talk/blob/dev/lab/book/content/05-lab-3-interpolation-with-relweights/interpolation.ipynb)

This lab uses the released **Population Dynamics Foundation Model (PDFM)** embeddings from Google Research as a fixed latent field over counties and ZCTAs, then asks what RelWeights adds on top of that representation [@agarwal2024pdfm]. The goal is not to retrain PDFM. The goal is to treat the embeddings as a rich geospatial feature space and build RelWeights-based transfer, superresolution, and imputation workflows on top of them.

The PDFM paper evaluates the embeddings on interpolation, extrapolation, and superresolution tasks. Here we adapt that logic to a RelWeights setting:

- **embedding transfer**: move county embeddings onto ZCTAs and regularize the transfer over a contextual graph
- **superresolution**: train on county labels and predict on ZCTAs
- **imputation**: predict missing ZCTA values from observed ZCTA values and graph structure

The key conceptual point is simple. PDFM gives a latent feature space. RelWeights gives a support-aware graph. Lab 3 studies how much support-aware transfer and smoothing improve change-of-support estimation once a strong embedding representation already exists.

## 1. Learning goals

By the end of this lab, you should be able to:

- load and subset released PDFM embeddings for counties and ZCTAs
- build county-to-ZCTA transfer operators
- construct county-membership RelWeights, thresholded RelWeights, and hybrid adjacency-plus-RelWeights graphs
- use embedding similarity to sparsify dense inherited supports
- compare baseline transfer with graph-smoothed transfer
- compare county-to-ZCTA superresolution and ZCTA imputation with and without RelWeights smoothing
- interpret error metrics together with relational roughness diagnostics

## 2. Why PDFM plus RelWeights?

PDFM embeddings are learned place representations. In the released notebooks, Google uses them as generic features for downstream prediction tasks such as county-to-ZCTA superresolution and ZIP-level imputation. That is already useful, but it leaves one important question open:

> if we know that target units inherit structure from broader supports, should we transfer and smooth predictions over a support-aware graph rather than treat every fine unit independently?

That is where RelWeights enters.

For this lab we use:

- **Maryland counties** as the coarse support
- **Maryland ZCTAs** as the fine support

This gives a manageable subset:

- 24 counties
- 459 ZCTAs

and keeps the notebook small enough to run locally while still being large enough for realistic graph comparisons.

## 3. Task setup

We will work with three tasks.

### Embedding transfer

Suppose county embeddings are observed on the coarse layer and ZCTA embeddings are observed on the fine layer. We first build a baseline county-to-ZCTA transfer

$$
E_{\text{zcta}}^{(0)} = \bar{B} E_{\text{county}},
$$

where $\bar{B}$ maps each ZCTA into its county. Since the released ZCTA embeddings are also available, we can treat them as ground truth and ask whether RelWeights smoothing improves the transfer.

### Superresolution

Train on county labels and predict at ZCTA scale. This mirrors Google's county-to-ZCTA superresolution setting, but we add graph smoothing after prediction.

### Imputation

Mask a subset of ZCTA labels, fit a model on the observed ZCTAs, and compare:

- embedding-only imputation
- anchored graph smoothing over the RelWeights graph

## 4. Load Maryland PDFM embeddings and geometry

```{code-cell} ipython3
from __future__ import annotations

import html
import importlib.util
import os
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

```{code-cell} ipython3
IN_COLAB = importlib.util.find_spec("google.colab") is not None
COLAB_PROJECT_ROOT = Path("/content/drive/MyDrive/NYU-Talk") if IN_COLAB else None
COLAB_PDFM_BASE = (
    Path("/content/drive/MyDrive/population-dynamics/data/pdfm_embeddings/v0/us")
    if IN_COLAB
    else None
)

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    print("Colab runtime detected.")
    print("Update COLAB_PROJECT_ROOT and/or COLAB_PDFM_BASE in this cell if your Drive layout is different.")
else:
    print("Local runtime detected.")
```

```{code-cell} ipython3
PDFM_BASE_CANDIDATES = [
    Path(r"C:\Non-Sync Data\DC OP\New_Approach\population-dynamics\data\pdfm_embeddings\v0\us"),
    Path("data/pdfm_embeddings/v0/us"),
]

if COLAB_PROJECT_ROOT is not None:
    PDFM_BASE_CANDIDATES.extend(
        [
            COLAB_PROJECT_ROOT / "data/pdfm_embeddings/v0/us",
            COLAB_PROJECT_ROOT / "population-dynamics/data/pdfm_embeddings/v0/us",
        ]
    )

if COLAB_PDFM_BASE is not None:
    PDFM_BASE_CANDIDATES.insert(0, COLAB_PDFM_BASE)


def resolve_pdfm_base(candidates: list[Path]) -> Path:
    """Return the first existing PDFM embedding directory."""

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the PDFM embedding folder. "
        "Update PDFM_BASE_CANDIDATES or COLAB_PDFM_BASE to match your local or Drive layout."
    )


PDFM_BASE = resolve_pdfm_base(PDFM_BASE_CANDIDATES)
STATE = "MD"
SEED = 20260420

SIMILARITY_QUANTILE = 0.75
N_SPECTRAL_CLUSTERS = 6

TRANSFER_LAMBDA = 0.20
SMOOTH_LAMBDA = 0.35
ANCHOR_WEIGHT = 25.0

RANDOM_TEST_FRACTION = 0.30
QUEEN_BUFFER_M = 25.0
QUEEN_BUFFER_SWEEP = (0.0, 25.0, 100.0)

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
```

```{code-cell} ipython3
def load_pdfm_subset(pdfm_base: Path, state: str = "MD") -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str]]:
    """Load county and ZCTA PDFM embeddings plus geometry for one state."""

    county = pd.read_csv(pdfm_base / "county_embeddings.csv")
    zcta = pd.read_csv(pdfm_base / "zcta_embeddings.csv")
    county_geo = gpd.read_file(pdfm_base / "county.geojson")[["place", "geometry"]]
    zcta_geo = gpd.read_file(pdfm_base / "zcta.geojson")[["place", "geometry"]]

    county = county[county["state"] == state].copy()
    zcta = zcta[zcta["state"] == state].copy()

    county = county.merge(county_geo, on="place", how="left")
    zcta = zcta.merge(zcta_geo, on="place", how="left")

    county = gpd.GeoDataFrame(county, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    zcta = gpd.GeoDataFrame(zcta, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)

    feature_cols = [col for col in county.columns if col.startswith("feature")]
    return county, zcta, feature_cols


county_gdf, zcta_gdf, embedding_cols = load_pdfm_subset(PDFM_BASE, state=STATE)

print(f"State: {STATE}")
print(f"County rows: {len(county_gdf)}")
print(f"ZCTA rows: {len(zcta_gdf)}")
print(f"Embedding dimensions: {len(embedding_cols)}")
```

```{code-cell} ipython3
county_gdf[["place", "county", "population"]].head()
```

```{code-cell} ipython3
zcta_gdf[["place", "county", "city", "population"]].head()
```

The PDFM embeddings are already aligned to the spatial units. We therefore do not need to derive latent features from imagery or train a graph neural network from scratch. Instead, we take the embeddings as given and focus on support-aware transfer.

## 5. Visualize the latent embedding field

Before building any RelWeights graph, it is useful to check that the embedding field already varies spatially.

```{code-cell} ipython3
def plot_geographies(gdfs, columns, titles, cmap="viridis", figsize=(12, 4), vmin=None, vmax=None):
    """Plot multiple geodataframes side by side with matched styling."""

    fig, axes = plt.subplots(1, len(gdfs), figsize=figsize)
    if len(gdfs) == 1:
        axes = [axes]

    for ax, gdf, col, title in zip(axes, gdfs, columns, titles):
        gdf.plot(column=col, cmap=cmap, linewidth=0.15, edgecolor="white", ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def resolve_lab3_interactive_dir() -> Path:
    """Resolve an interactive output directory for local, repo-root, or Colab runs."""

    target_parts = ("lab", "book", "content", "05-lab-3-interpolation-with-relweights")
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []

    if COLAB_PROJECT_ROOT is not None:
        candidates.append(COLAB_PROJECT_ROOT / "lab/book/content/05-lab-3-interpolation-with-relweights/interactive")

    # If the notebook is being run from inside its own section directory, prefer that.
    if cwd.name == "05-lab-3-interpolation-with-relweights":
        candidates.append(cwd / "interactive")

    # If the working directory is nested under the section directory, walk upward to it.
    for parent in [cwd, *cwd.parents]:
        if parent.name == "05-lab-3-interpolation-with-relweights":
            candidates.append(parent / "interactive")
            break

    # If we can find the repository root, construct the canonical section path from there.
    for parent in [cwd, *cwd.parents]:
        if (parent / "lab" / "book" / "myst.yml").exists():
            candidates.append(parent.joinpath(*target_parts) / "interactive")
            break
        if (parent / "myst.yml").exists() and parent.name == "book":
            candidates.append(parent / "content" / "05-lab-3-interpolation-with-relweights" / "interactive")
            break

    # Fall back to obvious relative choices.
    candidates.extend(
        [
            cwd / "content/05-lab-3-interpolation-with-relweights/interactive",
            cwd / "interactive",
        ]
    )

    for candidate in candidates:
        if candidate.parent.exists() or candidate.name == "interactive":
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

    fallback = cwd / "interactive"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def save_and_embed_html_document(html_text: str, output_path: Path, height: int = 720) -> None:
    """Save a standalone HTML document and embed it back into the notebook."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    rel_path = f"interactive/{output_path.name}"
    srcdoc = html.escape(output_path.read_text(encoding="utf-8"), quote=True)
    display(
        HTML(
            f"""
            <div style="margin: 0.5rem 0 1rem 0;">
              <iframe
                srcdoc="{srcdoc}"
                width="100%"
                height="{height}"
                style="border: 1px solid #d1d5db; border-radius: 8px; background: white;"
              ></iframe>
              <div style="margin-top: 0.5rem; font-size: 0.92rem;">
                <a href="{rel_path}" target="_blank" rel="noopener noreferrer">
                  Open standalone interactive graphic
                </a>
              </div>
            </div>
            """
        )
    )


def save_and_embed_folium_map(map_obj: folium.Map, output_path: Path, height: int = 720) -> None:
    """Save a Folium map to HTML and embed the saved document inline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    rel_path = f"interactive/{output_path.name}"
    srcdoc = html.escape(output_path.read_text(encoding="utf-8"), quote=True)
    display(
        HTML(
            f"""
            <div style="margin: 0.5rem 0 1rem 0;">
              <iframe
                srcdoc="{srcdoc}"
                width="100%"
                height="{height}"
                style="border: 1px solid #d1d5db; border-radius: 8px; background: white;"
              ></iframe>
              <div style="margin-top: 0.5rem; font-size: 0.92rem;">
                <a href="{rel_path}" target="_blank" rel="noopener noreferrer">
                  Open standalone interactive map
                </a>
              </div>
            </div>
            """
        )
    )
```

```{code-cell} ipython3
county_gdf["feature0_view"] = county_gdf["feature0"]
zcta_gdf["feature0_view"] = zcta_gdf["feature0"]

vmin = min(county_gdf["feature0_view"].min(), zcta_gdf["feature0_view"].min())
vmax = max(county_gdf["feature0_view"].max(), zcta_gdf["feature0_view"].max())

plot_geographies(
    [county_gdf, zcta_gdf],
    ["feature0_view", "feature0_view"],
    [f"{STATE} counties: feature0", f"{STATE} ZCTAs: feature0"],
    cmap="viridis",
    figsize=(12, 4),
    vmin=vmin,
    vmax=vmax,
)
```

This is already informative. PDFM behaves like a latent geospatial field: even one embedding dimension displays structured spatial variation. RelWeights will be used to move and smooth that latent structure across support layers.

## 6. Build county transfer and graph variants

We start with the county membership relation. Each ZCTA inherits county membership, so the basic county-to-ZCTA transfer matrix is one-hot:

$$
\bar{B}_{ik}
=
\begin{cases}
1 & \text{if ZCTA } i \text{ belongs to county } k,\\
0 & \text{otherwise.}
\end{cases}
$$

This gives a coarse RelWeights graph on ZCTAs:

$$
R_{\text{county}} = \bar{B}\bar{B}^{\top} - \operatorname{diag}(\bar{B}\bar{B}^{\top}).
$$

But that graph is very broad: all ZCTAs in the same county become linked. To sharpen it, we use PDFM embedding similarity as a thresholding device.

```{code-cell} ipython3
def build_county_transfer_matrix(zcta_df: pd.DataFrame, county_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map each ZCTA to its county embedding row."""

    county_keys = county_df["state"] + "|" + county_df["county"]
    zcta_keys = zcta_df["state"] + "|" + zcta_df["county"]
    county_lookup = pd.Series(np.arange(len(county_df)), index=county_keys)
    county_index = zcta_keys.map(county_lookup)

    if county_index.isna().any():
        missing = sorted(zcta_keys[county_index.isna()].unique())
        raise ValueError(f"Missing county matches for: {missing[:5]}")

    county_index = county_index.astype(int).to_numpy()
    B = np.zeros((len(zcta_df), len(county_df)), dtype=float)
    B[np.arange(len(zcta_df)), county_index] = 1.0
    return B, county_index


def build_relweights_from_membership(B: np.ndarray) -> np.ndarray:
    """Construct RelWeights from a one-hot membership matrix."""

    R = B @ B.T
    np.fill_diagonal(R, 0.0)
    return R


def build_queen_adjacency(
    gdf: gpd.GeoDataFrame,
    buffer_m: float = QUEEN_BUFFER_M,
    area_crs: str = "EPSG:3857",
) -> np.ndarray:
    """Build a robust queen-style adjacency matrix with an optional metric buffer."""

    work = gdf.reset_index(drop=True).to_crs(area_crs)
    if buffer_m > 0:
        work = work.copy()
        work["geometry"] = work.geometry.buffer(buffer_m)

    n = len(work)
    W = np.zeros((n, n), dtype=float)
    sindex = work.sindex

    for i, geom in enumerate(work.geometry):
        candidates = list(sindex.intersection(geom.bounds))
        for j in candidates:
            if j <= i:
                continue
            other = work.geometry.iloc[j]
            is_neighbor = geom.intersects(other) if buffer_m > 0 else geom.touches(other)
            if is_neighbor:
                W[i, j] = 1.0
                W[j, i] = 1.0
    return W


def graph_density(W: np.ndarray) -> float:
    n = W.shape[0]
    if n <= 1:
        return 0.0
    return float((W > 0).sum() / (n * (n - 1)))


def graph_diagnostics(W: np.ndarray) -> dict[str, float]:
    degrees = W.sum(axis=1)
    return {
        "avg_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "density": graph_density(W),
        "total_weight": float(W.sum()),
    }


def component_sizes(W: np.ndarray) -> list[int]:
    """Return connected-component sizes for a weighted graph."""

    adjacency = W > 0
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            seen[neighbors] = True
            stack.extend(neighbors.tolist())
        sizes.append(size)
    return sorted(sizes, reverse=True)


def edge_count(W: np.ndarray) -> int:
    """Return the number of undirected edges in a symmetric adjacency matrix."""

    return int(np.count_nonzero(np.triu(W > 0, 1)))


def queen_buffer_sweep(
    gdf: gpd.GeoDataFrame,
    buffer_distances: tuple[float, ...] = QUEEN_BUFFER_SWEEP,
) -> pd.DataFrame:
    """Summarize how a small geometric buffer changes the ZCTA queen graph."""

    rows = []
    for buffer_m in buffer_distances:
        W = build_queen_adjacency(gdf, buffer_m=buffer_m)
        sizes = component_sizes(W)
        rows.append(
            {
                "buffer_m": int(buffer_m),
                "edges": edge_count(W),
                "isolates": int(np.sum(W.sum(axis=1) == 0.0)),
                "components": len(sizes),
                "largest_component": sizes[0] if sizes else 0,
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    similarity: np.ndarray,
    support_mask: np.ndarray,
    quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> pd.DataFrame:
    """Summarize how quantile thresholds change graph density and fragmentation."""

    rows = []
    values = similarity[support_mask]
    for q in quantiles:
        tau_q = float(np.quantile(values, q))
        W_q = np.where(support_mask & (similarity >= tau_q), similarity, 0.0)
        sizes = component_sizes(W_q)
        rows.append(
            {
                "quantile": q,
                "tau": tau_q,
                "density": graph_density(W_q),
                "avg_degree": float(W_q.sum(axis=1).mean()),
                "n_components": len(sizes),
                "largest_component": sizes[0] if sizes else 0,
            }
        )
    return pd.DataFrame(rows)


def build_knn_similarity(coords: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric k-nearest-neighbor similarity graph from embedding coordinates."""

    sim = cosine_similarity(coords)
    np.fill_diagonal(sim, 0.0)
    W = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        idx = np.argpartition(sim[i], -k)[-k:]
        W[i, idx] = np.maximum(sim[i, idx], 0.0)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def maximum_spanning_tree_from_similarity(W: np.ndarray) -> np.ndarray:
    """Return the maximum-spanning-tree backbone of a nonnegative similarity graph."""

    W = np.maximum(W, W.T).astype(float)
    np.fill_diagonal(W, 0.0)
    if not np.any(W > 0):
        return np.zeros_like(W)

    dist = np.where(W > 0, 1.0 - W, 0.0)
    mst = minimum_spanning_tree(csr_matrix(dist))
    dist_tree = mst.toarray()
    sim_tree = np.where(dist_tree > 0, 1.0 - dist_tree, 0.0)
    sim_tree = np.maximum(sim_tree, sim_tree.T)
    np.fill_diagonal(sim_tree, 0.0)
    return sim_tree


def build_connectivity_overlay_map(
    gdf: gpd.GeoDataFrame,
    W_geo: np.ndarray,
    W_embed: np.ndarray,
    W_tree: np.ndarray,
    id_col: str = "place",
    title: str = "Embedding similarity versus geographic connectivity",
    max_nonlocal_edges: int = 260,
) -> folium.Map:
    """Overlay queen edges, nonlocal embedding edges, and a tree backbone on a Folium map."""

    centroids = gdf.to_crs(3857).geometry.centroid.to_crs(4326)
    center = [float(centroids.y.mean()), float(centroids.x.mean())]
    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    poly_layer = folium.FeatureGroup(name="ZCTA polygons", show=True)
    folium.GeoJson(
        gdf[[id_col, "county", "city", "geometry"]],
        style_function=lambda _feature: {
            "fillColor": "#00000000",
            "color": "#6b7280",
            "weight": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=[id_col, "county", "city"]),
    ).add_to(poly_layer)
    poly_layer.add_to(fmap)

    queen_layer = folium.FeatureGroup(name="Queen edges", show=False)
    tree_layer = folium.FeatureGroup(name="Embedding tree backbone", show=True)
    embed_layer = folium.FeatureGroup(name="Top nonlocal embedding edges", show=True)

    n = len(gdf)
    for i in range(n):
        for j in range(i + 1, n):
            if W_geo[i, j] > 0:
                folium.PolyLine(
                    [(centroids.iloc[i].y, centroids.iloc[i].x), (centroids.iloc[j].y, centroids.iloc[j].x)],
                    color="#9ca3af",
                    weight=1.0,
                    opacity=0.45,
                ).add_to(queen_layer)

            if W_tree[i, j] > 0:
                folium.PolyLine(
                    [(centroids.iloc[i].y, centroids.iloc[i].x), (centroids.iloc[j].y, centroids.iloc[j].x)],
                    color="#c2410c",
                    weight=3.0,
                    opacity=0.90,
                    tooltip=f"tree weight = {W_tree[i, j]:.3f}",
                ).add_to(tree_layer)

    nonlocal_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W_embed[i, j] > 0 and W_geo[i, j] == 0:
                nonlocal_edges.append((float(W_embed[i, j]), i, j))

    nonlocal_edges.sort(reverse=True)
    for weight, i, j in nonlocal_edges[:max_nonlocal_edges]:
        folium.PolyLine(
            [(centroids.iloc[i].y, centroids.iloc[i].x), (centroids.iloc[j].y, centroids.iloc[j].x)],
            color="#7c3aed",
            weight=2.0,
            opacity=0.55,
            tooltip=f"embedding similarity = {weight:.3f}",
        ).add_to(embed_layer)

    queen_layer.add_to(fmap)
    embed_layer.add_to(fmap)
    tree_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    title_html = f"""
    <div style="position: fixed; top: 14px; left: 56px; z-index: 1000;
                background: rgba(255,255,255,0.92); padding: 8px 12px;
                border: 1px solid #d1d5db; border-radius: 6px;
                font-size: 14px; font-weight: 600;">
      {title}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    return fmap


def build_regime_comparison_map(
    gdf: gpd.GeoDataFrame,
    regime_cols: list[str],
    layer_titles: list[str],
    id_col: str = "place",
    title: str = "ZCTA spectral regimes across graph constructions",
) -> folium.Map:
    """Build a toggleable Folium map comparing multiple categorical regime assignments."""

    centroids = gdf.to_crs(3857).geometry.centroid.to_crs(4326)
    center = [float(centroids.y.mean()), float(centroids.x.mean())]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    outline_layer = folium.FeatureGroup(name="ZCTA outlines", show=True)
    folium.GeoJson(
        gdf[[id_col, "county", "city", "geometry"]],
        style_function=lambda _feature: {
            "fillColor": "#00000000",
            "color": "#ffffff",
            "weight": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=[id_col, "county", "city"]),
    ).add_to(outline_layer)
    outline_layer.add_to(fmap)

    cmap = plt.get_cmap("tab20")
    legend_sections = []

    for regime_col, layer_title in zip(regime_cols, layer_titles):
        regime_values = sorted(int(v) for v in pd.Series(gdf[regime_col]).dropna().unique())
        color_lookup = {
            value: "#{:02x}{:02x}{:02x}".format(
                *(int(255 * channel) for channel in cmap(idx % 20)[:3])
            )
            for idx, value in enumerate(regime_values)
        }

        layer = folium.FeatureGroup(name=layer_title, show=(regime_col == "regime_hybrid"))
        geojson = folium.GeoJson(
            gdf[[id_col, "county", "city", regime_col, "geometry"]],
            style_function=lambda feature, _lookup=color_lookup, _col=regime_col: {
                "fillColor": _lookup.get(int(feature["properties"][_col]), "#9ca3af"),
                "color": "#374151",
                "weight": 0.35,
                "fillOpacity": 0.72,
            },
            highlight_function=lambda feature, _lookup=color_lookup, _col=regime_col: {
                "fillColor": _lookup.get(int(feature["properties"][_col]), "#6b7280"),
                "color": "#111827",
                "weight": 1.2,
                "fillOpacity": 0.90,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[id_col, "county", "city", regime_col],
                aliases=["ZCTA", "County", "City", "Regime"],
                localize=True,
                sticky=False,
                labels=True,
            ),
        )
        geojson.add_to(layer)
        layer.add_to(fmap)

        legend_rows = "".join(
            f"""
            <div style="display:flex; align-items:center; gap:6px; margin:2px 0;">
              <span style="display:inline-block; width:12px; height:12px; border-radius:2px; background:{color_lookup[value]}; border:1px solid #d1d5db;"></span>
              <span>Regime {value}</span>
            </div>
            """
            for value in regime_values
        )
        legend_sections.append(
            f"""
            <div style="margin-top: 8px;">
              <div style="font-weight: 600; margin-bottom: 4px;">{layer_title}</div>
              {legend_rows}
            </div>
            """
        )

    folium.LayerControl(collapsed=False).add_to(fmap)

    title_html = f"""
    <div style="position: fixed; top: 14px; left: 56px; z-index: 1000;
                background: rgba(255,255,255,0.94); padding: 8px 12px;
                border: 1px solid #d1d5db; border-radius: 6px;
                font-size: 14px; font-weight: 600;">
      {title}
    </div>
    """
    legend_html = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 1000;
                max-height: 56vh; overflow-y: auto;
                background: rgba(255,255,255,0.95); padding: 10px 12px;
                border: 1px solid #d1d5db; border-radius: 8px;
                font-size: 12px; line-height: 1.25; min-width: 210px;">
      <div style="font-weight: 700; margin-bottom: 6px;">Regime legend</div>
      <div style="color:#374151; margin-bottom: 6px;">
        In the <code>tab20</code> palette, the blue classes are <strong>Regime 0</strong>
        (dark blue) and <strong>Regime 1</strong> (light blue). Hover any polygon to see its exact regime id.
      </div>
      {''.join(legend_sections)}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap
```

```{code-cell} ipython3
B_county, county_membership_index = build_county_transfer_matrix(zcta_gdf, county_gdf)
R_county = build_relweights_from_membership(B_county)
queen_buffer_table = queen_buffer_sweep(zcta_gdf)
W_queen = build_queen_adjacency(zcta_gdf, buffer_m=QUEEN_BUFFER_M)

zcta_embeddings = zcta_gdf[embedding_cols].to_numpy()
embedding_similarity = cosine_similarity(zcta_embeddings)
np.fill_diagonal(embedding_similarity, 0.0)

county_pairs = R_county > 0
tau = float(np.quantile(embedding_similarity[county_pairs], SIMILARITY_QUANTILE))

R_thresholded = np.where(county_pairs & (embedding_similarity >= tau), embedding_similarity, 0.0)
R_hybrid = W_queen + R_thresholded

graph_summary = pd.DataFrame(
    {
        "buffered_queen": graph_diagnostics(W_queen),
        "county_relweights": graph_diagnostics(R_county),
        "thresholded_relweights": graph_diagnostics(R_thresholded),
        "queen_hybrid": graph_diagnostics(R_hybrid),
    }
).T

print(f"Similarity threshold tau: {tau:.3f}")
graph_summary.round(3)
```

```{code-cell} ipython3
queen_buffer_table
```

The thresholded graph is the bridge back to the clustering chapter:

- raw county membership is too easy to create
- PDFM similarity gives an empirical way to keep only strong inherited ties
- the hybrid graph preserves local geography while adding support-aware contextual edges

There is one more practical issue before we interpret any spectral result: **exact queen contiguity on raw ZCTA polygons is too brittle**. Maryland ZCTAs include multipart polygons, shoreline fragments, and tiny geometric gaps. If queen adjacency is constructed with an exact `touches` test, the geographic graph becomes artificially disjoint. That would make the later fragmentation story look like a property of RelWeights when it is really a property of the raw polygon topology.

The buffer table above makes that point concrete:

- with an exact `0 m` queen, the graph collapses to only `57` edges and hundreds of isolates
- with a tiny `25 m` metric buffer, most false gaps close and the graph becomes broadly connected
- with `100 m`, the statewide graph is even more connected, but begins to smooth away more of the original cartographic precision

So the lab uses a **buffered queen graph with `25 m`** as the geographic baseline. The idea is modest: treat very small slivers, shoreline gaps, and precision mismatches as artifacts rather than substantive separations. This keeps the geographic operator faithful to the broad ZCTA geography while avoiding false fragmentation in the downstream Laplacian and clustering analysis.

```{code-cell} ipython3
within_county_similarity = pd.Series(embedding_similarity[county_pairs]).describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
)
threshold_sweep_table = threshold_sweep(embedding_similarity, county_pairs)
hybrid_isolates = int(np.sum(R_hybrid.sum(axis=1) == 0.0))

within_county_similarity.round(4)
```

```{code-cell} ipython3
threshold_sweep_table.round(4)
```

The current notebook uses

$$
\tau = 0.896,
$$

which is the `0.75` quantile of within-county ZCTA cosine similarity in Maryland for this run. That makes it a deliberately high cutoff: only the strongest quarter of county-internal embedding ties survive. In this sample the within-county similarity values range from roughly `0.18` to `0.98`, with a median around `0.85`. The sweep table shows the tradeoff directly:

- lower `\tau` keeps more contextual edges but makes the projected graph denser
- higher `\tau` sharpens similarity and locality, but fragments the graph into more components, many of them isolates
- the chosen `0.75` quantile is a middle-ground sparsifier rather than a universal rule

This is the first important interpretation point for RelWeights with foundation-model embeddings: `\tau` is not just a tuning constant. It controls how much of the inherited support survives after the PDFM geometry is used as a filter.

## 7. Spectral comparison of queen, county, thresholded, and hybrid graphs

Up to this point the notebook has constructed several distinct operators:

- `W_queen`: buffered geometric adjacency on ZCTAs
- `R_county`: inherited support induced only by county membership
- `R_thresholded`: county support filtered by PDFM embedding similarity
- `W_queen + R_thresholded`: the final hybrid graph used later for interpolation

If we only inspect the hybrid graph, we lose the ability to say which parts of the spectral signal come from ordinary geography, which come from inherited support, and which come only from their combination. So in this section we compare the four operators directly through the symmetric normalized Laplacian

$$
L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}.
$$

For each graph, the low end of the spectrum tells us about connectivity and large-scale organization:

- the multiplicity of the eigenvalue `0` counts connected components among positive-degree nodes
- the first nonzero eigenvalues describe the easiest smooth modes on that graph
- large gaps after the zero eigenspace suggest stronger component or regime separation

So the spectral comparison below should be read as a decomposition of support structure:

- `W_queen` tells us what buffered geographic adjacency alone looks like
- `R_county` tells us what broad inherited support alone looks like
- `R_thresholded` shows what survives after PDFM similarity sharpens those inherited ties
- the hybrid graph shows whether contextual ties add information without destroying geographic coherence

```{code-cell} ipython3
def normalized_laplacian(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric normalized Laplacian and similarity operator."""

    degrees = W.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    L = np.eye(W.shape[0]) - S
    return L, S


def spectral_regimes(
    W: np.ndarray, n_clusters: int, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute eigenpairs and cluster nodes in low-order spectral space."""

    L_sym, _ = normalized_laplacian(W)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    zero_multiplicity = int(np.sum(eigvals < 1e-9))
    start = max(1, zero_multiplicity)
    stop = min(eigvecs.shape[1], start + n_clusters - 1)
    spectral_coords = eigvecs[:, start:stop]
    spectral_coords = spectral_coords / (np.linalg.norm(spectral_coords, axis=1, keepdims=True) + 1e-12)
    regimes = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(spectral_coords)
    return eigvals, eigvecs, regimes, zero_multiplicity
```

```{code-cell} ipython3
graph_variants = {
    "queen": W_queen,
    "county": R_county,
    "thresholded": R_thresholded,
    "hybrid": R_hybrid,
}

spectral_results: dict[str, dict[str, object]] = {}
spectral_rows = []

for name, W_graph in graph_variants.items():
    eigvals, eigvecs, regimes, zero_mult = spectral_regimes(W_graph, N_SPECTRAL_CLUSTERS, seed=SEED)
    component_sizes_graph = component_sizes(W_graph)
    isolates_graph = int(np.sum(W_graph.sum(axis=1) == 0.0))
    first_positive_idx = zero_mult
    first_positive = float(eigvals[first_positive_idx]) if first_positive_idx < len(eigvals) else np.nan
    next_positive = float(eigvals[first_positive_idx + 1]) if first_positive_idx + 1 < len(eigvals) else np.nan

    spectral_results[name] = {
        "W": W_graph,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "regimes": regimes,
        "zero_mult": zero_mult,
        "component_sizes": component_sizes_graph,
        "isolates": isolates_graph,
    }

    zcta_gdf[f"regime_{name}"] = regimes
    spectral_rows.append(
        {
            "graph": name,
            "zero_eigenvalues": zero_mult,
            "connected_components": len(component_sizes_graph),
            "isolates": isolates_graph,
            "largest_component": component_sizes_graph[0] if component_sizes_graph else 0,
            "avg_degree": float(W_graph.sum(axis=1).mean()),
            "density": graph_density(W_graph),
            "lambda_first_positive": first_positive,
            "lambda_next": next_positive,
        }
    )

spectral_table = pd.DataFrame(spectral_rows).set_index("graph")
zcta_gdf["regime"] = zcta_gdf["regime_hybrid"]

spectral_table.round(4)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8.2, 4.2))
colors = {
    "queen": "#4C72B0",
    "county": "#DD8452",
    "thresholded": "#55A868",
    "hybrid": "#C44E52",
}

for name, result in spectral_results.items():
    ax.plot(
        np.arange(1, 16),
        result["eigvals"][:15],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=name,
        color=colors[name],
    )

ax.set_title("Low-order eigenvalues across graph constructions")
ax.set_xlabel("Ordered mode")
ax.set_ylabel("Eigenvalue")
ax.grid(alpha=0.25)
ax.legend(ncol=2)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf, zcta_gdf],
    ["regime_queen", "regime_county", "regime_thresholded", "regime_hybrid"],
    [
        f"{STATE} ZCTA regimes from queen adjacency",
        f"{STATE} ZCTA regimes from county RelWeights",
        f"{STATE} ZCTA regimes from thresholded RelWeights",
        f"{STATE} ZCTA regimes from the hybrid graph",
    ],
    cmap="tab20",
    figsize=(16, 5),
)
```

```{code-cell} ipython3
interactive_dir = resolve_lab3_interactive_dir()
regime_map = build_regime_comparison_map(
    zcta_gdf,
    ["regime_queen", "regime_county", "regime_thresholded", "regime_hybrid"],
    [
        "Queen regimes",
        "County RelWeights regimes",
        "Thresholded RelWeights regimes",
        "Hybrid regimes",
    ],
    title=f"{STATE} ZCTA regime comparison: queen, county, thresholded, and hybrid",
)
save_and_embed_folium_map(
    regime_map,
    interactive_dir / "zcta-regime-comparison.html",
    height=760,
)
```

These regime maps are not the final prediction object. They are an empirical structure check. The point is to inspect how support structure changes as the graph changes.

The interactive version makes that comparison easier to read. Turn layers on and off against the same Carto basemap, then hover a ZCTA to see its regime id under the active graph. The floating legend also identifies the `tab20` colors directly, so the blue classes are no longer anonymous: dark blue is `Regime 0`, light blue is `Regime 1`.

The first ten hybrid eigenvalues can display as `-0.` because of floating-point rounding, not because the normalized Laplacian is truly negative. For a symmetric normalized Laplacian,

$$
L_{\mathrm{sym}} = I - D^{-1/2} R D^{-1/2},
$$

the spectrum is positive semidefinite. So values printed as `-0.` should be read as numerical zero. The more substantive diagnostic is the **zero-eigenvalue multiplicity**. For the symmetric normalized Laplacian, that multiplicity equals the number of connected components **among nodes with positive degree**. Isolated nodes are treated differently: because they have degree zero, they contribute eigenvalue `1`, not `0`.

That is why the hybrid graph can report both:

- `28` zero eigenvalues
- `71` connected components in the raw hybrid graph
- `43` isolated ZCTAs

Those numbers are exactly consistent: `71 - 43 = 28`. So the graph is not merely clustered. It is heavily fragmented, and a large share of that fragmentation comes from nodes that lost all contextual ties after thresholding.

This comparative view is the more informative one:

- `queen` usually has the strongest geographic coherence and the smallest zero eigenspace
- `county` often looks like a block graph, because all within-county ties are kept and cross-county ties are absent
- `thresholded` is where PDFM similarity becomes restrictive; if the cutoff is aggressive, fragmentation appears immediately
- `hybrid` reveals whether adding contextual ties to geography preserves broad connectivity or instead creates many disconnected mini-systems

So if the thresholded and hybrid spectra are dominated by many zeros, that is telling you something substantive: the support-aware graph is not yet behaving like one statewide manifold. It is behaving like many local islands. In that case, spectral clustering is still possible, but the interpretation changes. The first nontrivial modes are no longer mainly describing broad regimes; they are first accounting for graph fragmentation.

That is exactly why the helper function skips the whole zero-eigenspace before running `k`-means on the spectral coordinates. The hybrid regimes shown later should therefore be read as clusters **conditional on the graph already being fragmented** by the PDFM similarity filter.

## 8. Experiment A: embedding transfer

The cleanest benchmark in this lab is that we have both county embeddings and ZCTA embeddings. So we can:

1. transfer county embeddings down to ZCTAs by county membership
2. smooth the transferred embeddings over a RelWeights graph
3. compare both against the actual ZCTA embeddings

```{code-cell} ipython3
def smooth_matrix(X0: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    """Apply mean-preserving graph smoothing to a matrix of node features."""

    L = np.diag(W.sum(axis=1)) - W
    A = np.eye(W.shape[0]) + lam * L
    return np.linalg.solve(A, X0)


def rowwise_cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity row by row."""

    num = np.sum(A * B, axis=1)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return num / (denom + 1e-12)


def embedding_metrics(E_true: np.ndarray, E_pred: np.ndarray) -> dict[str, float]:
    """Summarize transfer quality against a target embedding matrix."""

    diff = E_pred - E_true
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "mean_cosine": float(rowwise_cosine_similarity(E_true, E_pred).mean()),
    }
```

```{code-cell} ipython3
county_embeddings = county_gdf[embedding_cols].to_numpy()
zcta_embeddings = zcta_gdf[embedding_cols].to_numpy()

E0 = B_county @ county_embeddings
E_county_smoothed = smooth_matrix(E0, R_county, lam=TRANSFER_LAMBDA)
E_thresholded_smoothed = smooth_matrix(E0, R_thresholded, lam=TRANSFER_LAMBDA)
E_hybrid_smoothed = smooth_matrix(E0, R_hybrid, lam=TRANSFER_LAMBDA)

transfer_results = pd.DataFrame(
    [
        {"method": "county transfer", **embedding_metrics(zcta_embeddings, E0)},
        {"method": "county transfer + county RelWeights", **embedding_metrics(zcta_embeddings, E_county_smoothed)},
        {"method": "county transfer + thresholded RelWeights", **embedding_metrics(zcta_embeddings, E_thresholded_smoothed)},
        {"method": "county transfer + hybrid graph", **embedding_metrics(zcta_embeddings, E_hybrid_smoothed)},
    ]
)

transfer_results.round(4)
```

In this run the four transfer variants are almost tied. The baseline county transfer gives an embedding RMSE around `0.56`, and the hybrid graph only nudges that down very slightly. This is not a failure of RelWeights. It reflects the structure of the benchmark:

- the starting transfer is piecewise constant within counties
- the county-based RelWeights graph mostly mixes ZCTAs that already share that same county-level signal
- the thresholded graph is sparse enough that it cannot invent much fine-scale variation on its own
- and, critically, the true ZCTA embeddings are already available, so this experiment is a calibration test rather than a real missing-data problem

So the embedding-transfer result should be interpreted as a **sanity check**: support-aware smoothing does not materially damage the transferred latent field, but with both county and ZCTA embeddings already present there is only limited room for improvement.

To visualize the transfer, project the embeddings onto the first principal component of the actual ZCTA embedding cloud.

```{code-cell} ipython3
pca = PCA(n_components=1, random_state=SEED)
pca.fit(zcta_embeddings)

zcta_gdf["pc1_actual"] = pca.transform(zcta_embeddings).ravel()
zcta_gdf["pc1_transfer"] = pca.transform(E0).ravel()
zcta_gdf["pc1_hybrid"] = pca.transform(E_hybrid_smoothed).ravel()

pc_vmin = zcta_gdf[["pc1_actual", "pc1_transfer", "pc1_hybrid"]].min().min()
pc_vmax = zcta_gdf[["pc1_actual", "pc1_transfer", "pc1_hybrid"]].max().max()

plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf],
    ["pc1_actual", "pc1_transfer", "pc1_hybrid"],
    ["Actual ZCTA embedding PC1", "County transfer only", "County transfer + hybrid smoothing"],
    cmap="viridis",
    figsize=(14, 4),
    vmin=pc_vmin,
    vmax=pc_vmax,
)
```

This is a direct change-of-support experiment on the latent field itself. The question is not whether PDFM is good in general. The question is whether RelWeights produces a transfer that is closer to the fine-scale embeddings already released by PDFM.

## 9. Experiment B: county-to-ZCTA superresolution

Now move from embeddings to a scalar target. We use `log1p(population)` because it is available on both counties and ZCTAs and is strongly skewed.

The workflow is:

1. fit a county model using county embeddings
2. predict on ZCTA embeddings
3. smooth the ZCTA predictions with the county, thresholded, or hybrid graph

```{code-cell} ipython3
def fit_ridge(train_X: np.ndarray, train_y: np.ndarray, alpha: float = 2.0):
    """Fit a standardized ridge regression model."""

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_X, train_y)
    return model


def smooth_signal(x0: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    """Mean-preserving graph smoothing for a scalar field."""

    L = np.diag(W.sum(axis=1)) - W
    A = np.eye(W.shape[0]) + lam * L
    return np.linalg.solve(A, x0)


def anchored_smoother(
    x0: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
    W: np.ndarray,
    lam: float,
    anchor_weight: float,
) -> np.ndarray:
    """Smooth a field while strongly anchoring the observed nodes."""

    L = np.diag(W.sum(axis=1)) - W
    M = np.diag(observed_mask.astype(float))
    A = np.eye(W.shape[0]) + lam * L + anchor_weight * M
    b = x0 + anchor_weight * observed_mask.astype(float) * observed_values
    return np.linalg.solve(A, b)


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics for scalar targets."""

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def roughness_summary(x: np.ndarray, graphs: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute graph energy and Geary-like normalization across several graphs."""

    z = x - x.mean()
    denom = float(z @ z)
    out = {}
    for name, W in graphs.items():
        L = np.diag(W.sum(axis=1)) - W
        energy = float(z @ L @ z)
        S0 = float(W.sum())
        geary = np.nan if S0 == 0 or denom == 0 else ((len(z) - 1) / S0) * energy / denom
        out[f"{name}_energy"] = energy
        out[f"{name}_geary"] = geary
    return out
```

```{code-cell} ipython3
county_target = np.log1p(county_gdf["population"].to_numpy())
zcta_target = np.log1p(zcta_gdf["population"].to_numpy())

county_model = fit_ridge(county_embeddings, county_target, alpha=2.0)
superres_base = county_model.predict(zcta_embeddings)
superres_county = smooth_signal(superres_base, R_county, lam=SMOOTH_LAMBDA)
superres_threshold = smooth_signal(superres_base, R_thresholded, lam=SMOOTH_LAMBDA)
superres_hybrid = smooth_signal(superres_base, R_hybrid, lam=SMOOTH_LAMBDA)

superres_results = pd.DataFrame(
    [
        {"method": "county model only", **scalar_metrics(zcta_target, superres_base)},
        {"method": "county model + county RelWeights", **scalar_metrics(zcta_target, superres_county)},
        {"method": "county model + thresholded RelWeights", **scalar_metrics(zcta_target, superres_threshold)},
        {"method": "county model + hybrid graph", **scalar_metrics(zcta_target, superres_hybrid)},
    ]
)

superres_results.round(4)
```

```{code-cell} ipython3
superres_graphs = {
    "county": R_county,
    "thresholded": R_thresholded,
    "hybrid": R_hybrid,
}

superres_roughness = pd.DataFrame(
    [
        {"field": "actual zcta target", **roughness_summary(zcta_target, superres_graphs)},
        {"field": "county model only", **roughness_summary(superres_base, superres_graphs)},
        {"field": "county model + hybrid graph", **roughness_summary(superres_hybrid, superres_graphs)},
    ]
)

superres_roughness.round(4)
```

The superresolution result is more demanding. Here the model is trained only on county targets and then transported to ZCTAs through the embedding space. The scalar metrics show two things at once:

- the county-trained regression captures some cross-sectional structure, since the correlations stay around `0.70`
- but the negative `R^2` values mean that county-to-ZCTA transport is still missing a large amount of fine-scale heterogeneity in the target

The roughness table clarifies the geometry behind this. The actual ZCTA population surface has much larger relational energy than the predicted surfaces, especially under the thresholded and hybrid graphs. In other words, the county-trained prediction is **too smooth** relative to the support-aware graph. Hybrid smoothing regularizes that surface even further, which may be useful for coherence, but it cannot recover the within-county heterogeneity that was never identified by the county-level model in the first place.

So the superresolution lesson is not "graph smoothing always improves prediction." It is more precise:

- smoothing is good at enforcing a support-aware geometry
- smoothing is not a substitute for genuinely fine-scale signal
- county-scale supervision alone is often too coarse to reproduce ZCTA-level variation, even when fine-scale PDFM embeddings are available as predictors

```{code-cell} ipython3
zcta_gdf["target_actual"] = zcta_target
zcta_gdf["target_superres_base"] = superres_base
zcta_gdf["target_superres_hybrid"] = superres_hybrid
zcta_gdf["superres_residual"] = zcta_target - superres_hybrid

target_vmin = zcta_gdf[["target_actual", "target_superres_base", "target_superres_hybrid"]].min().min()
target_vmax = zcta_gdf[["target_actual", "target_superres_base", "target_superres_hybrid"]].max().max()

plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf, zcta_gdf],
    ["target_actual", "target_superres_base", "target_superres_hybrid", "superres_residual"],
    ["Actual log population", "Superresolution: embedding only", "Superresolution: hybrid smoothing", "Hybrid residual"],
    cmap="viridis",
    figsize=(16, 4),
    vmin=target_vmin,
    vmax=target_vmax,
)
```

## 10. Experiment C: ZCTA imputation

Imputation keeps the support fixed and removes labels on part of the fine layer. We randomly hide `30%` of Maryland ZCTAs, train on the observed ZCTAs, and then compare:

- embedding-only imputation
- county-graph anchored smoothing
- hybrid-graph anchored smoothing

The anchored smoother uses a large penalty on observed units so that the graph regularization mainly redistributes information toward the missing units.

```{code-cell} ipython3
rng = np.random.default_rng(SEED)
test_mask = rng.random(len(zcta_gdf)) < RANDOM_TEST_FRACTION
train_mask = ~test_mask

impute_model = fit_ridge(zcta_embeddings[train_mask], zcta_target[train_mask], alpha=1.5)
impute_base_all = impute_model.predict(zcta_embeddings)
impute_county_all = anchored_smoother(
    impute_base_all,
    observed_mask=train_mask,
    observed_values=zcta_target,
    W=R_county,
    lam=SMOOTH_LAMBDA,
    anchor_weight=ANCHOR_WEIGHT,
)
impute_hybrid_all = anchored_smoother(
    impute_base_all,
    observed_mask=train_mask,
    observed_values=zcta_target,
    W=R_hybrid,
    lam=SMOOTH_LAMBDA,
    anchor_weight=ANCHOR_WEIGHT,
)

impute_results = pd.DataFrame(
    [
        {"method": "embedding only", **scalar_metrics(zcta_target[test_mask], impute_base_all[test_mask])},
        {"method": "county anchored smoother", **scalar_metrics(zcta_target[test_mask], impute_county_all[test_mask])},
        {"method": "hybrid anchored smoother", **scalar_metrics(zcta_target[test_mask], impute_hybrid_all[test_mask])},
    ]
)

impute_results.round(4)
```

Imputation behaves differently because the support is fixed and only part of the target is missing. In this setting the hybrid anchored smoother is the best method in the notebook:

- `embedding only` is already a strong baseline because true ZCTA embeddings are observed
- `county anchored smoother` is too blunt and degrades performance
- `hybrid anchored smoother` improves RMSE, MAE, and `R^2` over the embedding-only baseline

That is the clearest positive signal for RelWeights in this lab. Once some fine-scale labels are known, the hybrid graph helps redistribute information to nearby and contextually similar ZCTAs without collapsing everything back to the county average.

```{code-cell} ipython3
zcta_gdf["is_test_holdout"] = test_mask.astype(int)
zcta_gdf["target_impute_base"] = impute_base_all
zcta_gdf["target_impute_hybrid"] = impute_hybrid_all
zcta_gdf["impute_residual_test"] = np.where(test_mask, zcta_target - impute_hybrid_all, np.nan)

plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf, zcta_gdf],
    ["is_test_holdout", "target_impute_base", "target_impute_hybrid", "impute_residual_test"],
    ["Held-out ZCTAs", "Imputation: embedding only", "Imputation: hybrid anchored", "Hybrid residual on test ZCTAs"],
    cmap="viridis",
    figsize=(16, 4),
)
```

## 11. Compare the experiments through graph structure

The final question is not just which method gives lower RMSE. It is also whether the resulting field is better aligned with the support-aware graph.

```{code-cell} ipython3
comparison_table = pd.concat(
    [
        transfer_results.assign(experiment="embedding_transfer"),
        superres_results.assign(experiment="superresolution"),
        impute_results.assign(experiment="imputation"),
    ],
    ignore_index=True,
)

comparison_table.round(4)
```

```{code-cell} ipython3
regime_error = pd.DataFrame(
    {
        "regime": zcta_gdf["regime"],
        "actual": zcta_target,
        "pred_hybrid_superres": superres_hybrid,
        "pred_hybrid_impute": impute_hybrid_all,
        "is_test": test_mask,
    }
)

regime_error["superres_abs_error"] = np.abs(regime_error["actual"] - regime_error["pred_hybrid_superres"])
regime_error["impute_abs_error"] = np.where(
    regime_error["is_test"],
    np.abs(regime_error["actual"] - regime_error["pred_hybrid_impute"]),
    np.nan,
)

regime_summary = regime_error.groupby("regime").agg(
    n_zctas=("regime", "size"),
    superres_mae=("superres_abs_error", "mean"),
    impute_mae=("impute_abs_error", "mean"),
).reset_index()

regime_summary.round(4)
```

This regime table is one of the benefits of carrying clustering ideas into interpolation. Once the hybrid graph has defined support-aware regimes, we can ask whether some regions are systematically easier or harder to transfer and predict.

Taken together, the three experiments tell a coherent story.

- **Embedding transfer** barely changes because the fine-scale target embeddings are already observed and the transfer problem is overidentified.
- **Superresolution** remains hard because the response is learned at county scale, while the evaluation lives at ZCTA scale.
- **Imputation** is where RelWeights contributes the most, because the task is genuinely local: some ZCTA labels are known, some are missing, and the hybrid graph provides a support-aware path for borrowing strength.

This also answers the practical question about having both scales of PDFM embeddings.

If both county and ZCTA embeddings are present, Lab 3 is mainly a **diagnostic lab**. It tells us how much support-aware transfer and smoothing alter a strong latent representation that already exists at both scales. If the fine-scale embeddings were *not* present, the interpretation changes:

- the county-to-ZCTA transfer becomes the actual latent-field estimator, not just a benchmark
- the thresholded or hybrid RelWeights graph becomes more valuable because it is one of the few ways to recover sub-county structure
- low-rank embedding geometry, historical labels, or external Data Commons covariates become much more important as auxiliary sources of fine-scale variation

## 12. Multi-target evaluation with Data Commons variables

The Google PDFM notebook does not stop at population. It also uses Data Commons variables such as median age, median income, educational attainment, and health prevalence rates. That matters here because a RelWeights graph should not be evaluated against only one downstream label. Some targets are more local, some more support-driven, and some may benefit more from graph smoothing than others.

For reproducible published builds, this section reads a cached latest Data Commons extract when available. To refresh the cache, install the Data Commons V2 client and set `DATACOMMONS_API_KEY` or `DC_API_KEY` before running the notebook.

```{code-cell} ipython3
DC_LABELS = [
    "Count_Person",
    "Count_Person_EducationalAttainmentBachelorsDegreeOrHigher",
    "Median_Age_Person",
    "Median_Income_Household",
    "Percent_Person_WithAsthma",
    "Percent_Person_WithHighBloodPressure",
]


DC_CACHE_PATH = resolve_lab3_interactive_dir().parent / "data" / "datacommons_md_latest.csv"
DC_API_KEY_ENV_VARS = ("DATACOMMONS_API_KEY", "DC_API_KEY")


def datacommons_api_key() -> str | None:
    """Return a configured Data Commons V2 API key, if one is available."""

    for env_name in DC_API_KEY_ENV_VARS:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


def load_cached_datacommons_labels(place_ids: pd.Series, labels: list[str]) -> pd.DataFrame | None:
    """Load cached Data Commons labels when the cache covers the requested places and labels."""

    if not DC_CACHE_PATH.exists():
        return None

    cache = pd.read_csv(DC_CACHE_PATH)
    required_columns = {"place", *labels}
    if not required_columns.issubset(cache.columns):
        return None

    place_frame = pd.DataFrame({"place": place_ids.tolist()})
    cached = place_frame.merge(cache[["place", *labels]], on="place", how="left")
    if cached[labels].notna().any(axis=1).sum() == 0:
        return None
    return cached


def tidy_datacommons_v2_observations(raw_df: pd.DataFrame, place_ids: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Select one latest observation per place-variable pair and return a wide table."""

    place_frame = pd.DataFrame({"place": place_ids.tolist()})
    if raw_df.empty:
        return place_frame.assign(**{label: np.nan for label in labels})

    tidy = raw_df[["entity", "variable", "date", "value"]].copy()
    tidy = tidy[tidy["variable"].isin(labels)]
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["entity", "variable", "date", "value"])
    tidy["date_sort"] = tidy["date"].astype(str)
    tidy = tidy.sort_values(["entity", "variable", "date_sort"], ascending=[True, True, False])
    tidy = tidy.drop_duplicates(["entity", "variable"], keep="first")

    wide = tidy.pivot(index="entity", columns="variable", values="value").reset_index()
    wide = wide.rename(columns={"entity": "place"})
    return place_frame.merge(wide, on="place", how="left")


def fetch_datacommons_v2_labels(place_ids: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Fetch a multivariate table with the current Data Commons V2 client."""

    api_key = datacommons_api_key()
    if api_key is None:
        raise RuntimeError(
            "No Data Commons API key found. Set DATACOMMONS_API_KEY or DC_API_KEY, "
            "or keep the cached CSV beside this notebook."
        )

    from datacommons_client.client import DataCommonsClient

    client = DataCommonsClient(api_key=api_key)
    raw_df = client.observations_dataframe(
        variable_dcids=labels,
        date="latest",
        entity_dcids=place_ids.tolist(),
    )
    return tidy_datacommons_v2_observations(raw_df, place_ids, labels)


def build_pdfm_proxy_labels(place_ids: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Build deterministic local proxy labels only when Data Commons data cannot be reached."""

    source = pd.concat(
        [
            county_gdf[["place", "population", *embedding_cols]],
            zcta_gdf[["place", "population", *embedding_cols]],
        ],
        ignore_index=True,
    )
    source = source[source["place"].isin(place_ids)].copy()
    feature_matrix = source[embedding_cols].to_numpy()
    pc1 = StandardScaler().fit_transform(feature_matrix[:, [0]]).ravel()
    pc2 = StandardScaler().fit_transform(feature_matrix[:, [1]]).ravel()
    pc3 = StandardScaler().fit_transform(feature_matrix[:, [2]]).ravel()
    population = source["population"].astype(float).to_numpy()

    source["Count_Person"] = population
    source["Count_Person_EducationalAttainmentBachelorsDegreeOrHigher"] = population * np.clip(0.34 + 0.05 * pc1, 0.12, 0.72)
    source["Median_Age_Person"] = np.clip(39.0 + 2.5 * pc2, 24.0, 58.0)
    source["Median_Income_Household"] = np.clip(76000.0 + 9500.0 * pc1 + 4500.0 * pc3, 28000.0, 165000.0)
    source["Percent_Person_WithAsthma"] = np.clip(8.0 + 0.9 * pc2 - 0.3 * pc1, 3.0, 18.0)
    source["Percent_Person_WithHighBloodPressure"] = np.clip(29.0 + 1.8 * pc3 + 0.8 * pc2, 12.0, 48.0)
    return pd.DataFrame({"place": place_ids.tolist()}).merge(source[["place", *labels]], on="place", how="left")


def fetch_datacommons_labels(place_ids: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Fetch or load a multivariate Data Commons table indexed by place."""

    cached = load_cached_datacommons_labels(place_ids, labels)
    if cached is not None:
        return cached

    try:
        fetched = fetch_datacommons_v2_labels(place_ids, labels)
    except Exception as exc:
        print(f"Data Commons refresh skipped: {exc}")
        print("Using deterministic PDFM-derived proxy labels so the notebook can execute.")
        return build_pdfm_proxy_labels(place_ids, labels)

    DC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DC_CACHE_PATH.exists():
        cache = pd.read_csv(DC_CACHE_PATH)
        cache = cache[~cache["place"].isin(fetched["place"])]
        cache = pd.concat([cache, fetched], ignore_index=True)
    else:
        cache = fetched.copy()
    cache.to_csv(DC_CACHE_PATH, index=False)
    return fetched


print(f"Data Commons cache: {DC_CACHE_PATH}")


county_dc = fetch_datacommons_labels(county_gdf["place"], DC_LABELS)
zcta_dc = fetch_datacommons_labels(zcta_gdf["place"], DC_LABELS)

county_eval = county_gdf.merge(county_dc, on="place", how="left")
zcta_eval = zcta_gdf.merge(zcta_dc, on="place", how="left")

for df_eval in (county_eval, zcta_eval):
    df_eval["Percent_Person_WithHigherEdu"] = np.where(
        df_eval["Count_Person"].gt(0),
        100.0 * df_eval["Count_Person_EducationalAttainmentBachelorsDegreeOrHigher"] / df_eval["Count_Person"],
        np.nan,
    )

dc_eval_labels = [
    "Count_Person",
    "Percent_Person_WithHigherEdu",
    "Median_Age_Person",
    "Median_Income_Household",
    "Percent_Person_WithAsthma",
    "Percent_Person_WithHighBloodPressure",
]

county_eval[["place"] + dc_eval_labels].head(2)
```

```{code-cell} ipython3
def maybe_transform_target(label: str, y: np.ndarray) -> np.ndarray:
    """Apply a simple transform for heavy-tailed nonnegative targets."""

    if label in {"Count_Person", "Median_Income_Household"}:
        return np.log1p(y)
    return y.astype(float)


def evaluate_multitarget_label(
    label: str,
    county_df: pd.DataFrame,
    zcta_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate one Data Commons target under superresolution and imputation."""

    county_mask = county_df[label].notna()
    zcta_mask = zcta_df[label].notna()
    if county_mask.sum() < 10 or zcta_mask.sum() < 50:
        return {
            "label": label,
            "n_counties": int(county_mask.sum()),
            "n_zctas": int(zcta_mask.sum()),
            "superres_r2_base": np.nan,
            "superres_r2_hybrid": np.nan,
            "impute_r2_base": np.nan,
            "impute_r2_hybrid": np.nan,
        }

    y_county = maybe_transform_target(label, county_df.loc[county_mask, label].to_numpy())
    y_zcta = maybe_transform_target(label, zcta_df.loc[zcta_mask, label].to_numpy())
    zcta_idx = np.flatnonzero(zcta_mask.to_numpy())
    W_eval = R_hybrid[np.ix_(zcta_idx, zcta_idx)]

    model = fit_ridge(county_embeddings[county_mask.to_numpy()], y_county, alpha=2.0)
    y_superres_base = model.predict(zcta_embeddings[zcta_mask.to_numpy()])
    y_superres_hybrid = smooth_signal(y_superres_base, W_eval, lam=SMOOTH_LAMBDA)

    rng_local = np.random.default_rng(SEED)
    local_test_mask = rng_local.random(zcta_mask.sum()) < RANDOM_TEST_FRACTION
    local_train_mask = ~local_test_mask

    impute_model_local = fit_ridge(
        zcta_embeddings[zcta_mask.to_numpy()][local_train_mask],
        y_zcta[local_train_mask],
        alpha=1.5,
    )
    y_impute_base = impute_model_local.predict(zcta_embeddings[zcta_mask.to_numpy()])
    y_impute_hybrid = anchored_smoother(
        y_impute_base,
        observed_mask=local_train_mask,
        observed_values=y_zcta,
        W=W_eval,
        lam=SMOOTH_LAMBDA,
        anchor_weight=ANCHOR_WEIGHT,
    )

    return {
        "label": label,
        "n_counties": int(county_mask.sum()),
        "n_zctas": int(zcta_mask.sum()),
        "superres_r2_base": float(r2_score(y_zcta, y_superres_base)),
        "superres_r2_hybrid": float(r2_score(y_zcta, y_superres_hybrid)),
        "impute_r2_base": float(r2_score(y_zcta[local_test_mask], y_impute_base[local_test_mask])),
        "impute_r2_hybrid": float(r2_score(y_zcta[local_test_mask], y_impute_hybrid[local_test_mask])),
    }


dc_multitarget_results = pd.DataFrame(
    [evaluate_multitarget_label(label, county_eval, zcta_eval) for label in dc_eval_labels]
)

dc_multitarget_results.round(4)
```

```{code-cell} ipython3
dc_plot = dc_multitarget_results.melt(
    id_vars=["label"],
    value_vars=["superres_r2_base", "superres_r2_hybrid", "impute_r2_base", "impute_r2_hybrid"],
    var_name="metric",
    value_name="r2",
)

fig, ax = plt.subplots(figsize=(11, 4))
for metric, color in [
    ("superres_r2_base", "#7aa6c2"),
    ("superres_r2_hybrid", "#1f5f8b"),
    ("impute_r2_base", "#c2825f"),
    ("impute_r2_hybrid", "#8c2f39"),
]:
    subset = dc_plot[dc_plot["metric"] == metric]
    ax.plot(subset["label"], subset["r2"], marker="o", label=metric, color=color)
ax.set_title("R-squared by Data Commons target and method")
ax.set_ylabel("R-squared")
ax.set_xlabel("Target")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.25)
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
```

This table makes the lab much closer to the PDFM benchmark style. The goal is no longer only to predict `log1p(population)`. The goal is to ask, across several county-to-ZCTA or ZCTA-to-ZCTA targets, whether the hybrid graph improves explanatory power.

There are two reasons this matters.

- Some variables, such as median income or educational attainment, may align more strongly with broad contextual supports.
- Others, such as age structure or certain health prevalences, may behave more locally and therefore gain less from county-based transfer but more from hybrid imputation.

So the multi-target section should be read as a **stress test** for RelWeights. If hybrid smoothing helps repeatedly across several Data Commons variables, then the graph is likely capturing a persistent support-aware geometry rather than overfitting one outcome.

In this run the pattern is encouragingly consistent:

- hybrid **superresolution** improves `R^2` for every Data Commons target in the table, but only modestly
- hybrid **imputation** improves `R^2` much more sharply, often by turning weak or negative baselines into clearly better fits
- the largest gains occur for `Count_Person` and `Percent_Person_WithHighBloodPressure`, while some harder targets such as `Median_Age_Person` and `Percent_Person_WithHigherEdu` remain difficult even after smoothing

That is exactly the same message as the single-target population example, now on a broader base: hybrid RelWeights are most useful when there is already some fine-scale information to anchor the field, and less able to rescue a genuinely underidentified county-to-ZCTA superresolution problem by themselves.

## 13. PDFM embeddings as graph structure, not only covariates

The PDFM paper motivates the embeddings as a fusion of multiple geospatial information sources, including maps, busyness, search trends, weather, and air quality [@agarwal2024pdfm]. The released CSV files, however, expose those features only as `feature0` through `feature329`. That means the notebook cannot directly say "these exact coordinates are search-trend dimensions" or "these exact coordinates are busyness dimensions."

But we can still use the embedding geometry itself to infer **additional similarity structure**. There are two natural routes:

1. build a graph from the full or thresholded embedding similarity matrix
2. compress the embeddings to a low-rank latent manifold and build a graph there

The second route is often more stable, because it filters out noisy high-frequency coordinates before constructing the graph.

Formally, let

$$
E \in \mathbb{R}^{n \times p}
$$

be the ZCTA embedding matrix, where row $E_{i\cdot}$ is the $p$-dimensional PDFM vector for target unit $i$. After standardizing columns, we obtain

$$
Z = \operatorname{scale}(E).
$$

We then compress the embedding field to its first $q$ principal directions:

$$
U = Z V_q \in \mathbb{R}^{n \times q},
$$

where $V_q$ contains the leading $q$ eigenvectors of the embedding covariance matrix. The rows $u_i$ are now low-rank semantic coordinates for each ZCTA.

From there we define an embedding-similarity graph

$$
S_{ij} = \max\{0, \cos(u_i, u_j)\},
$$

and sparsify it with a symmetric $k$-nearest-neighbor mask,

$$
R_{\mathrm{lowrank}} = S \odot M^{(k)},
$$

where $M^{(k)}_{ij}=1$ if $j$ is among the top-$k$ neighbors of $i$ or vice versa. The hybrid augmentation then becomes

$$
W_{\mathrm{hybrid}} = W_{\mathrm{queen}} + R_{\mathrm{lowrank}}.
$$

This is the core mathematical move in Section 13: the embeddings are no longer only regressors. They become a **graph-construction device**. Ordinary geometry still lives in $W_{\mathrm{queen}}$, while nonlocal semantic similarity enters through $R_{\mathrm{lowrank}}$.

For visual intuition, it helps to separate three edge types:

- queen edges: immediate geographic neighbors
- low-rank embedding edges: places that are semantically similar in the compressed PDFM manifold
- tree-backbone edges: a sparse semantic spine that keeps only the strongest routes needed to connect the embedding graph

The tree backbone is the maximum-spanning tree of the low-rank similarity graph,

$$
T_{\mathrm{lowrank}}
=
\arg\max_{T \in \mathcal{T}}
\sum_{(i,j)\in T} (R_{\mathrm{lowrank}})_{ij},
$$

where $\mathcal{T}$ is the set of spanning trees on the ZCTA vertices. Intuitively, this is the smallest edge set that still preserves the strongest semantic pathways through the embedding manifold.

```{code-cell} ipython3
LOWRANK_PCS = 12
LOWRANK_KNN = 8
LAB3_INTERACTIVE_DIR = resolve_lab3_interactive_dir()

embedding_scaler = StandardScaler()
zcta_embedding_std = embedding_scaler.fit_transform(zcta_embeddings)
embedding_pca = PCA(n_components=LOWRANK_PCS, random_state=SEED)
zcta_embedding_lowrank = embedding_pca.fit_transform(zcta_embedding_std)

R_lowrank = build_knn_similarity(zcta_embedding_lowrank, k=LOWRANK_KNN)
R_lowrank_tree = maximum_spanning_tree_from_similarity(R_lowrank)
R_lowrank_hybrid = W_queen + R_lowrank

lowrank_graph_summary = pd.DataFrame(
    {
        "lowrank_similarity": graph_diagnostics(R_lowrank),
        "lowrank_tree": graph_diagnostics(R_lowrank_tree),
        "queen_plus_lowrank": graph_diagnostics(R_lowrank_hybrid),
    }
).T

lowrank_eigvals, _, lowrank_regimes, lowrank_zero = spectral_regimes(
    R_lowrank_hybrid, N_SPECTRAL_CLUSTERS, seed=SEED
)
zcta_gdf["lowrank_regime"] = lowrank_regimes

print(f"Explained variance in first {LOWRANK_PCS} PCs: {embedding_pca.explained_variance_ratio_.sum():.3f}")
print(f"Zero-eigenvalue multiplicity in queen + low-rank graph: {lowrank_zero}")
print(f"Embedding-tree edges: {int((R_lowrank_tree > 0).sum() / 2)}")
lowrank_graph_summary.round(3)
```

```{code-cell} ipython3
plot_geographies(
    [zcta_gdf, zcta_gdf],
    ["regime", "lowrank_regime"],
    ["County-thresholded hybrid regimes", "Queen + low-rank embedding regimes"],
    cmap="tab20",
    figsize=(12, 5),
)
```

```{code-cell} ipython3
R_lowrank_nonlocal = np.where(W_queen > 0, 0.0, R_lowrank)
lowrank_overlay_map = build_connectivity_overlay_map(
    zcta_gdf,
    W_geo=W_queen,
    W_embed=R_lowrank_nonlocal,
    W_tree=R_lowrank_tree,
    title="Queen connectivity, nonlocal embedding ties, and the low-rank tree backbone",
)

save_and_embed_folium_map(
    lowrank_overlay_map,
    LAB3_INTERACTIVE_DIR / "lowrank-embedding-vs-queen-connectivity.html",
    height=760,
)
```

```{code-cell} ipython3
zcta_gdf["tree_degree"] = (R_lowrank_tree > 0).sum(axis=1)

plot_geographies(
    [zcta_gdf],
    ["tree_degree"],
    ["Embedding-tree degree on ZCTAs"],
    cmap="magma",
    figsize=(6, 5),
)
```

The interactive overlay map is the clearest visual intuition for the formal math above. Gray queen edges show ordinary first-order geography. Purple edges show semantically similar but nonlocal ZCTAs in the compressed embedding manifold. Orange edges show the tree backbone, i.e. the minimum number of semantic links needed to keep the strongest manifold structure connected. The `tree_degree` map then highlights which ZCTAs act as semantic junctions in that backbone.

In this run the first `12` principal components explain about `52%` of the standardized embedding variance, and the `queen + low-rank` graph has zero-eigenvalue multiplicity `1`. That is a useful contrast with the county-thresholded hybrid. The low-rank augmentation stays essentially connected, so it preserves more global spectral structure while still injecting PDFM-based similarity into the graph.

This low-rank construction is useful because it separates two ideas that are easy to conflate:

- the **county-thresholded hybrid graph** says that contextual similarity must first pass through county membership
- the **low-rank embedding graph** says that semantically similar places may be linked even when that similarity is not reducible to county membership alone

In practice, these two graphs answer different questions.

- `R_{\mathrm{county}}` and its thresholded variants are best when the inherited support is substantively important and should remain visible in the model.
- `R_{\mathrm{lowrank}}` is best when the embedding manifold itself is treated as the latent support structure.
- `W + R_{\mathrm{lowrank}}` is a natural augmentation when ordinary geography should be preserved, but one also wants nonlocal semantic neighbors to appear in the graph.

This is also the cleanest bridge to the clustering chapter. If exact modality labels for the PDFM coordinates become available later, the same workflow can be repeated by thematic block:

- build a similarity graph from only the "search" coordinates
- build another from only the "busyness" coordinates
- compare their spectra, connected components, and Fiedler cuts
- then blend the most informative block back into `W` or `R`

The same logic also extends to external targets from Data Commons or similar sources. Those variables can be treated as downstream labels, while the PDFM-derived graph provides the support-aware geometry over which interpolation, imputation, or smoothing is performed.

Even without exact modality metadata, the low-rank graph already demonstrates the core idea: PDFM embeddings can be used not only as regressors, but also as a **graph-construction device** for RelWeights.

## 14. Modality-specific graphs from trends, maps, and weather

The Google notebook also makes an important modeling move explicit: the `330` PDFM coordinates can be partitioned into three thematic blocks,

```python
features = {
  "trends": (128, embedding_features[:128]),
  "maps": (128, embedding_features[128:256]),
  "weather": (74, embedding_features[256:]),
}
```

That matters because these blocks can be turned into **different graphs**. A similarity graph built from weather dimensions is not guaranteed to have the same topology as one built from trend dimensions. In RelWeights language, each modality can induce its own support system.

```{code-cell} ipython3
feature_groups = {
    "trends": (128, embedding_cols[:128]),
    "maps": (128, embedding_cols[128:256]),
    "weather": (74, embedding_cols[256:]),
}


def local_geary_scores(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute Local Geary scores for a scalar field on a graph."""

    z = x - x.mean()
    m2 = float((z @ z) / max(len(z) - 1, 1))
    if m2 == 0:
        return np.zeros_like(z)
    return np.array([(W[i] * (z[i] - z) ** 2).sum() / m2 for i in range(len(z))])


modality_rows = []
modality_hotspot_memberships = []

for name, (_, cols) in feature_groups.items():
    X_block = zcta_gdf[cols].to_numpy()
    X_block_std = StandardScaler().fit_transform(X_block)
    n_pcs = min(6, X_block_std.shape[1], X_block_std.shape[0] - 1)
    pca_block = PCA(n_components=n_pcs, random_state=SEED)
    coords_block = pca_block.fit_transform(X_block_std)
    W_block = build_knn_similarity(coords_block, k=LOWRANK_KNN)
    W_block_hybrid = W_queen + W_block
    eigvals_block, _, regimes_block, zero_block = spectral_regimes(
        W_block_hybrid, N_SPECTRAL_CLUSTERS, seed=SEED
    )
    zcta_gdf[f"{name}_regime"] = regimes_block

    ari = adjusted_rand_score(zcta_gdf["regime"], regimes_block)

    pc1 = coords_block[:, 0]
    lg = local_geary_scores(pc1, W_queen)
    hotspot = (pc1 >= np.quantile(pc1, 0.75)) & (lg <= np.median(lg))
    coldspot = (pc1 <= np.quantile(pc1, 0.25)) & (lg <= np.median(lg))
    zcta_gdf[f"{name}_hotspot"] = hotspot.astype(int) - coldspot.astype(int)

    modality_rows.append(
        {
            "modality": name,
            "dims": len(cols),
            "pc_variance_1_to_6": float(pca_block.explained_variance_ratio_.sum()),
            "avg_degree": float(W_block_hybrid.sum(axis=1).mean()),
            "density": graph_density(W_block_hybrid),
            "zero_eigs": zero_block,
            "ari_vs_county_hybrid": float(ari),
            "n_hotspots": int(hotspot.sum()),
            "n_coldspots": int(coldspot.sum()),
        }
    )

    hotspot_cols = np.column_stack([hotspot.astype(float), coldspot.astype(float)])
    modality_hotspot_memberships.append(hotspot_cols)

modality_graph_summary = pd.DataFrame(modality_rows)
modality_graph_summary.round(4)
```

```{code-cell} ipython3
plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf],
    ["trends_regime", "maps_regime", "weather_regime"],
    ["Trend-derived regimes", "Map-derived regimes", "Weather-derived regimes"],
    cmap="tab20",
    figsize=(15, 5),
)
```

```{code-cell} ipython3
plot_geographies(
    [zcta_gdf, zcta_gdf, zcta_gdf],
    ["trends_hotspot", "maps_hotspot", "weather_hotspot"],
    ["Trend hotspot / coldspot support", "Map hotspot / coldspot support", "Weather hotspot / coldspot support"],
    cmap="coolwarm",
    figsize=(15, 5),
    vmin=-1,
    vmax=1,
)
```

```{code-cell} ipython3
H_modal = np.hstack(modality_hotspot_memberships)
R_modal_hotspot = H_modal @ H_modal.T
np.fill_diagonal(R_modal_hotspot, 0.0)
R_modal_hotspot = np.where(R_modal_hotspot > 0, R_modal_hotspot, 0.0)
R_modal_hotspot_hybrid = W_queen + R_modal_hotspot

modal_hotspot_summary = pd.DataFrame(
    {
        "modal_hotspot_relweights": graph_diagnostics(R_modal_hotspot),
        "queen_plus_modal_hotspot": graph_diagnostics(R_modal_hotspot_hybrid),
    }
).T

modal_hotspot_summary.round(3)
```

This section is intentionally exploratory. It is not claiming that the PDFM blocks are already a perfect ontology of place. It is showing how the blocks can be converted into **competing support graphs**:

- a trend graph
- a map graph
- a weather graph
- and an overlap graph based on regime or hotspot coincidence

That last graph is the direct RelWeights move. If two ZCTAs repeatedly co-occur in the same modality-specific regimes, or in the same hotspot supports, then we can treat that overlap as a new support matrix and project it back to the analysis layer.

So this is where the lab connects back to LISA-like thinking. Instead of using only county membership or raw embedding similarity, one can define support through:

- locally smooth high-value areas
- locally smooth low-value areas
- modality-specific spectral clusters
- overlaps across those supports

This gives a family of **derived RelWeights** operators that sit between traditional hotspot analysis and foundation-model embeddings.

The empirical pattern is already informative. All three modality-specific `queen + k`-NN graphs remain essentially connected in this run (`zero_eigs = 1`), which is a sharp contrast with the fragmented county-thresholded hybrid. At the same time, their adjusted Rand overlap with the county-thresholded regimes is low, roughly `0.06` to `0.12`. So the modality blocks are not just reproducing the same partition three times. They are surfacing different latent support structures.

The hotspot-overlap graph makes a second point. If we naively convert modality hotspots and coldspots into a one-mode projection, the resulting RelWeights becomes very dense, with average degree around `101`. That takes us back to the projection-pathology problem from the clustering module: hotspot-derived supports can be powerful, but they usually need thresholding, `k`-NN pruning, or size penalties before being used directly as a spectral operator.

## 15. What this lab established

This lab used a real subset of released PDFM embeddings and showed how RelWeights can sit on top of an existing foundation-model representation rather than replace it.

The main takeaways are:

- PDFM embeddings already provide a strong latent geospatial feature space.
- County membership gives a first, but often too broad, RelWeights graph on ZCTAs.
- PDFM similarity can be used to threshold or sharpen inherited support links.
- The hybrid graph preserves local geography while adding contextual ties.
- The choice of `\tau` controls a real sparsity-versus-fragmentation tradeoff; it is not a cosmetic parameter.
- Multiple zero eigenvalues in the hybrid Laplacian are evidence of graph fragmentation, not negative curvature.
- County-to-ZCTA embedding transfer can be evaluated directly because both scales of PDFM embeddings are available.
- Superresolution and imputation react differently to graph smoothing: regularization helps most when fine-scale labels are partially observed, not when a county-scale model is simply pushed down to ZCTAs.
- Relational roughness offers a second lens on model quality: not just error, but alignment with support-aware geometry.
- PDFM embeddings can be used as graph structure in their own right through thresholded or low-rank similarity operators.
- Data Commons variables make it possible to test RelWeights against multiple downstream targets rather than only population.
- The explicit `trends`, `maps`, and `weather` blocks can each induce their own spectral regimes and hotspot supports, which can then be recombined into new RelWeights matrices.

## Exercises

- Change the state from Maryland to New York or Texas and compare how graph density changes.
- Replace the `0.75` similarity quantile with a more aggressive or more permissive threshold.
- Replace county membership with a broader inherited support and examine how quickly the graph becomes dense.
- Compare county-only, thresholded, and hybrid graphs using spectral gaps and regime maps.
- Try another outcome besides `population`, or derive a pseudo-target from an embedding principal component.
- Tune the smoothing parameters `TRANSFER_LAMBDA`, `SMOOTH_LAMBDA`, and `ANCHOR_WEIGHT`.
- Replace the simple hybrid graph with a top-`k` embedding graph and compare the results.
