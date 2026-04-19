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

# Lab 1: Building RelWeights

This lab turns the matrix story in `Defining RelWeights` into an executable overlay workflow. The goal is to start from two polygon layers, build an incidence matrix $B$, and then construct the contextual similarity graph

$$
R = B B^{\top} - \operatorname{diag}(B B^{\top}).
$$

The core construction below follows the same logic as the public RelWeights repository and, in particular, the `tab2relweights` workflow implemented in [`relweights.py`](https://github.com/shamilkhedgikar/RelWeights/blob/main/relweights.py). The sample overlay data used here is a small packaged subset derived from the local `RelWeights` data folder so that the notebook can run quickly end to end.

## Learning goals

By the end of this lab, you should be able to:

- explain the difference between an incidence matrix $B$, a contextual similarity matrix $R$, and a Laplacian $L_R$
- build binary and area-share incidence matrices from two overlaid polygon layers
- see how the same districts can be connected by inherited supports even when they are not immediate geographic neighbors
- compute a relational Laplacian and evaluate a simple Laplacian energy for a demonstration signal

## Imports and configuration

```{code-cell} ipython3
from __future__ import annotations

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.lines import Line2D
from shapely.geometry import LineString
```

```{code-cell} ipython3
AREA_CRS = "EPSG:32644"
GEOGRAPHIC_CONTIGUITY = "queen"
SAMPLE_ZIP_CANDIDATES = [
    Path("data/lab1_relweights_sample.zip"),
    Path("lab/book/content/02-lab-1-building-relweights/data/lab1_relweights_sample.zip"),
]

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 120
```

## Helper functions

```{code-cell} ipython3
def resolve_sample_zip(candidates: list[Path]) -> Path:
    """Return the first existing sample zip path."""

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find lab1_relweights_sample.zip. "
        "Checked: " + ", ".join(str(p) for p in candidates)
    )


def interactive_output_dir_from_zip(zip_path: Path) -> Path:
    """Place standalone map HTML next to the notebook content folder."""

    output_dir = zip_path.parent.parent / "interactive"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_packaged_layers(zip_path: Path) -> tuple[TemporaryDirectory, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Extract the packaged sample zip and return the two GeoDataFrames."""

    tmpdir = TemporaryDirectory()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir.name)

    districts = gpd.read_file(Path(tmpdir.name) / "districts.geojson")
    supports = gpd.read_file(Path(tmpdir.name) / "subbasins.geojson")

    districts = districts[~districts.geometry.isna() & ~districts.geometry.is_empty].copy()
    supports = supports[~supports.geometry.isna() & ~supports.geometry.is_empty].copy()

    return tmpdir, districts, supports


def build_incidence_matrices(
    rel_layer: gpd.GeoDataFrame,
    inh_layer: gpd.GeoDataFrame,
    rel_id: str,
    inh_id: str,
    area_crs: str = AREA_CRS,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Construct binary and area-share incidence matrices from overlay intersections.

    The binary matrix records whether an overlap exists.
    The area-share matrix records what share of each analysis polygon is covered
    by each inherited support polygon.
    """

    rel = rel_layer[[rel_id, "geometry"]].copy()
    inh = inh_layer[[inh_id, "geometry"]].copy().to_crs(rel.crs)

    overlay = gpd.overlay(rel, inh, how="intersection", keep_geom_type=False)
    overlay_projected = overlay.to_crs(area_crs)
    rel_projected = rel.to_crs(area_crs)

    rel_area = rel_projected.set_index(rel_id).geometry.area
    overlay["overlap_area"] = overlay_projected.geometry.area
    overlay["area_share"] = overlay["overlap_area"] / overlay[rel_id].map(rel_area)

    binary = pd.crosstab(overlay[rel_id], overlay[inh_id])
    binary = binary.reindex(index=rel_layer[rel_id], columns=inh_layer[inh_id], fill_value=0)
    binary = (binary > 0).astype(int)

    area = (
        overlay.pivot_table(
            index=rel_id,
            columns=inh_id,
            values="area_share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=rel_layer[rel_id], columns=inh_layer[inh_id], fill_value=0.0)
        .astype(float)
    )

    return binary, area, overlay


def relweights_from_incidence(incidence: pd.DataFrame, row_standardize: bool = False) -> pd.DataFrame:
    """Compute R = B B' with a zero diagonal from an incidence matrix."""

    B = incidence.to_numpy(dtype=float)
    R = B @ B.T
    np.fill_diagonal(R, 0.0)

    if row_standardize:
        row_sums = R.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        R = R / row_sums

    return pd.DataFrame(R, index=incidence.index, columns=incidence.index)


def laplacian_from_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Construct D - W from a square weight matrix."""

    W = weights_df.to_numpy(dtype=float)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return pd.DataFrame(L, index=weights_df.index, columns=weights_df.columns)


def build_geographic_contiguity_weights(
    layer: gpd.GeoDataFrame,
    id_col: str,
    mode: str = GEOGRAPHIC_CONTIGUITY,
) -> pd.DataFrame:
    """Construct a binary queen- or rook-contiguity matrix on the analysis layer."""

    mode = mode.lower()
    if mode not in {"queen", "rook"}:
        raise ValueError("mode must be either 'queen' or 'rook'")

    ids = layer[id_col].tolist()
    n = len(layer)
    W = np.zeros((n, n), dtype=float)
    geoms = layer.geometry.reset_index(drop=True)

    for i in range(n):
        for j in range(i + 1, n):
            shared_boundary = geoms.iloc[i].boundary.intersection(geoms.iloc[j].boundary)
            is_neighbor = (not shared_boundary.is_empty) and (
                mode == "queen" or shared_boundary.length > 0
            )
            if is_neighbor:
                W[i, j] = 1.0
                W[j, i] = 1.0

    return pd.DataFrame(W, index=ids, columns=ids)


def build_centroid_nodes(
    layer: gpd.GeoDataFrame,
    id_col: str,
    label_cols: list[str] | None = None,
    area_crs: str = AREA_CRS,
) -> gpd.GeoDataFrame:
    """Return centroid nodes in the original layer CRS for graph-style plotting."""

    label_cols = label_cols or []
    keep_cols = list(dict.fromkeys([id_col, *label_cols, "geometry"]))
    projected = layer[keep_cols].copy().to_crs(area_crs)
    centroids = gpd.GeoSeries(projected.geometry.centroid, crs=area_crs)
    nodes = projected.drop(columns="geometry").copy()
    nodes = gpd.GeoDataFrame(nodes, geometry=centroids, crs=area_crs)
    return nodes.to_crs(layer.crs)


def build_centroid_edges(
    layer: gpd.GeoDataFrame,
    id_col: str,
    weights_df: pd.DataFrame,
    edge_type: str,
    area_crs: str = AREA_CRS,
) -> gpd.GeoDataFrame:
    """Connect centroid pairs wherever the weight matrix has a positive entry."""

    projected = layer[[id_col, "geometry"]].copy().to_crs(area_crs)
    centroids = gpd.GeoSeries(projected.geometry.centroid, crs=area_crs)
    centroid_lookup = dict(zip(projected[id_col], centroids))
    edge_records: list[dict[str, object]] = []

    for i, source_id in enumerate(weights_df.index):
        for j in range(i + 1, len(weights_df.columns)):
            target_id = weights_df.columns[j]
            weight = float(weights_df.iat[i, j])
            if weight <= 0:
                continue
            edge_records.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "weight": weight,
                    "edge_type": edge_type,
                    "geometry": LineString([centroid_lookup[source_id], centroid_lookup[target_id]]),
                }
            )

    if not edge_records:
        empty_edges = gpd.GeoDataFrame(
            {"source_id": [], "target_id": [], "weight": [], "edge_type": []},
            geometry=gpd.GeoSeries([], crs=layer.crs),
            crs=layer.crs,
        )
        return empty_edges

    edges = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=area_crs)
    return edges.to_crs(layer.crs)


def build_neighbor_lookup(weights_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return neighbor ids for every district under a binary support graph."""

    binary = (weights_df > 0).astype(int)
    return {
        str(source_id): [str(target_id) for target_id, value in row.items() if value > 0]
        for source_id, row in binary.iterrows()
    }


def build_hover_edge_geojson(
    centroid_nodes: gpd.GeoDataFrame,
    id_col: str,
    weights_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Precompute hover-time centroid edges for each focal district."""

    centroid_lookup = {
        str(row[id_col]): row.geometry
        for _, row in centroid_nodes[[id_col, "geometry"]].iterrows()
    }
    edge_lookup: dict[str, dict[str, object]] = {}

    for source_id in weights_df.index:
        source_key = str(source_id)
        features: list[dict[str, object]] = []
        for target_id, value in weights_df.loc[source_id].items():
            if float(value) <= 0:
                continue
            target_key = str(target_id)
            line = LineString([centroid_lookup[source_key], centroid_lookup[target_key]])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "source_id": source_key,
                        "target_id": target_key,
                        "weight": float(value),
                    },
                    "geometry": line.__geo_interface__,
                }
            )
        edge_lookup[source_key] = {"type": "FeatureCollection", "features": features}

    return edge_lookup


def add_centroid_markers(
    map_obj: folium.Map,
    centroid_nodes: gpd.GeoDataFrame,
    id_col: str,
    label_col: str,
    layer_name: str = "Centroid nodes",
) -> folium.FeatureGroup:
    """Add centroid markers as a toggleable map layer."""

    group = folium.FeatureGroup(name=layer_name, show=True)
    for _, row in centroid_nodes.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color="#111827",
            fill=True,
            fill_color="#111827",
            fill_opacity=0.9,
            weight=1,
            tooltip=f"{row[id_col]}: {row[label_col]}",
        ).add_to(group)
    group.add_to(map_obj)
    return group


def add_neighbor_hover_behavior(
    map_obj: folium.Map,
    geojson_layer: folium.GeoJson,
    id_property: str,
    neighbor_map: dict[str, list[str]],
    edge_map: dict[str, dict[str, object]],
    base_style: dict[str, object],
    hover_style: dict[str, object],
    neighbor_style: dict[str, object],
    edge_style: dict[str, object],
) -> None:
    """Highlight hovered districts, their neighbors, and the corresponding centroid edges."""

    script = f"""
    setTimeout(function() {{
        var neighborMap = {json.dumps(neighbor_map)};
        var edgeMap = {json.dumps(edge_map)};
        var baseStyle = {json.dumps(base_style)};
        var hoverStyle = {json.dumps(hover_style)};
        var neighborStyle = {json.dumps(neighbor_style)};
        var edgeStyle = {json.dumps(edge_style)};
        var mapObject = {map_obj.get_name()};
        var geoLayer = {geojson_layer.get_name()};
        var edgeLayer = L.geoJSON(null, {{
            style: function() {{
                return edgeStyle;
            }}
        }}).addTo(mapObject);
        var layerIndex = {{}};

        function resetNeighborExplorer() {{
            edgeLayer.clearLayers();
            geoLayer.eachLayer(function(otherLayer) {{
                otherLayer.setStyle(baseStyle);
            }});
        }}

        geoLayer.eachLayer(function(layer) {{
            var featureId = String(layer.feature.properties["{id_property}"]);
            layerIndex[featureId] = layer;
            layer.on({{
                mouseover: function() {{
                    resetNeighborExplorer();
                    layer.setStyle(hoverStyle);
                    (neighborMap[featureId] || []).forEach(function(neighborId) {{
                        if (layerIndex[neighborId]) {{
                            layerIndex[neighborId].setStyle(neighborStyle);
                            if (layerIndex[neighborId].bringToFront) {{
                                layerIndex[neighborId].bringToFront();
                            }}
                        }}
                    }});
                    if (edgeMap[featureId]) {{
                        edgeLayer.addData(edgeMap[featureId]);
                    }}
                    if (layer.bringToFront) {{
                        layer.bringToFront();
                    }}
                }},
                mouseout: function() {{
                    resetNeighborExplorer();
                }}
            }});
        }});
        resetNeighborExplorer();
    }}, 0);
    """

    map_obj.get_root().script.add_child(folium.Element(script))


def make_connectivity_map(
    districts: gpd.GeoDataFrame,
    centroid_nodes: gpd.GeoDataFrame,
    geographic_edges: gpd.GeoDataFrame,
    rel_only_edges: gpd.GeoDataFrame,
    id_col: str,
    label_col: str,
    geographic_label: str,
) -> folium.Map:
    """Create a layer-toggle map for full queen/rook and RelWeights-only graphs."""

    study_centroid = districts.geometry.union_all().centroid
    center = [
        float(study_centroid.y),
        float(study_centroid.x),
    ]
    graph_map = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    district_layer = folium.FeatureGroup(name="District polygons", show=True)
    folium.GeoJson(
        districts[["ac_id", "AC_NAME", "DIST_NAME", "geometry"]].to_json(),
        name="District polygons",
        style_function=lambda _feature: {
            "color": "#9ca3af",
            "weight": 1.0,
            "fillColor": "#e5e7eb",
            "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(fields=["ac_id", "AC_NAME", "DIST_NAME"]),
    ).add_to(district_layer)
    district_layer.add_to(graph_map)

    add_centroid_markers(graph_map, centroid_nodes, id_col=id_col, label_col=label_col)

    if not geographic_edges.empty:
        folium.GeoJson(
            geographic_edges.to_json(),
            name=f"{geographic_label} edges",
            style_function=lambda _feature: {
                "color": "#475569",
                "weight": 2.0,
                "opacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["source_id", "target_id"],
                aliases=["Source", "Target"],
            ),
        ).add_to(graph_map)

    if not rel_only_edges.empty:
        folium.GeoJson(
            rel_only_edges.to_json(),
            name="RelWeights-only inherited ties",
            style_function=lambda _feature: {
                "color": "#ea580c",
                "weight": 2.6,
                "opacity": 0.8,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["source_id", "target_id", "weight"],
                aliases=["Source", "Target", "Shared supports"],
            ),
        ).add_to(graph_map)

    folium.LayerControl(collapsed=False).add_to(graph_map)
    return graph_map


def make_hover_neighbor_map(
    districts: gpd.GeoDataFrame,
    centroid_nodes: gpd.GeoDataFrame,
    weights_df: pd.DataFrame,
    id_col: str,
    label_col: str,
    title: str,
    neighbor_color: str,
    edge_color: str,
) -> folium.Map:
    """Create a hover-based neighborhood explorer for one support graph."""

    study_centroid = districts.geometry.union_all().centroid
    center = [
        float(study_centroid.y),
        float(study_centroid.x),
    ]
    explorer = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    base_style = {
        "color": "#6b7280",
        "weight": 1.0,
        "fillColor": "#f3f4f6",
        "fillOpacity": 0.45,
    }
    hover_style = {
        "color": "#111827",
        "weight": 2.5,
        "fillColor": "#111827",
        "fillOpacity": 0.2,
    }
    neighbor_style = {
        "color": neighbor_color,
        "weight": 2.2,
        "fillColor": neighbor_color,
        "fillOpacity": 0.35,
    }

    geojson_layer = folium.GeoJson(
        districts[[id_col, "AC_NAME", "DIST_NAME", "geometry"]].to_json(),
        name=title,
        style_function=lambda _feature: base_style.copy(),
        tooltip=folium.GeoJsonTooltip(fields=[id_col, "AC_NAME", "DIST_NAME"]),
    )
    geojson_layer.add_to(explorer)
    add_centroid_markers(explorer, centroid_nodes, id_col=id_col, label_col=label_col)

    neighbor_map = build_neighbor_lookup(weights_df)
    edge_map = build_hover_edge_geojson(centroid_nodes, id_col=id_col, weights_df=weights_df)
    add_neighbor_hover_behavior(
        map_obj=explorer,
        geojson_layer=geojson_layer,
        id_property=id_col,
        neighbor_map=neighbor_map,
        edge_map=edge_map,
        base_style=base_style,
        hover_style=hover_style,
        neighbor_style=neighbor_style,
        edge_style={
            "color": edge_color,
            "weight": 3.0,
            "opacity": 0.9,
        },
    )

    instructions = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 12px;
        line-height: 1.35;
    ">
        <strong>{title}</strong><br>
        Hover on one district to reveal its first-order neighbors and centroid edges.
    </div>
    """
    explorer.get_root().html.add_child(folium.Element(instructions))
    return explorer


def save_and_embed_folium_map(
    map_obj: folium.Map,
    output_path: Path,
    height: int = 620,
) -> None:
    """Save a Folium map to standalone HTML and embed it back without file:// iframes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    rel_path = output_path.as_posix()
    data_uri = "data:text/html;base64," + base64.b64encode(output_path.read_bytes()).decode("ascii")
    display(
        HTML(
            f"""
            <div style="margin: 0.5rem 0 1rem 0;">
              <iframe
                src="{data_uri}"
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


def centered_laplacian_energy(values: np.ndarray, laplacian_df: pd.DataFrame) -> float:
    """Return z' L z for the centered signal z."""

    z = values - values.mean()
    L = laplacian_df.to_numpy(dtype=float)
    return float(z.T @ L @ z)


def plot_layer(
    layer: gpd.GeoDataFrame,
    color_column: str,
    title: str,
    cmap: str = "tab20",
    focus_bounds: np.ndarray | None = None,
) -> None:
    """Plot a single polygon layer with optional fixed plotting bounds."""

    ax = layer.plot(
        column=color_column,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )
    if focus_bounds is not None:
        minx, miny, maxx, maxy = focus_bounds
        padx = 0.05 * (maxx - minx)
        pady = 0.05 * (maxy - miny)
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()
```

## Load the packaged sample

```{code-cell} ipython3
zip_path = resolve_sample_zip(SAMPLE_ZIP_CANDIDATES)
interactive_dir = interactive_output_dir_from_zip(zip_path)
_tmpdir, districts, supports = load_packaged_layers(zip_path)
study_bounds = districts.total_bounds

print(f"Sample zip: {zip_path}")
print(f"Interactive HTML output folder: {interactive_dir}")
print(f"Districts: {len(districts)}")
print(f"Inherited supports: {len(supports)}")
print(f"District CRS: {districts.crs}")
print(f"Support CRS: {supports.crs}")
```

The packaged sample is intentionally small:

- the analysis layer contains 50 assembly constituencies
- the inherited layer contains 7 intersecting sub-basins clipped to a local study window

This is enough to illustrate RelWeights without pulling the full India-wide files into the lab runtime.

## Visualize the two layers

```{code-cell} ipython3
plot_layer(
    districts,
    "DIST_NAME",
    "Analysis layer: East Godavari, West Godavari, and Krishna constituencies",
    cmap="tab20b",
    focus_bounds=study_bounds,
)
```

```{code-cell} ipython3
plot_layer(
    supports,
    "support_id",
    "Inherited support layer: intersecting sub-basins",
    cmap="Set3",
    focus_bounds=study_bounds,
)
```

```{code-cell} ipython3
ax = districts.plot(facecolor="none", edgecolor="firebrick", linewidth=0.8)
supports.plot(ax=ax, column="support_id", cmap="Set3", alpha=0.35, edgecolor="navy", linewidth=1.0)
minx, miny, maxx, maxy = study_bounds
padx = 0.05 * (maxx - minx)
pady = 0.05 * (maxy - miny)
ax.set_xlim(minx - padx, maxx + padx)
ax.set_ylim(miny - pady, maxy + pady)
ax.set_title("Overlay view: districts inherit support from multiple sub-basins")
ax.set_axis_off()
plt.show()
```

The overlay view is the geometric foundation of the operator. Two districts can become related in RelWeights if they intersect the same support polygon, even if a simple same-layer contiguity rule would not connect them strongly.

```{code-cell} ipython3
interactive_map = districts.explore(
    column="DIST_NAME",
    cmap="tab20b",
    tooltip=["ac_id", "AC_NAME", "DIST_NAME"],
    popup=["ac_id", "AC_NAME", "DIST_NAME"],
    name="Districts",
    tiles="CartoDB positron",
    legend=False,
    show=True,
    style_kwds={"weight": 1.0, "fillOpacity": 0.45},
)

supports.explore(
    m=interactive_map,
    column="support_id",
    cmap="Set3",
    tooltip=["support_id", "pfaf_id"],
    popup=["support_id", "pfaf_id", "AreaSQKM_B"],
    name="Inherited supports",
    legend=False,
    show=True,
    style_kwds={"weight": 1.2, "fillOpacity": 0.25},
)

folium.LayerControl(collapsed=False).add_to(interactive_map)

save_and_embed_folium_map(
    interactive_map,
    interactive_dir / "overlay-support-map.html",
    height=560,
)
```

The interactive `explore()` map is now saved as a standalone HTML artifact before being embedded back into the notebook. That is more reliable than relying on the notebook renderer to interpret Folium objects directly, especially inside VS Code and MyST-derived notebook flows. The layer control still lets you toggle districts and inherited supports on and off, and the automatic legends are intentionally suppressed here because two categorical legends would overlap and clutter the map.

## Build the incidence matrices

```{code-cell} ipython3
B_bin, B_area, overlay = build_incidence_matrices(
    rel_layer=districts,
    inh_layer=supports,
    rel_id="ac_id",
    inh_id="support_id",
)

print("Binary incidence matrix shape:", B_bin.shape)
print("Area-share incidence matrix shape:", B_area.shape)
print()
print("First five rows of B_bin")
print(B_bin.head().to_string())
print()
print("First five rows of B_area")
print(B_area.head().round(3).to_string())
```

The two matrices encode different modeling choices:

- `B_bin` keeps only support presence/absence
- `B_area` records how much of each district is carried by each support polygon

The public `relweights.py` script uses the binary form by default when it turns positive intersections into a weight object. Here we inspect both versions before committing to one.

```{code-cell} ipython3
row_share_sums = B_area.sum(axis=1)
print("Area-share row sums (should be close to 1 when supports partition the district extent they cover):")
print(row_share_sums.round(4).head(10).to_string())
```

## Construct the RelWeights matrix

```{code-cell} ipython3
R_binary = relweights_from_incidence(B_bin, row_standardize=False)
R_binary_row = relweights_from_incidence(B_bin, row_standardize=True)

print("Top-left block of raw R from binary incidence")
print(R_binary.iloc[:8, :8].to_string())
print()
print("Top-left block of row-standardized R")
print(R_binary_row.iloc[:8, :8].round(3).to_string())
```

This is the operational RelWeights step:

1. start from a cross-layer incidence matrix $B$
2. compute $B B^{\top}$ to measure shared inherited support
3. zero out the diagonal so that self-similarity does not dominate the graph

If two districts share more sub-basins, the corresponding entry of $R$ is larger.

## Compare contextual ties with geographic contiguity

```{code-cell} ipython3
GEOGRAPHIC_LABEL = GEOGRAPHIC_CONTIGUITY.title()
W_geo = build_geographic_contiguity_weights(
    districts,
    "ac_id",
    mode=GEOGRAPHIC_CONTIGUITY,
)
geo_column_name = f"{GEOGRAPHIC_CONTIGUITY}_degree"

print(f"Geographic contiguity baseline: {GEOGRAPHIC_LABEL}")

degree_compare = pd.DataFrame(
    {
        geo_column_name: W_geo.sum(axis=1),
        "rel_degree_raw": R_binary.sum(axis=1),
        "rel_degree_binary": (R_binary > 0).sum(axis=1),
    }
)

print(degree_compare.describe().round(2).to_string())
```

```{code-cell} ipython3
example_id = degree_compare["rel_degree_raw"].sort_values(ascending=False).index[0]
example_neighbors = pd.DataFrame(
    {
        f"{GEOGRAPHIC_CONTIGUITY}_contiguity": W_geo.loc[example_id],
        "relweight_raw": R_binary.loc[example_id],
        "relweight_rowstd": R_binary_row.loc[example_id],
    }
)
example_neighbors = example_neighbors.query(f"{GEOGRAPHIC_CONTIGUITY}_contiguity > 0 or relweight_raw > 0")
print(f"Neighborhood structure for {example_id}")
print(example_neighbors.sort_values('relweight_raw', ascending=False).round(3).to_string())
```

The notebook now defaults to **queen contiguity**, but the helper is explicit about the modeling choice. If you set `GEOGRAPHIC_CONTIGUITY = "rook"`, the same code reruns with shared-edge adjacency instead of shared-corner-or-edge adjacency. That makes the geographic baseline comparable across both common first-order contiguity conventions.

This comparison is the main conceptual payoff. A geographic weight matrix asks, "who is contiguous with me under the chosen same-layer rule?" A RelWeights matrix asks, "who inherits the same supports I inherit?" The two overlap, but they are not the same graph.

## Visualize the geographic graph and additional inherited ties

```{code-cell} ipython3
district_nodes = build_centroid_nodes(
    districts,
    id_col="ac_id",
    label_cols=["AC_NAME"],
)
R_indicator = (R_binary > 0).astype(float)
rel_only_indicator = ((R_indicator > 0) & (W_geo == 0)).astype(float)

geo_edges = build_centroid_edges(
    districts,
    id_col="ac_id",
    weights_df=W_geo,
    edge_type=f"{GEOGRAPHIC_CONTIGUITY}_contiguity",
)
rel_only_edges = build_centroid_edges(
    districts,
    id_col="ac_id",
    weights_df=rel_only_indicator,
    edge_type="relweights_only",
)

geo_edge_count = int(np.triu(W_geo.to_numpy(dtype=int), 1).sum())
rel_edge_count = int(np.triu(R_indicator.to_numpy(dtype=int), 1).sum())
rel_only_edge_count = int(np.triu(rel_only_indicator.to_numpy(dtype=int), 1).sum())

print(f"{GEOGRAPHIC_LABEL} edges: {geo_edge_count}")
print(f"RelWeights edges: {rel_edge_count}")
print(f"RelWeights-only inherited ties: {rel_only_edge_count}")
```

```{code-cell} ipython3
ax = districts.plot(facecolor="white", edgecolor="#d1d5db", linewidth=0.8)
if not geo_edges.empty:
    geo_edges.plot(ax=ax, color="#64748b", linewidth=1.1, alpha=0.5)
if not rel_only_edges.empty:
    rel_only_edges.plot(ax=ax, color="#ea580c", linewidth=1.8, alpha=0.75)
district_nodes.plot(ax=ax, color="#111827", markersize=14, zorder=3)

minx, miny, maxx, maxy = study_bounds
padx = 0.05 * (maxx - minx)
pady = 0.05 * (maxy - miny)
ax.set_xlim(minx - padx, maxx + padx)
ax.set_ylim(miny - pady, maxy + pady)
ax.set_title(f"{GEOGRAPHIC_LABEL} centroids versus inherited RelWeights ties")
ax.legend(
    handles=[
        Line2D([0], [0], color="#64748b", lw=2, label=f"{GEOGRAPHIC_LABEL} edge"),
        Line2D([0], [0], color="#ea580c", lw=2.5, label="RelWeights-only inherited edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#111827", markersize=6, label="District centroid"),
    ],
    loc="upper right",
    frameon=True,
)
ax.set_axis_off()
plt.show()
```

This graph view uses polygon centroids as nodes. Gray edges reproduce the chosen same-layer contiguity graph, while the orange lines isolate **additional inherited ties** that only appear after the overlay operator is built. Those orange ties are the empirical payoff of RelWeights: districts can become connected because they share contextual supports, not because they share a border.

```{code-cell} ipython3
connectivity_map = make_connectivity_map(
    districts=districts,
    centroid_nodes=district_nodes,
    geographic_edges=geo_edges,
    rel_only_edges=rel_only_edges,
    id_col="ac_id",
    label_col="AC_NAME",
    geographic_label=GEOGRAPHIC_LABEL,
)

save_and_embed_folium_map(
    connectivity_map,
    interactive_dir / f"{GEOGRAPHIC_CONTIGUITY}-vs-relweights-connectivity.html",
)
```

The layer-toggle map above is the cleanest way to inspect the full graph. You can switch the district polygons, centroid nodes, geographic edges, and RelWeights-only inherited ties on and off without leaving the notebook.

## Hover through geographic and RelWeights neighborhoods

```{code-cell} ipython3
queen_neighbor_map = make_hover_neighbor_map(
    districts=districts,
    centroid_nodes=district_nodes,
    weights_df=W_geo,
    id_col="ac_id",
    label_col="AC_NAME",
    title=f"{GEOGRAPHIC_LABEL} neighborhood explorer",
    neighbor_color="#2563eb",
    edge_color="#2563eb",
)

save_and_embed_folium_map(
    queen_neighbor_map,
    interactive_dir / f"{GEOGRAPHIC_CONTIGUITY}-neighbor-explorer.html",
)
```

```{code-cell} ipython3
rel_neighbor_map = make_hover_neighbor_map(
    districts=districts,
    centroid_nodes=district_nodes,
    weights_df=R_indicator,
    id_col="ac_id",
    label_col="AC_NAME",
    title="RelWeights neighborhood explorer",
    neighbor_color="#ea580c",
    edge_color="#ea580c",
)

save_and_embed_folium_map(
    rel_neighbor_map,
    interactive_dir / "relweights-neighbor-explorer.html",
)
```

These two hover explorers mimic the GeoDa intuition you asked for. In the geographic map, hovering a district reveals its first-order neighbors under the chosen contiguity rule. In the RelWeights map, hovering the same district reveals the irregular but non-arbitrary set of districts connected through shared inherited supports.

## Build the relational Laplacian

```{code-cell} ipython3
L_R = laplacian_from_weights(R_binary)
print("Top-left block of L_R")
print(L_R.iloc[:8, :8].to_string())
```

The Laplacian is the operator form of the graph. Once $R$ is built, the rest of the pipeline is standard:

$$
L_R = D_R - R.
$$

This is the object used for energies, smoothing, spectral decomposition, and contextual diagnostics.

## Compute a demonstration Laplacian energy

To keep the example transparent, we will use a simple east-west signal on the districts: the standardized projected x-coordinate of each district centroid.

```{code-cell} ipython3
districts_proj = districts.to_crs(AREA_CRS).copy()
districts_proj["centroid_x"] = districts_proj.geometry.centroid.x

signal = (
    (districts_proj["centroid_x"] - districts_proj["centroid_x"].mean())
    / districts_proj["centroid_x"].std(ddof=0)
).to_numpy(dtype=float)
energy_R = centered_laplacian_energy(signal, L_R)
energy_W = centered_laplacian_energy(signal, laplacian_from_weights(W_geo))

print(f"Centered Laplacian energy under contextual R: {energy_R:,.2f}")
print(f"Centered Laplacian energy under geographic {GEOGRAPHIC_LABEL.lower()} weights: {energy_W:,.2f}")
```

These two energies are not on the same normalization scale, so the comparison is not a formal test. But they do illustrate a useful idea:

- $z^{\top} L_W z$ measures roughness over geographic adjacency
- $z^{\top} L_R z$ measures roughness over inherited contextual support

The support graph changes the notion of what counts as "smooth."

## What this lab established

This notebook implemented the construction chain

$$
\text{overlay} \longrightarrow B \longrightarrow R \longrightarrow L_R
$$

using a real polygon overlay example derived from the RelWeights repository. The key takeaway is not only that RelWeights can be computed from real data, but that they define a different neighborhood concept from same-layer adjacency.

## Exercises

1. Replace `B_bin` with `B_area` in the RelWeights construction and compare the resulting degree distribution.
2. Change the sample to a different district subset and check how the density of $R$ changes.
3. Switch `GEOGRAPHIC_CONTIGUITY` between `"queen"` and `"rook"` and compare how the geographic baseline changes relative to the contextual graph.
4. Compute $R$ from two inherited support layers stacked side by side rather than one.
5. Choose a substantive district attribute and compare its geographic and contextual Laplacian energies.
