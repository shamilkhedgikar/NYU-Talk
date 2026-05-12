# From Overlays to Operators: Relational Spatial Weights for Data-Driven Policy

:::{div}
:class: journal-author-line

Shamil Khedgikar[^author-affiliation]
:::

<div id="abstract" class="intro-abstract-heading" role="heading" aria-level="2">Abstract</div>

Data-driven urban policymaking increasingly relies on ideas developed in spatial econometrics, geospatial machine learning, and high-dimensional data systems. However, most analytical workflows still encode spatial dependence with simple contiguity derived from geographic adjacency or distance-decay. These assumptions often fail to capture the relational structure of spatial systems, where interactions are mediated across layers such as zoning regimes, infrastructure networks, institutional boundaries, and environmental systems.

This notebook introduces Relational Spatial Weights (RelWeights) as a framework for turning traditionally thematic spatial overlays into formal spatial operators usable in spatial econometrics and data science. RelWeights provide the ability to formalize irregular but non-arbitrary spatial lags, enable empirical identification of layer-specific spatial externalities, and provide a mechanism for transferring spatial information across geographic scales and heterogeneous supports.

Conceptually, RelWeights aim to bridge this divergence between methodological traditions of *space* and *place* that have historically diverged. Regional science has usually modeled spatial dependence through geometric proximity across an abstract idea of space, while planning and urban theory have emphasized place as a contextual construct shaped by institutions, infrastructure, and layered territorial relations. RelWeights translates those contextual overlay relationships into quantitative operators that can be estimated, compared, and reused inside real policy workflows.

## Space, Place and the Limits of Conventional Spatial Weights

Spatial econometrics and regional science have historically modeled spatial dependence through geometrically defined spatial weights matrices based on contiguity, distance, or nearest-neighbor relationships [@anselin1988]. These approaches reflect the assumption that spatial interaction primarily operates through geographic proximity.

However, urban systems frequently exhibit relational dependencies that cannot be adequately captured through purely geometric representations of *space*. Urban theory has long emphasized the importance of *place* as a contextual construct shaped by institutional, infrastructural, and socio-economic relationships [@portugali2011]. In planning practice, these contextual relationships are frequently analyzed through overlay techniques combining multiple spatial layers such as zoning, infrastructure networks, environmental systems, and administrative boundaries. Starting with qualitative observation and analysis [@mcharg1969design], this tradition has has since incorporated methdologies such as Likert scoring, expert judgment, and other subjective ranking methods translating qualitative assessments into quasi-quantitative outputs.

Despite their conceptual importance, such overlay relationships have rarely been explicitly incorporated directly into spatial econometrics and data science. As a result, many spatial models rely on weights matrices that poorly or completely fail to capture the actual mechanisms generating spatial dependence in real-world spatial systems across multiple "layers".

```{figure} assets/images/intro_multiple_layers-1.jpg
:width: 40%
:name: fig-overlay

Overlay relationships across zoning, infrastructure, and environmental layers.
```

## The Importance of Choosing W

Classic spatial econometric weights assume that dependence is primarily a function of shared borders, nearest neighbors, or geographic distance. To capture this dependence through an explicit relationship, $W$, also known as the spatial weights matrix, is defined as an $N X N$ matrix which specifies how each unit of analysis is affected by neighboring units. $W$ is (mostly) exogenous, and a part of the model specification. It encodes the adjacency and the connectivity structure through which spatial dependence is exhibited and assessed. Across different specifications, competing $W's$ represent different structural hypotheses about the underlying spatial interactions that are being evaluated.

In many urban and regional systems, these interactions (and therefore spillovers) are organized by supports that do not align with the primary unit of analysis: aquifers crossing districts, transit systems connecting noncontiguous neighborhoods, school catchments cutting across census geographies, and zoning overlays influencing accessibility and land-use behavior. Therefore, many model specifications fail to capture the actual mechanisms generating spatial dependence in real-world spatial systems.

To overcome this problem, we consider the idea that overlays, which are are usually treated as descriptive or quasi-analytical technique, should be promoting them into *reusable analytical operators* [@khedgikar2023].

## Why This Matters Now

Recent advances in machine learning such as [@agarwal2024pdfm],[@brown2025alphaearth] have further expanded the analytical possibilities for spatial data through the development of high-dimensional spatial embeddings derived from satellite imagery, mobility data, and large-scale socioeconomic datasets. These embeddings represent spatial units as vectors in a latent feature space, capturing complex contextual relationships that are difficult to encode through traditional variables. However, a persistent challenge in applying such representations within spatial analysis is that embeddings are often generated at spatial supports that do not align with the units used in policy analysis or econometric modeling. As a result, there is a growing need for formal operators capable of transferring or propagating these representations across heterogeneous spatial supports. RelWeights provide one such mechanism: by defining spatial similarity through shared contextual overlays rather than purely geometric adjacency, they allow embeddings learned at one spatial layer to be systematically transferred or diffused to another. By defining similarity through shared contextual overlays rather than only through adjacency, we aim to develop a principled way to transfer, diffuse, and regularize information across heterogeneous spatial supports.

[^author-affiliation]: AECOM
