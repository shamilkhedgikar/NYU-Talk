# From Overlays to Operators: Relational Spatial Weights for Data-Driven Policy

## Abstract

Data-driven urban policymaking increasingly relies on ideas developed in spatial econometrics, geospatial machine learning, and high-dimensional data systems. However, most analytical workflows still encode spatial dependence with simple contiguity derived from geographic adjacency or distance-decay. These assumptions often fail to capture the relational structure of spatial systems, where interactions are mediated across layers such as zoning regimes, infrastructure networks, institutional boundaries, and environmental systems.

This notebook introduces Relational Spatial Weights (RelWeights) as a framework for turning traditionally thematic spatial overlays into formal spatial operators usable in spatial econometrics and geospatial data science. RelWeights provide the ability to formalize irregular but non-arbitrary spatial lags, enable empirical identification of layer-specific spatial externalities, and provide a mechanism for transferring spatial information across geographic scales and heterogeneous supports.

Conceptually, they were developed to bridge two traditions of *space* and *place* that have historically diverged. Regional science has usually modeled spatial dependence through geometric proximity across an abstract idea of space, while planning and urban theory have emphasized place as a contextual construct shaped by institutions, infrastructure, and layered territorial relations. RelWeights translates those contextual overlay relationships into quantitative operators that can be estimated, compared, and reused inside real policy workflows.

## Space, Place and the Limits of Conventional Spatial Weights

Spatial econometrics and regional science have historically modeled spatial dependence through geometrically defined spatial weights matrices based on contiguity, distance, or nearest-neighbor relationships [@anselin1988]. These approaches reflect the assumption that spatial interaction primarily operates through geographic proximity.

However, urban systems frequently exhibit relational dependencies that cannot be adequately captured through purely geometric representations of space. Urban theory has long emphasized the importance of *place* as a contextual construct shaped by institutional, infrastructural, and socio-economic relationships [@portugali2011]. In planning practice, these contextual relationships are frequently analyzed through overlay techniques combining multiple spatial layers such as zoning, infrastructure networks, environmental systems, and administrative boundaries. Starting with qualitative observation and analysis [@mcharg1969design], this tradition has has since accomodated methdologies such as Likert scoring, expert judgment, and other subjective ranking methods translating qualitative assessments into quasi-quantitative outputs.

Despite their conceptual importance, such overlay relationships have rarely been incorporated directly into spatial econometric models. As a result, many spatial models rely on weights matrices that poorly or completely fail to capture the actual mechanisms generating spatial dependence in real-world spatial systems across multiple "layers".

## Why Conventional Weights Are Not Enough

Classic spatial econometric weights assume that dependence is primarily a function of shared borders, nearest neighbors, or geographic distance. Those assumptions are useful, but they are only one special case. In many urban and regional systems, spillovers are organized by supports that do not align with administrative boundaries: aquifers cross districts, transit systems connect noncontiguous neighborhoods, school catchments cut across census geographies, and zoning overlays reshape accessibility and land-use behavior.

Overlay analysis has long been the practical way planners reason about these layered relationships, but overlays are usually left as descriptive GIS products rather than promoted into reusable analytical operators. The consequence is that many models still test spatial dependence with weights matrices that do not match the mechanism actually generating the dependence.

## Why This Matters Now

Recent geospatial machine learning advances, particularly the propogation of embeddings makes this gap gap more visible. Spatial embeddings derived from imagery, mobility traces, and large socioeconomic datasets can represent places as dense latent vectors, but those embeddings are often learned on supports that do not match the units used in econometric or policy analysis. That creates a transfer problem: how should information move from one geography to another without collapsing everything back to naive proximity?

RelWeights provides one answer. By defining similarity through shared contextual overlays rather than only through adjacency, it creates a principled way to transfer, diffuse, and regularize information across heterogeneous spatial supports.