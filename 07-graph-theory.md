# Graph Theory

Graph theory is a branch of mathematics that studies networks of connected objects, providing fundamental tools for analyzing complex systems in bioinformatics.

## Basic Definitions

### Graph
A graph G = (V, E) consists of:
- **V**: Set of vertices (nodes)
- **E**: Set of edges (links) connecting vertices

### Types of Graphs

#### By Edge Direction
- **Undirected graph**: Edges have no direction (symmetric relationships)
- **Directed graph (digraph)**: Edges have direction (asymmetric relationships)
- **Mixed graph**: Contains both directed and undirected edges

#### By Edge Weights
- **Unweighted graph**: All edges are equivalent
- **Weighted graph**: Edges have associated weights/costs

#### By Connectivity
- **Simple graph**: No loops (self-edges) or multiple edges
- **Multigraph**: Multiple edges between same vertices allowed
- **Pseudograph**: Self-loops allowed

### Graph Representations

#### Adjacency Matrix
**A[i,j] = 1** if edge exists between vertices i and j, 0 otherwise

**Properties**:
- Symmetric for undirected graphs
- Size: n × n where n = |V|
- Space complexity: O(n²)

#### Adjacency List
Each vertex maintains list of its neighbors

**Properties**:
- More space-efficient for sparse graphs
- Space complexity: O(|V| + |E|)
- Faster for many algorithms

#### Edge List
Simple list of all edges in the graph

**Format**: [(u₁, v₁), (u₂, v₂), ..., (uₘ, vₘ)]

## Fundamental Concepts

### Degree
- **Degree of vertex v**: Number of edges incident to v
- **In-degree**: Number of incoming edges (directed graphs)
- **Out-degree**: Number of outgoing edges (directed graphs)

#### Handshaking Lemma
**Σ deg(v) = 2|E|**

The sum of all vertex degrees equals twice the number of edges.

### Paths and Connectivity

#### Walk
Sequence of vertices where consecutive vertices are connected by edges.

#### Trail
Walk where no edge is repeated.

#### Path
Walk where no vertex is repeated.

#### Cycle
Closed path (starts and ends at same vertex).

#### Connected Graph
There exists a path between every pair of vertices.

#### Connected Components
Maximal sets of vertices that are mutually reachable.

### Distance and Diameter

#### Distance
**d(u,v)**: Length of shortest path between vertices u and v

#### Diameter
**diam(G) = max{d(u,v) : u,v ∈ V}**

Maximum distance between any two vertices.

#### Eccentricity
**ecc(v) = max{d(v,u) : u ∈ V}**

Maximum distance from vertex v to any other vertex.

#### Radius
**rad(G) = min{ecc(v) : v ∈ V}**

Minimum eccentricity among all vertices.

## Special Types of Graphs

### Trees
Connected acyclic graphs.

**Properties**:
- **n vertices, n-1 edges** for connected tree
- **Unique path** between any two vertices
- **Removing any edge** disconnects the graph
- **Adding any edge** creates exactly one cycle

#### Rooted Trees
Tree with designated root vertex.
- **Parent-child relationships**
- **Depth**: Distance from root
- **Height**: Maximum depth

#### Spanning Tree
Subgraph that includes all vertices and is a tree.
- **Minimum spanning tree (MST)**: Spanning tree with minimum total edge weight

### Bipartite Graphs
Vertices can be divided into two disjoint sets where edges only connect vertices from different sets.

**Applications**:
- Matching problems
- Gene-disease associations
- Protein-protein interactions across complexes

### Complete Graphs
Every pair of distinct vertices is connected by edge.
- **Notation**: Kₙ (complete graph on n vertices)
- **Number of edges**: n(n-1)/2

### Regular Graphs
All vertices have the same degree.
- **k-regular**: All vertices have degree k

### Planar Graphs
Can be drawn on a plane without edge crossings.

**Euler's Formula**: V - E + F = 2 (where F is number of faces)

## Graph Algorithms

### Traversal Algorithms

#### Breadth-First Search (BFS)
**Algorithm**:
1. Start from source vertex
2. Visit all neighbors at current depth
3. Move to next depth level
4. Repeat until all reachable vertices visited

**Applications**:
- Shortest path in unweighted graphs
- Level-order traversal
- Bipartiteness testing

**Time Complexity**: O(|V| + |E|)

#### Depth-First Search (DFS)
**Algorithm**:
1. Start from source vertex
2. Explore as far as possible along each branch
3. Backtrack when no more unvisited neighbors
4. Repeat until all reachable vertices visited

**Applications**:
- Topological sorting
- Strongly connected components
- Cycle detection

**Time Complexity**: O(|V| + |E|)

### Shortest Path Algorithms

#### Dijkstra's Algorithm
Finds shortest paths from source to all other vertices in weighted graph with non-negative weights.

**Algorithm**:
1. Initialize distances (0 for source, ∞ for others)
2. Maintain priority queue of unvisited vertices
3. Process vertex with minimum distance
4. Update distances to neighbors
5. Repeat until all vertices processed

**Time Complexity**: O((|V| + |E|) log |V|) with binary heap

#### Bellman-Ford Algorithm
Handles negative edge weights, detects negative cycles.

**Algorithm**:
1. Initialize distances
2. Relax all edges |V|-1 times
3. Check for negative cycles

**Time Complexity**: O(|V| × |E|)

#### Floyd-Warshall Algorithm
Finds shortest paths between all pairs of vertices.

**Algorithm**: Dynamic programming approach
**Time Complexity**: O(|V|³)

### Minimum Spanning Tree

#### Kruskal's Algorithm
**Algorithm**:
1. Sort edges by weight
2. Initialize each vertex as separate component
3. For each edge, if it connects different components, add to MST
4. Use union-find for efficient component tracking

**Time Complexity**: O(|E| log |E|)

#### Prim's Algorithm
**Algorithm**:
1. Start with arbitrary vertex
2. Repeatedly add minimum weight edge connecting tree to non-tree vertex
3. Continue until all vertices included

**Time Complexity**: O(|E| log |V|) with binary heap

### Network Flow

#### Maximum Flow Problem
Find maximum amount of flow from source to sink.

#### Ford-Fulkerson Algorithm
**Method**: Find augmenting paths until no more exist
**Time Complexity**: O(|E| × max_flow)

#### Edmonds-Karp Algorithm
Ford-Fulkerson with BFS for finding augmenting paths.
**Time Complexity**: O(|V| × |E|²)

## Graph Properties and Measures

### Centrality Measures

#### Degree Centrality
**C_D(v) = deg(v) / (n-1)**

Proportion of nodes that are neighbors of v.

#### Betweenness Centrality
**C_B(v) = Σ σ(s,t|v) / σ(s,t)**

Where σ(s,t) is number of shortest paths from s to t, and σ(s,t|v) is number passing through v.

#### Closeness Centrality
**C_C(v) = (n-1) / Σ d(v,u)**

Inverse of average distance to all other vertices.

#### Eigenvector Centrality
**x_v = (1/λ) Σ A_{v,t} x_t**

Where λ is the largest eigenvalue of adjacency matrix A.

#### PageRank
**PR(v) = (1-d)/n + d Σ PR(u)/L(u)**

Where d is damping factor, L(u) is out-degree of u.

### Clustering

#### Clustering Coefficient
**Local clustering coefficient**:
**C_i = 2e_i / (k_i(k_i-1))**

Where e_i is number of edges between neighbors of vertex i, k_i is degree of i.

**Global clustering coefficient**:
**C = 3 × (number of triangles) / (number of connected triples)**

### Small World Properties

#### Small World Networks
- **High clustering**: Vertices form tightly knit groups
- **Short path lengths**: Despite clustering, average distances are small

#### Watts-Strogatz Model
Interpolates between regular lattice and random graph.

### Scale-Free Networks

#### Power Law Degree Distribution
**P(k) ∝ k^(-γ)**

Where γ is typically between 2 and 3.

#### Preferential Attachment
"Rich get richer" - new vertices preferentially connect to high-degree vertices.

#### Barabási-Albert Model
Generates scale-free networks through preferential attachment.

## Graph Partitioning and Clustering

### Community Detection

#### Modularity
**Q = (1/2m) Σ [A_{ij} - k_i k_j / 2m] δ(c_i, c_j)**

Where m is number of edges, k_i is degree of vertex i, c_i is community of vertex i.

#### Louvain Algorithm
Greedy optimization of modularity.

#### Spectral Clustering
Uses eigenvectors of graph Laplacian matrix.

### Graph Cuts

#### Minimum Cut
Minimum number of edges to remove to disconnect graph.

#### Normalized Cut
**NCut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)**

Where vol(A) is sum of degrees in set A.

## Random Graphs

### Erdős–Rényi Model
**G(n,p)**: n vertices, each edge exists with probability p

**Properties**:
- **Expected degree**: (n-1)p
- **Giant component** emerges when p > 1/n
- **Poisson degree distribution** for large n

### Configuration Model
Generates random graph with specified degree sequence.

### Stochastic Block Model
Vertices divided into blocks, edge probabilities depend on block membership.

## Dynamic Graphs

### Temporal Networks
Graphs that change over time.

**Representations**:
- **Sequence of snapshots**
- **Stream of edge events**
- **Contact sequences**

### Graph Evolution
- **Preferential attachment**: Rich get richer
- **Small world emergence**: Rewiring process
- **Densification laws**: Real networks densify over time

## Bioinformatics Applications

### Protein-Protein Interaction Networks

#### Structure
- **Vertices**: Proteins
- **Edges**: Physical or functional interactions
- **Properties**: Scale-free, small world

#### Analysis
- **Essential proteins**: High degree, high betweenness centrality
- **Protein complexes**: Dense subgraphs
- **Functional modules**: Communities

### Gene Regulatory Networks

#### Structure
- **Vertices**: Genes/transcription factors
- **Directed edges**: Regulatory relationships
- **Edge weights**: Strength of regulation

#### Analysis
- **Master regulators**: High out-degree
- **Feedback loops**: Cycles in directed graph
- **Motifs**: Recurring regulatory patterns

### Metabolic Networks

#### Structure
- **Vertices**: Metabolites or reactions
- **Edges**: Substrate-product relationships
- **Bipartite representation**: Separate metabolite and reaction nodes

#### Analysis
- **Essential metabolites**: High degree, high betweenness
- **Metabolic pathways**: Paths through network
- **Flux analysis**: Network flow problems

### Phylogenetic Networks

#### Structure
- **Vertices**: Species or sequences
- **Edges**: Evolutionary relationships
- **Tree vs network**: Networks allow reticulation events

#### Analysis
- **Ancestral reconstruction**: Shortest path problems
- **Hybridization events**: Network-specific patterns
- **Evolutionary distance**: Graph distances

### Ecological Networks

#### Food Webs
- **Vertices**: Species
- **Directed edges**: Predator-prey relationships
- **Analysis**: Stability, trophic levels

#### Pollination Networks
- **Bipartite structure**: Plants and pollinators
- **Analysis**: Nestedness, robustness

### Disease Networks

#### Disease-Gene Networks
- **Vertices**: Diseases and genes
- **Edges**: Disease-gene associations
- **Analysis**: Disease similarity, gene prioritization

#### Epidemiological Networks
- **Vertices**: Individuals
- **Edges**: Contact or transmission events
- **Analysis**: Outbreak patterns, intervention strategies

## Computational Tools and Software

### General Purpose

#### NetworkX (Python)
Comprehensive graph analysis library.

#### igraph (R/Python/C)
Fast graph analysis with good visualization.

#### SNAP (C++/Python)
Stanford Network Analysis Platform for large networks.

### Bioinformatics-Specific

#### Cytoscape
Interactive network visualization and analysis.

#### STRING
Protein-protein interaction database and analysis.

#### Gephi
Graph visualization and exploration platform.

#### BioGraph
R package for biological network analysis.

### Databases

#### Pathway Databases
- **KEGG**: Metabolic pathways
- **Reactome**: Biological pathways
- **WikiPathways**: Community-curated pathways

#### Interaction Databases
- **BioGRID**: Protein and genetic interactions
- **IntAct**: Molecular interaction database
- **STRING**: Protein-protein interaction networks

## Advanced Topics

### Multilayer Networks
Networks with multiple types of relationships.

**Applications**:
- **Multiplex protein networks**: Different interaction types
- **Social-ecological systems**: Multiple relationship types

### Hypergraphs
Edges can connect more than two vertices.

**Applications**:
- **Protein complexes**: Multi-protein interactions
- **Metabolic reactions**: Multiple substrates/products

### Network Machine Learning

#### Node Classification
Predict properties of vertices based on network structure.

#### Link Prediction
Predict missing or future edges.

#### Graph Neural Networks
Deep learning on graph-structured data.

### Network Robustness
How networks respond to node/edge removal.

**Measures**:
- **Connectivity robustness**: Maintaining connectivity
- **Efficiency robustness**: Maintaining short paths
- **Targeted vs random attacks**

## Best Practices

### Data Quality
1. **Interaction confidence**: Weight edges by confidence scores
2. **False positives**: Account for experimental errors
3. **Missing interactions**: Consider network incompleteness

### Analysis Considerations
1. **Multiple testing**: Correct for multiple hypothesis tests
2. **Network bias**: Account for study bias in network construction
3. **Null models**: Use appropriate random network models

### Visualization
1. **Layout algorithms**: Choose appropriate layout for network type
2. **Node/edge attributes**: Use visual encoding effectively
3. **Scalability**: Handle large networks appropriately

### Reproducibility
1. **Data provenance**: Document data sources and processing
2. **Parameter settings**: Report all analysis parameters
3. **Software versions**: Document software and version numbers