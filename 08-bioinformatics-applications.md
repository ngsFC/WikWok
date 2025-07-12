# Bioinformatics Applications of Statistics and Graph Theory

This section demonstrates how statistical methods and graph theory are applied to solve real-world problems in bioinformatics and computational biology.

## Genomics and Statistics

### Genome-Wide Association Studies (GWAS)

#### Statistical Framework
GWAS tests associations between genetic variants and phenotypes across the genome.

**Basic Model**: For each SNP, test association with phenotype
- **Quantitative traits**: Linear regression
- **Binary traits**: Logistic regression
- **Survival traits**: Cox proportional hazards

**Example Linear Model**:
**Y = β₀ + β₁ × SNP + β₂ × PC1 + β₃ × PC2 + ε**

Where PC1, PC2 are principal components controlling for population structure.

#### Multiple Testing Challenge
- **Problem**: Testing millions of SNPs increases Type I error
- **Bonferroni correction**: α = 0.05 / (number of SNPs tested)
- **Typical threshold**: p < 5 × 10⁻⁸
- **FDR control**: Benjamini-Hochberg procedure

#### Population Structure
**Problem**: Confounding due to ancestry differences
**Solutions**:
- **Principal Component Analysis**: Include PCs as covariates
- **Genomic Control**: Adjust test statistics by genomic inflation factor λ
- **Mixed models**: Account for relatedness structure

#### Power Analysis
**Factors affecting power**:
- **Sample size**: Larger N increases power
- **Effect size**: Larger effects easier to detect
- **Allele frequency**: Rare variants need larger samples
- **Linkage disequilibrium**: Affects tagging efficiency

**Power formula** (approximate):
**Power ≈ Φ(√(N × h²) - z_(α/2))**

Where h² is proportion of variance explained by SNP.

### Copy Number Variation Analysis

#### Statistical Models
**Hidden Markov Models (HMMs)**:
- **States**: Normal, deletion, duplication
- **Observations**: Log intensity ratios
- **Transitions**: Probability of CNV boundaries

**Circular Binary Segmentation**:
- **Objective**: Identify segments with constant copy number
- **Method**: Recursive binary partitioning
- **Test**: Likelihood ratio test for segment differences

#### Bayesian Approaches
**Advantages**:
- **Uncertainty quantification**: Posterior probabilities
- **Prior information**: Incorporate known CNV regions
- **Model averaging**: Account for model uncertainty

### Population Genetics

#### Hardy-Weinberg Equilibrium Testing
**Expected frequencies** under HWE:
- **AA**: p²
- **Aa**: 2pq  
- **aa**: q²

**Chi-square test**:
**χ² = N[(O₁₁ - E₁₁)² / E₁₁ + (O₁₂ - E₁₂)² / E₁₂ + (O₂₂ - E₂₂)² / E₂₂]**

#### Linkage Disequilibrium
**Measures**:
- **D**: D = p₁₁p₂₂ - p₁₂p₂₁
- **D'**: D' = D / Dₘₐₓ
- **r²**: Correlation coefficient squared

**Applications**:
- **Tag SNP selection**: Choose representative SNPs
- **Fine mapping**: Narrow association signals
- **Haplotype inference**: Reconstruct haplotypes

#### Population Structure Analysis
**Principal Component Analysis**:
- **Input**: Genotype matrix (individuals × SNPs)
- **Output**: Principal components representing ancestry
- **Visualization**: PC plots show population clusters

**ADMIXTURE/STRUCTURE**:
- **Model**: Each individual has ancestry proportions
- **Method**: Maximum likelihood or MCMC
- **Output**: Ancestry coefficients for K populations

### Phylogenetics and Evolution

#### Sequence Evolution Models
**Jukes-Cantor Model**: Equal substitution rates
**P(t) = 1/4 + 3/4 × e^(-4μt)**

**Kimura 2-Parameter**: Different rates for transitions/transversions
**General Time Reversible (GTR)**: Most flexible substitution model

#### Maximum Likelihood Phylogeny
**Likelihood function**:
**L(T,θ) = ∏ P(xᵢ | T, θ)**

Where xᵢ are observed sequences, T is tree topology, θ are model parameters.

**Optimization**:
- **Tree search**: Heuristic algorithms (NJ, ML, MP)
- **Bootstrap**: Assess branch support
- **Model selection**: AIC, BIC for model comparison

#### Molecular Clock Analysis
**Relaxed clock models**: Allow rate variation across branches
**Bayesian methods**: BEAST, MrBayes for divergence time estimation

## Transcriptomics and Statistics

### RNA-Seq Analysis

#### Count Data Modeling
**Negative Binomial Distribution**: Models overdispersion in count data
**PMF**: P(X = k) = Γ(k + r) / (k!Γ(r)) × (r/(r+μ))ʳ × (μ/(r+μ))ᵏ

Where μ is mean, r is dispersion parameter.

#### Differential Expression Analysis
**DESeq2 Model**:
**log₂(μᵢⱼ) = xᵢⱼβⱼ**

Where μᵢⱼ is expected count for gene i in sample j.

**Statistical Testing**:
- **Wald test**: For single coefficients
- **Likelihood ratio test**: For multiple coefficients
- **Shrinkage estimation**: Improve dispersion estimates

#### Normalization Methods
**TMM (Trimmed Mean of M-values)**:
- **Account for library size differences**
- **Robust to highly expressed genes**

**FPKM/RPKM**: Fragments/Reads per kilobase per million
**TPM**: Transcripts per million (preferred over FPKM)

#### Multiple Testing Correction
**Independent Hypothesis Weighting (IHW)**:
- **Use mean expression as weight**
- **Increase power while controlling FDR**

### Single-Cell RNA-Seq

#### Unique Statistical Challenges
- **Zero inflation**: Many genes not detected
- **High noise**: Technical and biological variability
- **Dimensionality**: Thousands of cells, genes

#### Zero-Inflated Models
**ZINB-WaVE**: Zero-inflated negative binomial with random effects
**SCDE**: Mixture of negative binomial and Poisson

#### Dimensionality Reduction
**Principal Component Analysis**: Standard approach
**t-SNE**: Non-linear dimensionality reduction
**UMAP**: Uniform Manifold Approximation and Projection

#### Cell Type Classification
**Supervised methods**: Train on labeled cells
**Unsupervised clustering**: k-means, hierarchical clustering
**Trajectory inference**: Pseudotime analysis

### Gene Set Enrichment Analysis

#### Over-Representation Analysis (ORA)
**Fisher's Exact Test**: Test for enrichment in gene list
**Hypergeometric distribution**: Probability model

#### Gene Set Enrichment Analysis (GSEA)
**Enrichment Score**: ES(S) = Σ (|rⱼ|^p / N_R) - Σ (1 / (N - N_R))
**Normalization**: Account for gene set size
**Permutation testing**: Generate null distribution

#### Functional Class Scoring
**Camera**: Competitive gene set testing
**ROAST**: Rotation gene set testing
**QuSAGE**: Quantitative set analysis for gene expression

## Proteomics Applications

### Mass Spectrometry Data Analysis

#### Peak Detection
**Statistical approaches**:
- **Signal-to-noise ratio**: Identify significant peaks
- **Wavelet transforms**: Multi-scale peak detection
- **Mixture models**: Model peak and noise distributions

#### Protein Identification
**Database Search**:
- **Scoring functions**: XCorr, Hyperscore, E-values
- **FDR control**: Target-decoy approach
- **Statistical validation**: Percolator, PeptideProphet

#### Quantitative Proteomics
**Label-free quantification**:
- **Spectral counting**: Number of MS/MS spectra
- **Intensity-based**: Peak area/height measurements
- **Statistical testing**: t-tests, limma for differential abundance

**Stable isotope labeling**:
- **SILAC**: Metabolic labeling
- **iTRAQ/TMT**: Chemical labeling
- **Statistical models**: Mixed effects models for complex designs

### Protein Structure Analysis

#### Secondary Structure Prediction
**Machine learning approaches**:
- **Neural networks**: PSI-PRED, SPIDER
- **Support vector machines**: Classification of structure states
- **Ensemble methods**: Combine multiple predictors

#### Protein Folding Simulation
**Molecular dynamics**: Statistical mechanics simulations
**Monte Carlo methods**: Sample conformational space
**Free energy calculations**: Thermodynamic integration

## Network Biology Applications

### Protein-Protein Interaction Networks

#### Network Construction
**Data integration**:
- **Experimental data**: Y2H, co-immunoprecipitation
- **Computational prediction**: Sequence similarity, domain interactions
- **Literature mining**: Text extraction of interactions

**Quality assessment**:
- **Confidence scoring**: Combine evidence from multiple sources
- **Cross-validation**: Assess prediction accuracy
- **Gold standard evaluation**: Compare against known interactions

#### Network Analysis Methods
**Module detection**:
- **Clustering algorithms**: MCL, MCODE, ClusterONE
- **Community detection**: Modularity optimization
- **Statistical significance**: Compare to random networks

**Functional annotation**:
- **Guilt by association**: Function transfer between neighbors
- **Network-based GO enrichment**: Account for network structure
- **Essential gene prediction**: Centrality measures

### Gene Regulatory Networks

#### Network Inference
**Correlation-based methods**:
- **Pearson correlation**: Linear relationships
- **Mutual information**: Non-linear dependencies
- **Partial correlation**: Direct relationships

**Causal inference methods**:
- **Granger causality**: Temporal causation
- **Bayesian networks**: Probabilistic causal models
- **Instrumental variables**: Address confounding

#### Dynamic Networks
**Time series analysis**:
- **Vector autoregression**: Model temporal dependencies
- **State space models**: Hidden regulatory states
- **Change point detection**: Identify regime changes

### Metabolic Networks

#### Flux Balance Analysis (FBA)
**Optimization problem**:
**Maximize**: c^T v
**Subject to**: Sv = 0, vₗᵦ ≤ v ≤ vᵤᵦ

Where S is stoichiometric matrix, v is flux vector.

**Extensions**:
- **FVA**: Flux variability analysis
- **pFBA**: Parsimonious FBA
- **Dynamic FBA**: Time-varying constraints

#### Network-based Drug Discovery
**Target identification**:
- **Essential nodes**: High degree, high betweenness
- **Chokepoints**: Unique consumers/producers
- **Network controllability**: Minimum driver nodes

**Drug repurposing**:
- **Network proximity**: Distance between drug and disease modules
- **Random walk**: Diffusion of drug effects
- **Machine learning**: Predict drug-target interactions

## Structural Bioinformatics

### Protein Structure Networks

#### Contact Networks
**Construction**:
- **Residue contacts**: Distance-based criteria
- **Side chain contacts**: Cβ-Cβ distances
- **Backbone contacts**: Secondary structure elements

**Analysis**:
- **Hub residues**: High degree nodes (often important for stability)
- **Shortest paths**: Communication pathways
- **Centrality measures**: Identify key residues

#### Allosteric Communication
**Network flow models**:
- **Current flow betweenness**: Information flow through network
- **Suboptimal path analysis**: Multiple communication routes
- **Community detection**: Identify rigid domains

### Molecular Dynamics Networks

#### Correlation Networks
**Construction**:
- **Cross-correlation**: C_ij = ⟨Δr_i · Δr_j⟩
- **Mutual information**: Non-linear correlations
- **Transfer entropy**: Directional information flow

**Analysis**:
- **Dynamic communities**: Time-varying modules
- **Network robustness**: Response to perturbations
- **Allosteric pathways**: Communication routes

## Evolutionary Biology Applications

### Phylogenetic Networks

#### Reticulate Evolution
**Models**:
- **Hybridization networks**: Allow horizontal gene transfer
- **Recombination networks**: Capture recombination events
- **Migration networks**: Population mixing

**Inference methods**:
- **Parsimony**: Minimize reticulation events
- **Maximum likelihood**: Probabilistic models
- **Bayesian methods**: Posterior distributions over networks

#### Ancestral Sequence Reconstruction
**Statistical methods**:
- **Maximum likelihood**: Most probable ancestral states
- **Bayesian**: Posterior distributions over states
- **Parsimony**: Minimize changes along tree

### Comparative Genomics

#### Synteny Analysis
**Statistical tests**:
- **Hypergeometric test**: Gene order conservation
- **Permutation tests**: Random genome rearrangements
- **Markov models**: Synteny block evolution

#### Molecular Evolution Rates
**dN/dS analysis**:
- **Neutral evolution**: dN/dS = 1
- **Purifying selection**: dN/dS < 1
- **Positive selection**: dN/dS > 1

**Statistical tests**:
- **Likelihood ratio tests**: Compare selection models
- **Branch-site models**: Site-specific selection
- **Codon models**: Account for genetic code

## Epidemiological Networks

### Disease Transmission Models

#### Basic Reproduction Number (R₀)
**Definition**: Average number of secondary infections from one infected individual
**Network-based**: R₀ = λ⟨k²⟩/⟨k⟩ (for degree-based models)

Where λ is transmission rate, ⟨k⟩ is mean degree, ⟨k²⟩ is second moment.

#### SIR Models on Networks
**Compartments**: Susceptible → Infected → Recovered
**Network effects**: Contact structure affects transmission
**Threshold behavior**: Epidemic occurs if R₀ > 1

### Contact Tracing Networks

#### Network Reconstruction
**Data sources**:
- **GPS tracking**: Proximity-based contacts
- **Bluetooth beacons**: Close contact detection
- **Survey data**: Self-reported contacts

**Privacy considerations**:
- **Differential privacy**: Add noise to protect individuals
- **k-anonymity**: Ensure groups of size ≥ k
- **Data aggregation**: Remove identifying information

#### Intervention Strategies
**Targeted vaccination**:
- **Degree-based**: Vaccinate high-degree nodes
- **Betweenness-based**: Target bridge nodes
- **Random walk**: Identify influential spreaders

**Contact reduction**:
- **Social distancing**: Reduce edge weights
- **Quarantine**: Remove infected nodes
- **Cluster isolation**: Disconnect communities

## Machine Learning in Bioinformatics

### Deep Learning Applications

#### Sequence Analysis
**Convolutional Neural Networks**:
- **Motif detection**: Learn regulatory patterns
- **Splice site prediction**: Identify exon-intron boundaries
- **Protein binding**: Predict DNA-protein interactions

**Recurrent Neural Networks**:
- **Language models**: Protein/DNA sequence modeling
- **Structure prediction**: Secondary structure from sequence
- **Gene expression**: Time series analysis

#### Image Analysis
**Microscopy image analysis**:
- **Cell segmentation**: Identify cell boundaries
- **Phenotype classification**: Classify cellular states
- **Protein localization**: Subcellular location prediction

### Graph Neural Networks

#### Applications
**Drug discovery**:
- **Molecular property prediction**: QSAR modeling
- **Drug-target interaction**: Predict binding affinity
- **Synthetic accessibility**: Predict synthesis difficulty

**Protein analysis**:
- **Function prediction**: From structure networks
- **Interaction prediction**: Protein-protein binding
- **Stability prediction**: Mutation effects

#### Architectures
**Graph Convolutional Networks (GCN)**:
- **Message passing**: Aggregate neighbor information
- **Node embeddings**: Learn node representations
- **Graph-level prediction**: Entire graph properties

**Graph Attention Networks**:
- **Attention mechanism**: Weight neighbor contributions
- **Multi-head attention**: Learn multiple representations
- **Interpretability**: Visualize attention weights

## Statistical Challenges in Bioinformatics

### High-Dimensional Data

#### Curse of Dimensionality
**Problems**:
- **Sparse data**: More parameters than observations
- **Distance concentration**: All points equidistant in high dimensions
- **Overfitting**: Models memorize noise

**Solutions**:
- **Regularization**: Ridge, Lasso, Elastic Net
- **Dimension reduction**: PCA, t-SNE, UMAP
- **Feature selection**: Choose relevant variables

### Missing Data

#### Types of Missingness
**MCAR**: Missing completely at random
**MAR**: Missing at random (conditional on observed data)
**MNAR**: Missing not at random (depends on unobserved values)

#### Imputation Methods
**Simple imputation**:
- **Mean/median imputation**: Replace with central tendency
- **Last observation carried forward**: Time series data

**Advanced methods**:
- **Multiple imputation**: Generate multiple complete datasets
- **Matrix completion**: Low-rank matrix recovery
- **Deep learning**: Autoencoders for imputation

### Batch Effects

#### Sources
**Technical factors**:
- **Experimental date**: Day-to-day variation
- **Laboratory**: Cross-lab differences
- **Platform**: Different measurement technologies

#### Correction Methods
**Linear models**: Include batch as covariate
**ComBat**: Empirical Bayes batch correction
**SVA**: Surrogate variable analysis
**RUV**: Remove unwanted variation

### Reproducibility Crisis

#### Causes
**Multiple testing**: Inflated Type I error rates
**P-hacking**: Selective reporting of significant results
**Publication bias**: Negative results not published
**Data dredging**: Testing many hypotheses post-hoc

#### Solutions
**Pre-registration**: Register analysis plans before data collection
**Multiple testing correction**: Control family-wise error rate
**Cross-validation**: Assess generalization performance
**Replication studies**: Independent validation of findings

## Best Practices

### Study Design
1. **Power analysis**: Determine adequate sample sizes
2. **Randomization**: Reduce confounding bias
3. **Blinding**: Prevent observation bias
4. **Controls**: Include appropriate negative/positive controls

### Data Analysis
1. **Exploratory analysis**: Understand data before formal analysis
2. **Model assumptions**: Check and validate assumptions
3. **Sensitivity analysis**: Test robustness to assumptions
4. **Multiple comparisons**: Adjust for multiple testing

### Reporting
1. **Methods transparency**: Provide sufficient detail for reproduction
2. **Code availability**: Share analysis scripts
3. **Data sharing**: Make data available when possible
4. **Effect sizes**: Report biological significance alongside statistical significance

### Validation
1. **Cross-validation**: Assess predictive performance
2. **External validation**: Test on independent datasets
3. **Experimental validation**: Confirm computational predictions
4. **Negative controls**: Include appropriate controls