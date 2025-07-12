# Foundations of Statistics

## What is Statistics?

Statistics is the science of collecting, organizing, analyzing, interpreting, and presenting data. It provides tools and methods to make sense of uncertainty and variability in data, enabling evidence-based decision making.

### Key Branches of Statistics

1. **Descriptive Statistics**: Summarizing and describing data
2. **Inferential Statistics**: Making predictions and inferences about populations based on samples
3. **Bayesian Statistics**: Updating beliefs based on new evidence
4. **Mathematical Statistics**: Theoretical foundations and mathematical proofs

## Fundamental Concepts

### Population vs Sample

- **Population**: The complete set of all individuals or items of interest
- **Sample**: A subset of the population used to make inferences about the entire population
- **Parameter**: A numerical characteristic of a population (μ, σ)
- **Statistic**: A numerical characteristic of a sample (x̄, s)

### Types of Variables

#### By Nature

| Variable Type | Subtype | Description | Examples |
|---------------|---------|-------------|---------|
| **Quantitative (Numerical)** | Continuous | Can take any value within a range | Height, weight, temperature, gene expression levels |
| | Discrete | Countable values | Number of children, mutations, SNPs |
| **Qualitative (Categorical)** | Nominal | No natural order | Blood type (A,B,AB,O), species, treatment group |
| | Ordinal | Natural order exists | Disease severity (mild, moderate, severe), tumor stage |

```
Variable Classification Tree:

Variable
├── Quantitative (Numerical)
│   ├── Discrete (countable)
│   │   └── Examples: # of mutations, # of genes
│   └── Continuous (measurable) 
│       └── Examples: expression level, protein concentration
└── Qualitative (Categorical)
    ├── Nominal (no order)
    │   └── Examples: genotype, tissue type
    └── Ordinal (ordered)
        └── Examples: cancer stage, pain scale
```

#### By Role in Analysis
- **Independent Variable**: The predictor or explanatory variable
- **Dependent Variable**: The outcome or response variable

### Scales of Measurement

| Scale | Properties | Mathematical Operations | Examples |
|-------|-----------|------------------------|----------|
| **Nominal** | Categories, no order | = , ≠ | Blood type (A,B,AB,O), genotype (AA,Aa,aa) |
| **Ordinal** | Categories with order | = , ≠ , < , > | Pain scale (1-10), tumor grade (I,II,III,IV) |
| **Interval** | Equal intervals, no true zero | +, -, =, ≠, <, > | Temperature (°C), standardized test scores |
| **Ratio** | Equal intervals, true zero | ×, ÷, +, -, =, ≠, <, > | Age, height, gene expression, concentration |

**Visual Representation:**
```
Nominal:    [A] [B] [AB] [O]           (categories only)
Ordinal:    [Low] < [Med] < [High]      (order matters)
Interval:   |----20°C----30°C----40°C|  (equal intervals)
Ratio:      0kg---10kg---20kg---30kg   (true zero point)
```

## Data Collection Methods

### Experimental Design
- **Randomized Controlled Trial (RCT)**: Gold standard for causal inference
- **Observational Studies**: No intervention by researcher
- **Cross-sectional**: Snapshot at one time point
- **Longitudinal**: Following subjects over time
- **Case-control**: Comparing cases with disease to controls without

### Sampling Methods

| Method | Description | Advantages | Disadvantages | Best Used When |
|--------|-------------|------------|---------------|----------------|
| **Simple Random** | Each individual has equal probability | Unbiased, easy to analyze | May miss important subgroups | Population is homogeneous |
| **Stratified** | Population divided into strata, sample from each | Ensures representation | Requires knowledge of strata | Distinct subgroups exist |
| **Systematic** | Every nth individual selected | Simple to implement | Bias if pattern in population | Sampling frame is available |
| **Cluster** | Groups (clusters) randomly selected | Cost-effective for dispersed populations | Higher sampling error | Natural clusters exist |
| **Convenience** | Non-random, based on availability | Quick and inexpensive | High bias potential | Exploratory studies only |

**Sampling Illustration:**
```
Population: [●●●●●●●●●●●●●●●●●●●●] (20 individuals)

Simple Random:     [●○●○○●○●○●○○●○●○○●○●]
Stratified:        Stratum A [●○●○●] Stratum B [○●○●○] 
Systematic (n=5):  [●○○○○●○○○○●○○○○●○○○○]
Cluster:           Cluster 1 [●●●●●] Cluster 3 [●●●●●]
```

## Common Statistical Distributions

### Discrete Distributions

| Distribution | Parameters | Mean | Variance | Use Cases |
|--------------|------------|------|----------|----------|
| **Binomial** | n (trials), p (success prob) | np | np(1-p) | SNP calling, mutation counting |
| **Poisson** | λ (rate) | λ | λ | Gene expression counts, variant calls |
| **Geometric** | p (success prob) | 1/p | (1-p)/p² | Time to first success |
| **Negative Binomial** | r, p | r(1-p)/p | r(1-p)/p² | Overdispersed count data |

### Continuous Distributions

| Distribution | Parameters | Mean | Variance | Shape | Use Cases |
|--------------|------------|------|----------|-------|----------|
| **Normal** | μ (mean), σ² (variance) | μ | σ² | Bell-shaped, symmetric | Gene expression, measurement error |
| **Student's t** | ν (degrees of freedom) | 0 (if ν>1) | ν/(ν-2) | Bell-shaped, heavy tails | Small sample inference |
| **Chi-square** | ν (degrees of freedom) | ν | 2ν | Right-skewed | Goodness-of-fit, variance tests |
| **F** | ν₁, ν₂ (degrees of freedom) | ν₂/(ν₂-2) | Complex | Right-skewed | ANOVA, regression F-tests |
| **Beta** | α, β (shape parameters) | α/(α+β) | αβ/[(α+β)²(α+β+1)] | Flexible [0,1] | Proportions, Bayesian priors |

**Distribution Shapes:**
```
Normal:       /‾‾‾‾‾‾‾‾‾‾‾‾‾\        (symmetric bell)
t-distribution: /‾‾‾‾‾‾‾‾‾‾‾‾‾\      (heavier tails)
Chi-square:   /‾‾‾‾‾‾‾‾\___        (right-skewed)
F:            /‾‾‾‾‾‾\____         (right-skewed)
Beta:         Variable shapes depending on α,β
```

## Central Limit Theorem

One of the most important theorems in statistics:

> As sample size increases, the sampling distribution of the sample mean approaches a normal distribution, regardless of the shape of the population distribution.

**Implications**:
- Sample means are approximately normal for n ≥ 30
- Enables inference about population means
- Foundation for many statistical procedures

## Statistical Software and Tools

### Popular Software
- **R**: Open-source, extensive statistical packages
- **Python**: scikit-learn, pandas, numpy, scipy
- **SAS**: Commercial, widely used in pharma/biotech
- **SPSS**: User-friendly interface
- **Stata**: Popular in economics and epidemiology

### Bioinformatics-Specific Tools
- **Bioconductor**: R packages for bioinformatics
- **Galaxy**: Web-based platform
- **GATK**: Genomic analysis toolkit
- **Cytoscape**: Network analysis and visualization