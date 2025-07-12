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
- **Quantitative (Numerical)**: 
  - Continuous: Can take any value within a range (height, weight, temperature)
  - Discrete: Countable values (number of children, number of mutations)

- **Qualitative (Categorical)**:
  - Nominal: No natural order (blood type, species, treatment group)
  - Ordinal: Natural order exists (disease severity: mild, moderate, severe)

#### By Role in Analysis
- **Independent Variable**: The predictor or explanatory variable
- **Dependent Variable**: The outcome or response variable

### Scales of Measurement

1. **Nominal**: Categories without order (A, B, AB, O blood types)
2. **Ordinal**: Categories with order (low, medium, high expression)
3. **Interval**: Equal intervals, no true zero (temperature in Celsius)
4. **Ratio**: Equal intervals with true zero (age, concentration, counts)

## Data Collection Methods

### Experimental Design
- **Randomized Controlled Trial (RCT)**: Gold standard for causal inference
- **Observational Studies**: No intervention by researcher
- **Cross-sectional**: Snapshot at one time point
- **Longitudinal**: Following subjects over time
- **Case-control**: Comparing cases with disease to controls without

### Sampling Methods
- **Simple Random Sampling**: Each individual has equal probability of selection
- **Stratified Sampling**: Population divided into strata, sample from each
- **Systematic Sampling**: Every nth individual selected
- **Cluster Sampling**: Groups (clusters) randomly selected
- **Convenience Sampling**: Non-random, based on availability

## Common Statistical Distributions

### Discrete Distributions
- **Binomial**: Number of successes in n trials
- **Poisson**: Number of events in fixed time/space
- **Geometric**: Number of trials until first success

### Continuous Distributions
- **Normal (Gaussian)**: Bell-shaped, symmetric
- **Student's t**: Similar to normal but heavier tails
- **Chi-square**: Used in goodness-of-fit tests
- **F-distribution**: Used in ANOVA and regression

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