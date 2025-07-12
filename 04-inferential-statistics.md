# Inferential Statistics

Inferential statistics uses sample data to make generalizations and predictions about populations, testing hypotheses and estimating parameters.

## Sampling Distributions

### Concept
A sampling distribution is the probability distribution of a statistic (like the sample mean) across all possible samples of a given size from a population.

### Key Properties
- **Mean of sampling distribution of X̄**: μₓ̄ = μ
- **Standard deviation of sampling distribution of X̄**: σₓ̄ = σ/√n (standard error)
- **Shape**: Approaches normal as sample size increases (CLT)

### Standard Error
The standard deviation of a sampling distribution.

**For sample mean**: SE = σ/√n
**For sample proportion**: SE = √[p(1-p)/n]

## Confidence Intervals

### Definition
A range of values that likely contains the true population parameter with a specified level of confidence.

### Interpretation
A 95% confidence interval means that if we repeated the sampling process many times, 95% of the intervals would contain the true parameter.

### General Formula
**Point Estimate ± (Critical Value × Standard Error)**

### Confidence Interval for Population Mean

#### When σ is Known (Z-interval)
**X̄ ± z_(α/2) × (σ/√n)**

**Common Critical Values:**

| Confidence Level | α | α/2 | z-value | t-value (df=10) | t-value (df=30) |
|------------------|---|-----|---------|-----------------|------------------|
| 90% | 0.10 | 0.05 | 1.645 | 1.812 | 1.697 |
| 95% | 0.05 | 0.025 | 1.96 | 2.228 | 2.042 |
| 99% | 0.01 | 0.005 | 2.576 | 3.169 | 2.750 |

**Confidence Interval Interpretation:**
```
95% CI: [L, U]

Correct: "I am 95% confident the true parameter lies in [L, U]"
Incorrect: "There is a 95% chance the parameter lies in [L, U]"

Visualization:
    L────────────X────────────U
    ↑            ↑            ↑
  Lower       Point        Upper
  Bound      Estimate      Bound
```

#### When σ is Unknown (t-interval)
**X̄ ± t_(α/2,df) × (s/√n)**

where df = n - 1

**Conditions**:
- Random sample
- Normal population or large sample (n ≥ 30)
- Independent observations

### Confidence Interval for Population Proportion
**p̂ ± z_(α/2) × √[p̂(1-p̂)/n]**

**Conditions**:
- Random sample
- np̂ ≥ 10 and n(1-p̂) ≥ 10
- Independent observations

### Factors Affecting CI Width
1. **Confidence level**: Higher confidence → wider interval
2. **Sample size**: Larger n → narrower interval
3. **Population variability**: Higher σ → wider interval

## Hypothesis Testing

### Basic Concepts

#### Hypotheses
- **Null hypothesis (H₀)**: Statement of no effect or no difference
- **Alternative hypothesis (H₁ or Hₐ)**: Statement we're trying to prove

#### Types of Tests
- **Two-tailed**: H₁: μ ≠ μ₀
- **Right-tailed**: H₁: μ > μ₀
- **Left-tailed**: H₁: μ < μ₀

### Test Statistics
A standardized value that measures how far the sample statistic is from the hypothesized parameter.

**For means**: t = (X̄ - μ₀) / (s/√n)
**For proportions**: z = (p̂ - p₀) / √[p₀(1-p₀)/n]

### P-value
The probability of observing a test statistic as extreme or more extreme than the one calculated, assuming H₀ is true.

**Decision Rule**:
- If p-value ≤ α: Reject H₀
- If p-value > α: Fail to reject H₀

### Significance Level (α)
The probability of making a Type I error (rejecting a true H₀).

**Common levels**: α = 0.05, 0.01, 0.10

### Types of Errors

#### Type I Error (α)
Rejecting H₀ when it's actually true (false positive).

#### Type II Error (β)
Failing to reject H₀ when it's actually false (false negative).

#### Power (1 - β)
The probability of correctly rejecting a false H₀.

**Factors affecting power**:
- Effect size (larger effect → higher power)
- Sample size (larger n → higher power)
- Significance level (larger α → higher power)
- Population variability (smaller σ → higher power)

### Steps in Hypothesis Testing
1. State hypotheses (H₀ and H₁)
2. Choose significance level (α)
3. Check assumptions
4. Calculate test statistic
5. Find p-value
6. Make decision
7. State conclusion in context

## Common Statistical Tests

### One-Sample Tests

#### One-Sample t-Test
Tests whether a population mean equals a specific value.

**Hypotheses**:
- H₀: μ = μ₀
- H₁: μ ≠ μ₀ (or μ > μ₀ or μ < μ₀)

**Test statistic**: t = (X̄ - μ₀) / (s/√n)
**Degrees of freedom**: df = n - 1

#### One-Sample z-Test for Proportion
Tests whether a population proportion equals a specific value.

**Hypotheses**:
- H₀: p = p₀
- H₁: p ≠ p₀ (or p > p₀ or p < p₀)

**Test statistic**: z = (p̂ - p₀) / √[p₀(1-p₀)/n]

### Two-Sample Tests

#### Two-Sample t-Test
Compares means of two independent groups.

**Equal variances (pooled t-test)**:
t = (X̄₁ - X̄₂) / (sₚ√(1/n₁ + 1/n₂))

where sₚ = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]

**Unequal variances (Welch's t-test)**:
t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

#### Paired t-Test
Compares means of paired observations (before/after, matched subjects).

**Test statistic**: t = d̄ / (sₐ/√n)

where d̄ is the mean difference and sₐ is the standard deviation of differences.

#### Two-Sample z-Test for Proportions
Compares proportions between two independent groups.

**Test statistic**: z = (p̂₁ - p̂₂) / √[p̂(1-p̂)(1/n₁ + 1/n₂)]

where p̂ = (x₁ + x₂) / (n₁ + n₂) is the pooled proportion.

### Chi-Square Tests

#### Chi-Square Goodness of Fit
Tests whether sample data fits a specific distribution.

**Test statistic**: χ² = Σ[(Observed - Expected)² / Expected]
**Degrees of freedom**: k - 1 - (number of estimated parameters)

#### Chi-Square Test of Independence
Tests whether two categorical variables are independent.

**Test statistic**: χ² = Σ[(Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ]
**Degrees of freedom**: (r - 1)(c - 1)

where Eᵢⱼ = (row total × column total) / grand total

### Analysis of Variance (ANOVA)

#### One-Way ANOVA
Compares means across multiple groups.

**Hypotheses**:
- H₀: μ₁ = μ₂ = ... = μₖ
- H₁: At least one mean differs

**Test statistic**: F = MSB / MSW

**Components**:
- SST (Total Sum of Squares)
- SSB (Between-group Sum of Squares)
- SSW (Within-group Sum of Squares)
- MSB = SSB / (k-1)
- MSW = SSW / (N-k)

#### Two-Way ANOVA
Tests effects of two factors and their interaction.

**Sources of variation**:
- Main effect of Factor A
- Main effect of Factor B
- Interaction effect A×B
- Error

### Non-Parametric Tests

#### Mann-Whitney U Test (Wilcoxon Rank Sum)
Non-parametric alternative to two-sample t-test.

**Use when**:
- Data not normally distributed
- Ordinal data
- Small sample sizes

#### Wilcoxon Signed-Rank Test
Non-parametric alternative to paired t-test.

#### Kruskal-Wallis Test
Non-parametric alternative to one-way ANOVA.

#### Spearman's Rank Correlation
Non-parametric measure of association between two variables.

## Multiple Testing Correction

### Problem
When performing multiple tests, the probability of making at least one Type I error increases.

**Family-wise error rate**: FWER = 1 - (1 - α)ᵐ

where m is the number of tests.

### Correction Methods

#### Bonferroni Correction
**Adjusted α**: α' = α / m

**Conservative but simple**

#### False Discovery Rate (FDR)
Controls the expected proportion of false discoveries among rejected hypotheses.

**Benjamini-Hochberg procedure**:
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest k such that pₖ ≤ (k/m) × α
3. Reject H₀ for tests 1, 2, ..., k

#### Holm-Bonferroni
Step-down procedure that's less conservative than Bonferroni.

## Effect Size and Practical Significance

### Why Effect Size Matters

**Statistical vs Practical Significance:**

| Sample Size | Effect | p-value | Practical Importance |
|-------------|--------|---------|---------------------|
| n = 10,000 | Tiny | p < 0.001 | Statistically significant but meaningless |
| n = 20 | Large | p = 0.08 | Not significant but potentially important |

### Effect Size Measures

| Measure | Formula | Small | Medium | Large | Use Case |
|---------|---------|-------|--------|-------|----------|
| **Cohen's d** | (μ₁ - μ₂) / σ | 0.2 | 0.5 | 0.8 | Two-group comparisons |
| **Pearson's r** | Correlation coefficient | 0.1 | 0.3 | 0.5 | Associations |
| **Eta-squared (η²)** | SSB / SST | 0.01 | 0.06 | 0.14 | ANOVA (variance explained) |
| **Omega-squared (ω²)** | (SSB - (k-1)MSW) / (SST + MSW) | 0.01 | 0.06 | 0.14 | Unbiased η² |
| **Phi (φ)** | χ² / n | 0.1 | 0.3 | 0.5 | 2×2 contingency tables |

### Interpreting Effect Sizes in Bioinformatics

| Context | Measure | Small Effect | Large Effect |
|---------|---------|--------------|-------------|
| **Gene Expression** | Log₂ fold change | ±0.5 | ±2.0 |
| **GWAS** | Odds ratio | 1.1-1.2 | >2.0 |
| **Drug Response** | Response rate difference | 5% | 20% |
| **Survival Analysis** | Hazard ratio | 1.1-1.3 | >2.0 |

**Visual Effect Size Comparison:**
```
Cohen's d = 0.2 (Small):
Group 1: ●●●●●●●●●●
Group 2:   ●●●●●●●●●●  (slight shift)

Cohen's d = 0.8 (Large):
Group 1: ●●●●●●●●●●
Group 2:       ●●●●●●●●●●  (clear separation)
```

## Sample Size and Power Analysis

### Power Analysis Components

| Component | Symbol | Typical Values | Notes |
|-----------|--------|----------------|-------|
| **Power** | 1-β | 0.80, 0.90 | Probability of detecting true effect |
| **Significance Level** | α | 0.05, 0.01 | Type I error rate |
| **Effect Size** | δ | Varies | Magnitude of difference to detect |
| **Variance** | σ² | From pilot data | Population variability |

### Sample Size Formulas

#### Two-Sample Comparison
**n = 2(z_(α/2) + z_β)² × σ² / (μ₁ - μ₂)²**

#### One-Sample t-test
**n = (z_(α/2) + z_β)² × σ² / (μ - μ₀)²**

#### Proportion Test
**n = (z_(α/2) + z_β)² × [p₁(1-p₁) + p₂(1-p₂)] / (p₁ - p₂)²**

### Power Analysis Example

**Study Design:** Compare gene expression between cases and controls
- Desired power: 80%
- Significance level: 5%
- Expected difference: 1.5 units
- Standard deviation: 2.0 units

**Calculation:**
n = 2(1.96 + 0.84)² × 2² / 1.5² = 2 × 7.84 × 4 / 2.25 = **28 per group**

### Sample Size Table (Two-group t-test, α=0.05)

| Effect Size (d) | Power = 0.80 | Power = 0.90 |
|-----------------|--------------|---------------|
| 0.2 (Small) | 393 per group | 526 per group |
| 0.5 (Medium) | 64 per group | 85 per group |
| 0.8 (Large) | 26 per group | 34 per group |
| 1.0 (Very Large) | 17 per group | 22 per group |

## Bioinformatics Applications

### Differential Gene Expression
- **t-tests**: Comparing expression between conditions
- **Multiple testing correction**: FDR for thousands of genes
- **Effect size**: Log fold change

### GWAS (Genome-Wide Association Studies)
- **Chi-square tests**: Association between SNPs and traits
- **Bonferroni correction**: Conservative approach for millions of SNPs
- **Power calculations**: Sample size for detecting small effects

### Phylogenetic Analysis
- **Bootstrap confidence intervals**: Branch support
- **Likelihood ratio tests**: Model comparison

### Quality Control
- **Control charts**: Monitoring sequencing metrics
- **Hypothesis testing**: Detecting batch effects

### Experimental Design
- **Power analysis**: Determining sample sizes for experiments
- **ANOVA**: Comparing multiple treatments or conditions
- **Blocking**: Accounting for known sources of variation

## Common Mistakes and Considerations

1. **Multiple testing**: Always correct for multiple comparisons
2. **Assumptions**: Check normality, independence, equal variances
3. **Effect size**: Report alongside statistical significance
4. **Practical significance**: Consider biological meaning
5. **Sample size**: Ensure adequate power
6. **Data dredging**: Avoid fishing for significant results
7. **Publication bias**: Consider negative results