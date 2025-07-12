# Bayesian Statistics

Bayesian statistics provides a framework for updating beliefs about parameters as new evidence becomes available, fundamentally different from frequentist approaches.

## Fundamental Concepts

### Bayes' Theorem

The foundation of Bayesian inference:

**P(θ|data) = P(data|θ) × P(θ) / P(data)**

| Component | Name | Description |
|-----------|------|-------------|
| P(θ\|data) | Posterior | Updated belief about θ after seeing data |
| P(data\|θ) | Likelihood | Probability of observing data given θ |
| P(θ) | Prior | Initial belief about θ before seeing data |
| P(data) | Marginal likelihood | Normalizing constant |

### Key Differences from Frequentist Statistics

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Parameters** | Fixed but unknown | Random variables with distributions |
| **Probability** | Long-run frequency | Degree of belief |
| **Inference** | Confidence intervals | Credible intervals |
| **Prior knowledge** | Not formally incorporated | Explicitly included |
| **Multiple testing** | Requires correction | Naturally accounts for multiplicity |

## Prior Distributions

### Types of Priors

#### Informative Priors
Incorporate substantial prior knowledge or expert opinion.

**Example**: Normal prior for gene expression fold changes
- Prior: θ ~ N(0, 1) 
- Interpretation: Most genes show modest expression changes

#### Non-informative (Vague) Priors
Express minimal prior knowledge, letting data dominate.

**Common non-informative priors**:
| Parameter | Non-informative Prior |
|-----------|----------------------|
| Mean (μ) | Uniform or Normal with large variance |
| Variance (σ²) | Inverse-Gamma with small parameters |
| Proportion (p) | Beta(1,1) = Uniform(0,1) |

#### Conjugate Priors
Mathematical convenience: posterior has same family as prior.

| Likelihood | Conjugate Prior | Posterior |
|------------|----------------|-----------|
| Binomial | Beta | Beta |
| Normal (known σ²) | Normal | Normal |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma |
| Poisson | Gamma | Gamma |

### Choosing Priors

```
Prior Selection Flowchart:

Strong prior knowledge? 
├─ Yes → Informative prior
└─ No → Weak prior knowledge?
   ├─ Yes → Weakly informative prior
   └─ No → Non-informative prior

Mathematical convenience needed?
├─ Yes → Consider conjugate priors
└─ No → Any appropriate distribution
```

## Bayesian Inference Methods

### Analytical Solutions

**Beta-Binomial Model** (Conjugate)

Given:
- Likelihood: X ~ Binomial(n, p)
- Prior: p ~ Beta(α, β)

Posterior: p|X ~ Beta(α + X, β + n - X)

**Example**: Gene variant frequency
- Prior: p ~ Beta(2, 8) [believe frequency is low]
- Data: 5 variants in 50 samples
- Posterior: p ~ Beta(7, 53)

### Markov Chain Monte Carlo (MCMC)

#### Metropolis-Hastings Algorithm

```
1. Start with initial parameter value θ₀
2. For each iteration t:
   a. Propose new value θ* from proposal distribution
   b. Calculate acceptance ratio: 
      α = min(1, [P(θ*|data) × q(θₜ₋₁|θ*)] / [P(θₜ₋₁|data) × q(θ*|θₜ₋₁)])
   c. Accept θ* with probability α, otherwise keep θₜ₋₁
3. After burn-in, use samples for inference
```

#### Gibbs Sampling
Special case where we sample from full conditional distributions.

**Advantages**:
- No tuning of proposal distributions
- High acceptance rate

**Requirements**:
- Full conditional distributions must be known
- Must be easy to sample from

### Variational Inference

Approximate posterior with simpler distribution by minimizing KL divergence.

**Advantages**:
- Faster than MCMC
- Scalable to large datasets
- Deterministic results

**Disadvantages**:
- Approximation may be poor
- Can underestimate uncertainty

## Bayesian Model Comparison

### Bayes Factors

Ratio of marginal likelihoods comparing two models:

**BF₁₂ = P(data|M₁) / P(data|M₂)**

| Bayes Factor | Evidence for M₁ |
|--------------|-----------------|
| > 100 | Decisive |
| 30-100 | Very strong |
| 10-30 | Strong |
| 3-10 | Substantial |
| 1-3 | Weak |
| < 1 | Evidence for M₂ |

### Information Criteria

#### Deviance Information Criterion (DIC)
**DIC = D̄ + pD**

Where:
- D̄ = posterior mean deviance
- pD = effective number of parameters

#### Widely Applicable Information Criterion (WAIC)
More general than DIC, works with non-exponential family models.

### Model Averaging
Instead of selecting single "best" model, average predictions across models weighted by posterior probabilities.

**Advantages**:
- Accounts for model uncertainty
- Often better predictive performance
- More honest about uncertainty

## Hierarchical Models

### Structure

```
Level 1 (Data): yᵢⱼ ~ Normal(θᵢ, σ²)
Level 2 (Group): θᵢ ~ Normal(μ, τ²)  
Level 3 (Hyperpriors): μ ~ Normal(0, 100²), τ ~ Half-Cauchy(0, 5)
```

### Advantages

| Benefit | Description |
|---------|-------------|
| **Shrinkage** | Extreme estimates pulled toward group mean |
| **Borrowing strength** | Information shared across groups |
| **Natural pooling** | Partial pooling between complete pooling and no pooling |
| **Uncertainty propagation** | Properly accounts for all sources of uncertainty |

### Example: Gene Expression Across Tissues

```
Expression model:
Level 1: log₂(expressionᵢⱼ) ~ Normal(αᵢ, σ²)    [i = gene, j = sample]
Level 2: αᵢ ~ Normal(μ_tissue, τ²)               [tissue-specific means]
Level 3: μ_tissue ~ Normal(0, 10²)               [overall expression level]
         τ ~ Half-Cauchy(0, 2)                   [between-gene variation]
         σ ~ Half-Cauchy(0, 1)                   [measurement error]
```

## Bayesian Linear Regression

### Model Specification

**Likelihood**: y ~ Normal(Xβ, σ²I)
**Priors**: 
- β ~ Normal(0, λI)
- σ² ~ Inverse-Gamma(a, b)

### Posterior Distribution

With conjugate priors, posterior is analytically tractable:

**β|y ~ Normal(μ_post, Σ_post)**

Where:
- Σ_post = (X'X/σ² + λ⁻¹I)⁻¹
- μ_post = Σ_post × X'y/σ²

### Regularization as Priors

| Frequentist Penalty | Bayesian Prior |
|---------------------|----------------|
| Ridge (L2) | Normal(0, λ) |
| Lasso (L1) | Laplace(0, λ) |
| Elastic Net | Normal + Laplace mixture |

### Credible Intervals vs Confidence Intervals

**95% Credible Interval**: 
"There is a 95% probability that the parameter lies in this interval"

**95% Confidence Interval**: 
"If we repeated this procedure many times, 95% of intervals would contain the true parameter"

## Bayesian Hypothesis Testing

### Approach 1: Bayes Factors

Compare evidence for competing hypotheses directly.

**Example**: Testing if gene expression differs between conditions
- H₁: μ₁ ≠ μ₂ (difference exists)
- H₀: μ₁ = μ₂ (no difference)
- BF₁₀ = P(data|H₁) / P(data|H₀)

### Approach 2: Posterior Probabilities

Calculate probability that effect exceeds threshold of practical significance.

**Example**: P(|effect| > 0.5 | data)

### Approach 3: Decision Theory

Minimize expected loss function.

| Decision | True State | Loss |
|----------|------------|------|
| Accept H₁ | H₁ true | 0 |
| Accept H₁ | H₀ true | L₁₀ |
| Accept H₀ | H₀ true | 0 |
| Accept H₀ | H₁ true | L₀₁ |

## Computational Tools

### Software Packages

| Software | Language | Strengths |
|----------|----------|-----------|
| **Stan** | R/Python | General purpose, HMC sampling |
| **JAGS** | R | Flexible, Gibbs sampling |
| **PyMC3/4** | Python | Pythonic, variational inference |
| **BUGS/WinBUGS** | R | Pioneer, many examples |
| **brms** | R | High-level interface to Stan |
| **rstanarm** | R | Pre-compiled Stan models |

### Diagnostic Tools

#### Convergence Diagnostics

| Metric | Purpose | Good Values |
|--------|---------|-------------|
| **R̂ (Rhat)** | Between/within chain variance | < 1.01 |
| **ESS** | Effective sample size | > 400 |
| **Trace plots** | Visual chain mixing | Well-mixed chains |
| **Autocorrelation** | Sample independence | Low autocorrelation |

#### Model Checking

**Posterior Predictive Checks**:
1. Generate replicated datasets from posterior
2. Compare to observed data
3. Look for systematic discrepancies

```R
# Example in R with brms
pp_check(model, nsamples = 100)
```

## Bioinformatics Applications

### Differential Gene Expression

**Traditional approach**: t-tests with multiple testing correction

**Bayesian approach**: Hierarchical model with shrinkage

```
Model:
log₂(FCᵢ) ~ Normal(μᵢ, σ²)        [i = gene]
μᵢ ~ Normal(0, τ²)                [shrinkage toward 0]

Advantages:
- Natural shrinkage for lowly expressed genes
- Borrows strength across genes  
- Uncertainty quantification
- No multiple testing correction needed
```

### Variant Calling

**Model**: 
- True genotype: G ~ Categorical(prior_frequencies)
- Observed reads: R|G ~ Binomial(coverage, error_rate(G))

**Output**: Posterior probability distribution over genotypes

### Phylogenetic Inference

**Bayesian phylogenetics**:
- Prior on tree topologies and branch lengths
- Likelihood based on sequence evolution model
- MCMC over tree space

**Advantages**:
- Uncertainty in tree topology
- Natural model comparison
- Integration over nuisance parameters

### Population Genetics

**Coalescent models**:
- Prior: Coalescent process with demographic parameters
- Likelihood: Observed genetic variation
- Inference: Population size changes, migration rates

### Genomic Prediction

**Bayesian whole-genome regression**:
```
Phenotype: y ~ Normal(Xβ, σ²ₑI)
Effects: βⱼ ~ π₀δ₀ + π₁Normal(0, σ²ᵦ)    [spike-and-slab prior]
```

**Features**:
- Variable selection through spike-and-slab
- Shrinkage of small effects
- Uncertainty in predictions

## Advanced Topics

### Non-parametric Bayesian Methods

#### Dirichlet Process
Infinite mixture models with unknown number of components.

**Applications**:
- Clustering with unknown number of clusters
- Density estimation
- Topic modeling in text analysis

#### Gaussian Processes
Non-parametric function estimation.

**Model**: f(x) ~ GP(m(x), k(x,x'))

**Applications**:
- Gene expression over time
- Spatial modeling of disease
- Optimization of experimental conditions

### Bayesian Deep Learning

**Bayesian Neural Networks**:
- Place priors on network weights
- Variational inference for posterior approximation
- Uncertainty quantification in predictions

**Applications**:
- Medical diagnosis with uncertainty
- Drug discovery
- Genomic sequence analysis

### Approximate Bayesian Computation (ABC)

When likelihood is intractable:

```
Algorithm:
1. Simulate data from model with parameters θ
2. Compare simulated to observed data
3. Accept θ if distance < tolerance
4. Repeat to build posterior sample
```

**Applications**:
- Population genetics models
- Epidemiological modeling
- Systems biology

## Case Study: Bayesian Analysis of RNA-seq Data

### Problem
Compare gene expression between cancer and normal tissues.

### Traditional Approach
1. Calculate fold changes and p-values
2. Apply multiple testing correction
3. Select genes with FDR < 0.05

### Bayesian Approach

#### Model
```
Level 1: log₂(count_ij) ~ Normal(μᵢⱼ, σ²)
Level 2: μᵢⱼ = αᵢ + βᵢ × I(condition_j = cancer)
Level 3: αᵢ ~ Normal(μ_α, τ²_α)           [baseline expression]
         βᵢ ~ Normal(0, τ²_β)              [log fold change]
Level 4: Hyperpriors on variance parameters
```

#### Implementation in Stan
```stan
data {
  int<lower=0> N;           // number of observations
  int<lower=0> G;           // number of genes
  vector[N] log_expression;
  int<lower=1,upper=G> gene[N];
  int<lower=0,upper=1> condition[N];
}

parameters {
  vector[G] alpha;          // baseline expression
  vector[G] beta;           // log fold changes
  real mu_alpha;
  real<lower=0> tau_alpha;
  real<lower=0> tau_beta;
  real<lower=0> sigma;
}

model {
  // Priors
  mu_alpha ~ normal(0, 10);
  tau_alpha ~ cauchy(0, 5);
  tau_beta ~ cauchy(0, 2);
  sigma ~ cauchy(0, 1);
  
  alpha ~ normal(mu_alpha, tau_alpha);
  beta ~ normal(0, tau_beta);
  
  // Likelihood
  for (n in 1:N) {
    log_expression[n] ~ normal(alpha[gene[n]] + beta[gene[n]] * condition[n], sigma);
  }
}

generated quantities {
  vector[G] prob_upregulated = to_vector(beta > log(2));    // P(FC > 2)
  vector[G] prob_downregulated = to_vector(beta < -log(2)); // P(FC < 0.5)
}
```

#### Advantages of Bayesian Approach

1. **Natural shrinkage**: Small effect sizes shrunk toward zero
2. **Borrowing strength**: Information shared across genes
3. **Uncertainty quantification**: Full posterior distribution for each gene
4. **No multiple testing**: Posterior probabilities naturally account for multiplicity
5. **Flexible thresholds**: Can ask about any effect size of interest

#### Results Interpretation

| Gene | Posterior Mean FC | 95% Credible Interval | P(FC > 2) | P(FC < 0.5) |
|------|-------------------|----------------------|-----------|-------------|
| GENE1 | 3.2 | [2.1, 4.8] | 0.95 | 0.01 |
| GENE2 | 1.1 | [0.8, 1.6] | 0.15 | 0.05 |
| GENE3 | 0.3 | [0.1, 0.7] | 0.02 | 0.85 |

**Interpretation**: 
- GENE1: Strong evidence of upregulation (95% probability of >2-fold increase)
- GENE2: Uncertain, modest effect
- GENE3: Strong evidence of downregulation

## Best Practices and Common Pitfalls

### Best Practices

1. **Prior sensitivity analysis**: Test different reasonable priors
2. **Model checking**: Use posterior predictive checks
3. **Convergence diagnostics**: Always check MCMC convergence
4. **Interpretable parameterization**: Use parameters that have biological meaning
5. **Report uncertainty**: Don't just report point estimates

### Common Pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| **Vague priors on variance** | Improper posteriors | Use proper priors (e.g., half-Cauchy) |
| **Ignoring convergence** | Unreliable results | Check R̂, effective sample size |
| **Over-fitting** | Poor generalization | Use regularizing priors, cross-validation |
| **Misspecified likelihood** | Biased inference | Model checking, robust models |
| **Computational shortcuts** | Approximation errors | Understand limitations of methods |

### Model Building Strategy

```
1. Start simple → Add complexity gradually
2. Check identifiability → Ensure parameters can be estimated
3. Prior predictive checks → Do priors give reasonable predictions?
4. Fit model → Run MCMC or variational inference  
5. Convergence diagnostics → Check mixing, R̂, ESS
6. Posterior predictive checks → Does model fit data well?
7. Sensitivity analysis → How robust are results to prior choice?
8. Biological interpretation → Do results make scientific sense?
```

## Resources for Further Learning

### Books
- **Gelman et al.**: "Bayesian Data Analysis" (comprehensive reference)
- **McElreath**: "Statistical Rethinking" (intuitive introduction)
- **Kruschke**: "Doing Bayesian Data Analysis" (gentle introduction)
- **Lambert**: "A Student's Guide to Bayesian Statistics" (accessible)

### Online Resources
- **Stan User's Guide**: https://mc-stan.org/docs/
- **Bayesian Analysis Recipes**: https://github.com/ericmjl/bayesian-analysis-recipes
- **Statistical Rethinking Course**: https://github.com/rmcelreath/stat_rethinking_2023

### Software Documentation
- **brms**: https://paul-buerkner.github.io/brms/
- **PyMC**: https://docs.pymc.io/
- **rstanarm**: https://mc-stan.org/rstanarm/

The Bayesian approach provides a coherent framework for incorporating prior knowledge, quantifying uncertainty, and making probabilistic statements about parameters of interest. While computationally more intensive than frequentist methods, modern software makes Bayesian analysis accessible for a wide range of bioinformatics applications.