# Probability Theory

Probability theory provides the mathematical foundation for statistical inference, quantifying uncertainty and randomness.

## Basic Probability Concepts

### Definition of Probability
Probability measures the likelihood of an event occurring, expressed as a number between 0 and 1.

**Interpretations**:
1. **Classical**: P(A) = (Number of favorable outcomes) / (Total number of possible outcomes)
2. **Frequentist**: Long-run relative frequency of an event
3. **Subjective**: Personal degree of belief about an event

### Sample Space and Events
- **Sample Space (Ω)**: Set of all possible outcomes
- **Event**: Subset of the sample space
- **Elementary Event**: Single outcome
- **Compound Event**: Collection of elementary events

**Example**: Rolling a die
- Sample space: Ω = {1, 2, 3, 4, 5, 6}
- Event A = "even number": {2, 4, 6}
- P(A) = 3/6 = 0.5

## Probability Rules and Axioms

### Axioms of Probability (Kolmogorov)
1. **Non-negativity**: P(A) ≥ 0 for any event A
2. **Normalization**: P(Ω) = 1
3. **Additivity**: For mutually exclusive events: P(A ∪ B) = P(A) + P(B)

### Basic Rules

#### Complement Rule
P(A') = 1 - P(A)
where A' is the complement of A

#### Addition Rule
- **General**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- **Mutually exclusive events**: P(A ∪ B) = P(A) + P(B)

#### Multiplication Rule
- **General**: P(A ∩ B) = P(A) × P(B|A) = P(B) × P(A|B)
- **Independent events**: P(A ∩ B) = P(A) × P(B)

## Conditional Probability

### Definition
The probability of event A given that event B has occurred:

**P(A|B) = P(A ∩ B) / P(B)**, provided P(B) > 0

### Independence
Events A and B are independent if:
P(A|B) = P(A) or equivalently P(A ∩ B) = P(A) × P(B)

### Law of Total Probability
If B₁, B₂, ..., Bₙ form a partition of the sample space:

**P(A) = Σ P(A|Bᵢ) × P(Bᵢ)**

### Bayes' Theorem
Fundamental theorem for updating probabilities with new evidence:

**P(A|B) = P(B|A) × P(A) / P(B)**

**Components**:
- P(A|B): Posterior probability
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Marginal probability

**Example in Medical Testing**:
- Disease prevalence: P(D) = 0.01
- Test sensitivity: P(+|D) = 0.95
- Test specificity: P(-|D') = 0.99
- P(D|+) = P(+|D) × P(D) / P(+) = 0.95 × 0.01 / 0.0194 ≈ 0.49

## Random Variables

### Definition
A function that assigns numerical values to outcomes of a random experiment.

### Types
- **Discrete**: Countable values (number of mutations, allele count)
- **Continuous**: Uncountable values (gene expression, concentration)

### Probability Distributions

#### Discrete Random Variables
- **Probability Mass Function (PMF)**: P(X = x)
- **Properties**: Σ P(X = x) = 1, P(X = x) ≥ 0

#### Continuous Random Variables
- **Probability Density Function (PDF)**: f(x)
- **Properties**: ∫ f(x)dx = 1, f(x) ≥ 0
- **Note**: P(X = x) = 0 for any specific value x

### Cumulative Distribution Function (CDF)
F(x) = P(X ≤ x)

**Properties**:
- Non-decreasing
- F(-∞) = 0, F(∞) = 1
- Right-continuous

## Expected Value and Variance

### Expected Value (Mean)
The average value of a random variable over many trials.

**Discrete**: E[X] = Σ x × P(X = x)
**Continuous**: E[X] = ∫ x × f(x)dx

### Properties of Expected Value
- E[aX + b] = aE[X] + b
- E[X + Y] = E[X] + E[Y]
- If X and Y are independent: E[XY] = E[X]E[Y]

### Variance
Measure of spread around the mean.

**Definition**: Var(X) = E[(X - μ)²] = E[X²] - [E[X]]²

**Properties**:
- Var(aX + b) = a²Var(X)
- If X and Y are independent: Var(X + Y) = Var(X) + Var(Y)

### Standard Deviation
σ = √Var(X)

## Common Probability Distributions

### Discrete Distributions

#### Bernoulli Distribution
Single trial with two outcomes (success/failure).
- **Parameter**: p (probability of success)
- **PMF**: P(X = 1) = p, P(X = 0) = 1-p
- **Mean**: p
- **Variance**: p(1-p)

#### Binomial Distribution
Number of successes in n independent Bernoulli trials.
- **Parameters**: n (trials), p (probability of success)
- **PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- **Mean**: np
- **Variance**: np(1-p)

**Example**: Number of mutation-carrying individuals in a sample

#### Poisson Distribution
Number of events in a fixed interval.
- **Parameter**: λ (rate parameter)
- **PMF**: P(X = k) = (λ^k × e^(-λ)) / k!
- **Mean**: λ
- **Variance**: λ

**Examples**: Number of mutations per genome, number of reads per gene

#### Geometric Distribution
Number of trials until first success.
- **Parameter**: p (probability of success)
- **PMF**: P(X = k) = (1-p)^(k-1) × p
- **Mean**: 1/p
- **Variance**: (1-p)/p²

### Continuous Distributions

#### Uniform Distribution
All values in an interval are equally likely.
- **Parameters**: a (minimum), b (maximum)
- **PDF**: f(x) = 1/(b-a) for a ≤ x ≤ b
- **Mean**: (a+b)/2
- **Variance**: (b-a)²/12

#### Normal Distribution
Bell-shaped, symmetric distribution.
- **Parameters**: μ (mean), σ² (variance)
- **PDF**: f(x) = (1/√(2πσ²)) × exp[-(x-μ)²/(2σ²)]
- **Properties**: 68-95-99.7 rule

**Standard Normal**: μ = 0, σ = 1

#### Exponential Distribution
Time between events in a Poisson process.
- **Parameter**: λ (rate parameter)
- **PDF**: f(x) = λe^(-λx) for x ≥ 0
- **Mean**: 1/λ
- **Variance**: 1/λ²

**Example**: Time between mutations

#### Beta Distribution
Continuous distribution on [0,1], often used for proportions.
- **Parameters**: α, β (shape parameters)
- **Mean**: α/(α+β)
- **Use**: Prior distribution for probabilities in Bayesian analysis

## Multivariate Distributions

### Joint Distributions
For multiple random variables X and Y:

**Joint PMF/PDF**: P(X = x, Y = y) or f(x,y)
**Marginal distributions**: P(X = x) = Σ P(X = x, Y = y)

### Independence
X and Y are independent if:
P(X = x, Y = y) = P(X = x) × P(Y = y)

### Covariance and Correlation
**Covariance**: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
**Correlation**: ρ = Cov(X,Y) / (σₓσᵧ)

**Properties of correlation**:
- -1 ≤ ρ ≤ 1
- ρ = 0 for independent variables (but not vice versa)
- |ρ| = 1 indicates perfect linear relationship

## Central Limit Theorem

### Statement
For large sample sizes, the distribution of sample means approaches normal distribution, regardless of the population distribution.

**Conditions**:
- Random sampling
- Independent observations
- Large sample size (typically n ≥ 30)

**Mathematical form**:
If X₁, X₂, ..., Xₙ are iid with mean μ and variance σ², then:

**(X̄ - μ) / (σ/√n) → N(0,1) as n → ∞**

### Applications
- Confidence intervals for means
- Hypothesis testing
- Quality control
- Approximating binomial with normal

## Law of Large Numbers

### Weak Law
Sample mean converges in probability to population mean:
**X̄ₙ →ᵖ μ as n → ∞**

### Strong Law
Sample mean converges almost surely to population mean:
**X̄ₙ →ᵃ·ˢ· μ as n → ∞**

## Bioinformatics Applications

### Hardy-Weinberg Equilibrium
Population genetics model using probability:
- **Allele frequencies**: p + q = 1
- **Genotype frequencies**: p² + 2pq + q² = 1

### Sequence Analysis
- **Markov chains**: Modeling DNA sequences
- **Hidden Markov Models**: Gene finding, sequence alignment
- **Poisson processes**: Mutation occurrence

### Phylogenetics
- **Substitution models**: Probability of nucleotide changes
- **Branch lengths**: Expected number of substitutions

### Expression Analysis
- **Negative binomial**: Modeling count data (RNA-seq)
- **Beta distribution**: Proportion of methylated sites
- **Mixture models**: Identifying differentially expressed genes