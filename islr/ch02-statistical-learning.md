# Chapter 2: Statistical Learning

> *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani

## What is Statistical Learning?

Suppose we observe a quantitative response $Y$ and $p$ predictors $X_1, X_2, \ldots, X_p$. We assume:

$$Y = f(X) + \epsilon$$

where $f$ is a fixed but unknown function and $\epsilon$ is a random **error term** with mean zero, independent of $X$.

Statistical learning refers to approaches for estimating $f$.

---

## Why Estimate $f$?

### Prediction
When inputs $X$ are readily available but output $Y$ is not, we predict:
$$\hat{Y} = \hat{f}(X)$$

The expected squared prediction error decomposes as:

$$E\left[(Y - \hat{f}(X))^2\right] = \underbrace{\left[f(X) - \hat{f}(X)\right]^2}_{\text{Reducible}} + \underbrace{\text{Var}(\epsilon)}_{\text{Irreducible}}$$

The **irreducible error** $\text{Var}(\epsilon)$ is a floor — no matter how good $\hat{f}$ is, prediction error cannot go below this.

### Inference
When we want to understand **how** $Y$ is affected by changes in $X_1, \ldots, X_p$:
- Which predictors are associated with the response?
- What is the relationship between response and each predictor?
- Can the relationship be summarized linearly?

---

## Parametric vs Non-Parametric Methods

### Parametric (Model-based)
1. Assume a functional form for $f$, e.g. linearity: $f(X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$
2. Use training data to **fit** (train) the model
3. **Advantage**: reduces estimation problem to fitting a set of parameters
4. **Disadvantage**: model may be far from true $f$; overfitting risk

### Non-Parametric
- Make no explicit assumptions about the form of $f$
- Seek an estimate that fits data closely without being too rough/wiggly
- **Advantage**: can fit a wider range of shapes for $f$
- **Disadvantage**: need many more observations; risk of overfitting

---

## The Bias-Variance Trade-off

For a test observation $x_0$, the expected test MSE decomposes as:

$$E\left[(y_0 - \hat{f}(x_0))^2\right] = \underbrace{\text{Var}(\hat{f}(x_0))}_{\text{Variance}} + \underbrace{\left[\text{Bias}(\hat{f}(x_0))\right]^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\epsilon)}_{\text{Irreducible}}$$

| Term | Meaning | How to reduce |
|------|---------|---------------|
| **Variance** | How much $\hat{f}$ changes with different training sets | Use simpler model (more bias) |
| **Bias** | Error from wrong assumptions in learning algorithm | Use more flexible model (more variance) |
| **Irreducible** | Variance of $\epsilon$ | Cannot be reduced |

```
Test MSE
   ^
   |   \         /
   |    \       / (flexible model overfits)
   |     \_____/
   |
   +---------------------> Model Flexibility

   Low flexibility → high bias, low variance
   High flexibility → low bias, high variance
```

**Key insight**: as model flexibility increases, bias decreases but variance increases. The optimal model balances these.

---

## The Classification Setting

When $Y$ is qualitative (categorical), the **training error rate** is:

$$\frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(y_i \neq \hat{y}_i)$$

### Bayes Classifier

The **Bayes classifier** assigns each observation to the most likely class:

$$\Pr(Y = j \mid X = x_0)$$

Assign to class $j$ with the largest conditional probability.

- **Bayes error rate**: $1 - E\left[\max_j \Pr(Y=j \mid X)\right]$ — the irreducible error for classification
- In practice the Bayes classifier is unknown (we don't know the true conditional distributions)

### K-Nearest Neighbors (KNN)

A simple non-parametric classifier that approximates the Bayes classifier:

1. Find $K$ training points closest to $x_0$ (the neighborhood $\mathcal{N}_0$)
2. Estimate conditional probability as fraction of neighbors in class $j$:

$$\Pr(Y = j \mid X = x_0) = \frac{1}{K}\sum_{i \in \mathcal{N}_0} \mathbf{1}(y_i = j)$$

3. Classify $x_0$ to class with highest estimated probability

```r
library(class)
knn.pred <- knn(train = X_train, test = X_test, cl = y_train, k = 3)
```

| $K$ | Flexibility | Bias | Variance |
|-----|-------------|------|----------|
| Small (e.g. 1) | High | Low | High |
| Large (e.g. 100) | Low | High | Low |

**Rule of thumb**: Choose $K$ by cross-validation (Chapter 5).

---

## Flexibility vs Interpretability

```
High Interpretability          Low Interpretability
        |                                |
   Subset Selection              Deep Learning
   Lasso                         Boosting
   Linear Regression             Bagging/RF
   GAMs                          SVM
        |________________________|
   Low Flexibility            High Flexibility
```

**For inference**: prefer interpretable models (linear regression, lasso)
**For prediction accuracy**: more flexible models often win

---

## Supervised vs Unsupervised

| | Supervised | Unsupervised |
|--|-----------|--------------|
| Response $Y$? | Yes | No |
| Goal | Predict/model $Y$ from $X$ | Discover structure in $X$ |
| Examples | Regression, classification | PCA, clustering |
| Evaluation | Test error, accuracy | Visual inspection, internal metrics |

---

## Example: Flexibility and Test MSE

```r
# Simulated example: true f is non-linear
set.seed(1)
x <- seq(0, 1, length.out = 100)
y <- sin(2*pi*x) + rnorm(100, sd = 0.3)

# Fit models of increasing flexibility
fit1 <- lm(y ~ x)              # linear (low flex)
fit5 <- lm(y ~ poly(x, 5))     # degree-5 polynomial (medium flex)
fit15 <- lm(y ~ poly(x, 15))   # degree-15 polynomial (high flex)

# Training MSE decreases; test MSE is U-shaped
```

---

*Previous: [Chapter 1 — Introduction](ch01-introduction.md) | Next: [Chapter 3 — Linear Regression](ch03-linear-regression.md)*
