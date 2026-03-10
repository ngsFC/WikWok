# Chapter 4: Classification

> *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani

## Overview

**Classification** predicts a qualitative (categorical) response $Y$ from predictors $X$.

Why not linear regression for classification?
- Coding $Y \in \{0, 1\}$ as numeric is arbitrary for $>2$ classes
- Predictions can fall outside $[0,1]$ — not valid probabilities
- Linear regression may work for binary response but logistic regression is preferred

---

## 4.1 Logistic Regression

### The Logistic Function

Model the **probability** that $Y = 1$ given $X$:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

This S-shaped (sigmoid) curve guarantees $p(X) \in [0,1]$.

### Log-Odds (Logit)

Taking the log of the odds:

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X$$

The left side is the **logit** — a linear function of $X$. Increasing $X$ by 1 unit changes log-odds by $\beta_1$.

### Estimating Coefficients — Maximum Likelihood

Minimize (maximize) the **log-likelihood**:

$$\ell(\beta_0, \beta_1) = \sum_{i:y_i=1} \log p(x_i) + \sum_{i:y_i=0} \log(1 - p(x_i))$$

**Example: Default dataset** — predict credit card default from account balance

**Table 4.1: Simple logistic regression (balance only)**

| Coefficient | Estimate | Std. Error | z-statistic | p-value |
|-------------|----------|------------|-------------|---------|
| Intercept | −10.6513 | 0.3612 | −29.5 | <0.0001 |
| balance | 0.0055 | 0.0002 | 24.9 | <0.0001 |

Interpretation: one-unit increase in balance increases log-odds of default by 0.0055.

**Table 4.2: Student indicator**

| Coefficient | Estimate | Std. Error | z-statistic | p-value |
|-------------|----------|------------|-------------|---------|
| Intercept | −3.5041 | 0.0707 | −49.55 | <0.0001 |
| student[Yes] | 0.4049 | 0.1150 | 3.52 | 0.0004 |

Students have higher default probability: $\hat{p} = 0.0431$ vs $0.0292$ for non-students.

### Multiple Logistic Regression

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Table 4.3: Multiple logistic (balance + income + student)**

| Coefficient | Estimate | Std. Error | z-statistic | p-value |
|-------------|----------|------------|-------------|---------|
| Intercept | −10.8690 | 0.4923 | −22.08 | <0.0001 |
| balance | 0.0057 | 0.0002 | 24.74 | <0.0001 |
| income | 0.0030 | 0.0082 | 0.37 | 0.7115 |
| student[Yes] | **−0.6468** | 0.2362 | −2.74 | 0.0062 |

> **Confounding**: Student coefficient **flips sign** in multiple regression. Students carry higher balance → correlated with balance → simple regression was confounded. At fixed balance level, students are less likely to default.

---

## 4.2 Linear Discriminant Analysis (LDA)

### Motivation

- Logistic regression can be unstable when classes are well-separated
- LDA is more stable with small $n$; works naturally for $>2$ classes
- Assumes predictors are **normally distributed within each class**

### Bayes Theorem for Classification

$$P(Y = k \mid X = x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^K \pi_l f_l(x)}$$

Where:
- $\pi_k$ = **prior probability** of class $k$ (proportion of training observations in class $k$)
- $f_k(x) = P(X = x \mid Y = k)$ = **density function** of $X$ within class $k$
- $p_k(x) = P(Y = k \mid X = x)$ = **posterior probability**

### LDA with One Predictor ($p = 1$)

Assume Gaussian densities with **equal variances**:

$$f_k(x) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left(-\frac{(x - \mu_k)^2}{2\sigma^2}\right)$$

Plugging into Bayes theorem → the **discriminant function**:

$$\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$

Assign observation to class with **largest $\delta_k(x)$**.

**Bayes decision boundary** (two classes): the point $x$ where $\delta_1(x) = \delta_2(x)$:

$$x = \frac{\mu_1^2 - \mu_2^2}{2(\mu_1 - \mu_2)} = \frac{\mu_1 + \mu_2}{2}$$

### LDA with Multiple Predictors ($p > 1$)

Assume $X = (X_1, \ldots, X_p)$ drawn from a **multivariate Gaussian** with class-specific means but **shared covariance matrix $\Sigma$**:

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)$$

**Decision boundaries are linear** in $x$ (hence "Linear" DA).

### LDA Performance: Confusion Matrix

**Example: Default dataset** ($n = 10,000$, 333 defaults)

|  | Predicted No | Predicted Yes |
|--|-------------|--------------|
| **Actual No** | 9,644 | 23 |
| **Actual Yes** | 252 | 81 |

- **Overall error rate**: 2.75%
- **False negative rate** (defaults predicted as no): 252/333 = **75.7%** — very high!
- **Specificity** (non-defaults correct): 9,644/9,667 = 99.8%

### Adjusting the Decision Threshold

By default, LDA uses threshold $p(X) = 0.5$. For imbalanced data (default is rare):

```r
# Lower threshold → more aggressive prediction of defaults
lda.pred.thresh0.2 <- ifelse(lda.pred$posterior[,2] > 0.2, "Yes", "No")
```

**Trade-off**: lowering threshold reduces false negatives (miss fewer defaults) but increases false positives.

### ROC Curve

Plots **True Positive Rate** vs **False Positive Rate** across all thresholds:

$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

- **AUC** (Area Under Curve): 0.5 = random; 1.0 = perfect classifier
- For Default dataset: LDA AUC ≈ **0.95** (excellent discrimination)

---

## 4.3 Quadratic Discriminant Analysis (QDA)

Like LDA but allows **each class to have its own covariance matrix $\Sigma_k$**:

$$\delta_k(x) = -\frac{1}{2} x^T \Sigma_k^{-1} x + x^T \Sigma_k^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma_k^{-1} \mu_k - \frac{1}{2} \log|\Sigma_k| + \log(\pi_k)$$

**Decision boundaries are quadratic** in $x$.

| Method | Covariance | Bias | Variance | Best when |
|--------|-----------|------|----------|-----------|
| LDA | Shared $\Sigma$ | Higher | Lower | $n$ small; true boundaries near linear |
| QDA | Per-class $\Sigma_k$ | Lower | Higher | $n$ large; true boundaries non-linear |

---

## 4.4 Naive Bayes

Assumes all $p$ predictors are **independent within each class**:

$$f_k(x) = \prod_{j=1}^p f_{kj}(x_j)$$

- Very efficient (estimates $K \times p$ 1D densities instead of full covariance)
- Surprisingly effective despite the independence assumption (often violated)
- **Gaussian Naive Bayes**: assume $f_{kj}$ is Gaussian
- **Multinomial NB**: for categorical predictors (text classification)

---

## 4.5 Comparison of Methods

| Method | Assumptions | # Parameters | Notes |
|--------|-------------|--------------|-------|
| Logistic Regression | No distributional assumption | $p+1$ | Robust; max-likelihood; works for binary/multinomial |
| LDA | Gaussian, shared $\Sigma$ | $Kp + p(p+1)/2$ | Stable; good for multi-class; handles separation |
| QDA | Gaussian, class-specific $\Sigma_k$ | $Kp(p+1)/2$ | Flexible boundaries; needs large $n$ |
| Naive Bayes | Independent predictors | $Kp$ | Very fast; good for text/high-dim |
| KNN | Non-parametric | $K$ | Most flexible; poor interpretability; curse of dim |

**Practical rules:**
- When decision boundary is linear → **LDA or logistic** regression
- When decision boundary is moderately non-linear → **QDA**
- When boundary is complex, $n$ large → **KNN** (choose $K$ by CV)
- When $p$ is large → **Naive Bayes** (avoids covariance matrix estimation)

---

## 4.6 Lab: Classification in R

### Logistic Regression — Stock Market Data (Smarket)

```r
library(ISLR)
# 1250 trading days, 2001-2005: Lag1-Lag5, Volume, Today, Direction (Up/Down)

# Fit model
glm.fits <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                data = Smarket, family = binomial)
summary(glm.fits)
# Lag1 coeff = -0.073 (p = 0.149) — none significant!

# Predict probabilities
glm.probs <- predict(glm.fits, type = "response")
glm.pred <- ifelse(glm.probs > 0.5, "Up", "Down")

# Confusion matrix
table(glm.pred, Smarket$Direction)
mean(glm.pred == Smarket$Direction)  # 0.5216 — barely above 50%!

# Train/test split: train on 2001-2004, test on 2005
train <- Smarket$Year < 2005
glm.fits2 <- glm(Direction ~ Lag1 + Lag2,  # use only 2 least correlated
                 data = Smarket, subset = train, family = binomial)
glm.probs2 <- predict(glm.fits2, Smarket[!train,], type = "response")
glm.pred2 <- ifelse(glm.probs2 > 0.5, "Up", "Down")
mean(glm.pred2 == Smarket$Direction[!train])  # 0.5595 — better than chance
```

### LDA — Stock Market Data

```r
library(MASS)
lda.fit <- lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
lda.pred <- predict(lda.fit, Smarket[!train,])
# lda.pred$class: Up/Down
# lda.pred$posterior: posterior probabilities

table(lda.pred$class, Direction.2005)
mean(lda.pred$class == Direction.2005)  # 0.5595 — same as logistic here
```

### QDA — Stock Market Data

```r
qda.fit <- qda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
qda.pred <- predict(qda.fit, Smarket[!train,])$class
mean(qda.pred == Direction.2005)  # 0.5992 — best of the three!
```

### KNN Classification

```r
library(class)
# k=1
knn.pred <- knn(train = cbind(Lag1, Lag2)[train,],
                test  = cbind(Lag1, Lag2)[!train,],
                cl    = Direction[train], k = 1)
mean(knn.pred == Direction.2005)  # 0.5 — no better than random (overfit)

# k=3 improves
knn.pred3 <- knn(..., k = 3)
mean(knn.pred3 == Direction.2005)  # 0.5317
```

### Caravan Insurance Dataset

```r
# Highly imbalanced: 5.9% bought insurance
# Standardize predictors! (KNN is distance-based)
standardized.X <- scale(Caravan[,-86])

# Split train/test
test <- 1:1000
knn.pred <- knn(train = standardized.X[-test,],
                test  = standardized.X[test,],
                cl    = Purchase[-test], k = 5)
# Among predicted buyers: 26.7% actually bought — vs 5.9% baseline
# Success rate: 11.7% (k=3)
```

---

## Summary

| Method | Key Idea | Formula/Rule |
|--------|---------|-------------|
| Logistic regression | Model $p(X)$ as sigmoid | $\log(p/(1-p)) = \beta^T X$ |
| LDA | Gaussian class densities, shared $\Sigma$ | $\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$ |
| QDA | Per-class $\Sigma_k$ | Quadratic decision boundary |
| Naive Bayes | Independence within class | $f_k(x) = \prod_j f_{kj}(x_j)$ |
| KNN | Majority vote of $K$ neighbors | $\hat{P}(Y=j\|X=x_0) = \frac{1}{K}\sum_{i\in\mathcal{N}_0}\mathbf{1}(y_i = j)$ |
| Threshold | Adjust for imbalanced classes | Lower threshold → fewer false negatives |
| ROC/AUC | Evaluate across all thresholds | AUC = 1 perfect, 0.5 = random |

---

*Previous: [Chapter 3 — Linear Regression](ch03-linear-regression.md) | Next: [Chapter 5 — Resampling Methods](ch05-resampling.md)*
