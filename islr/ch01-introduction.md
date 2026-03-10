# Chapter 1: Introduction to Statistical Learning

> *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani

## What is Statistical Learning?

Statistical learning refers to a vast set of tools for **understanding data**. These tools can be classified as:

- **Supervised** — building a statistical model for predicting or estimating an output based on one or more inputs
- **Unsupervised** — there is no supervising output, but we can still learn relationships and structure from the data

---

## Three Motivating Examples

### 1. Wage Data (Regression)

A survey of males in the central Atlantic US, examining factors affecting **wage** as a function of age, education, and year.

```r
library(ISLR)
data(Wage)
# Wage ~ age + year + education
# Key insight: wage increases with age (up to ~60), with year, and with education level
```

Key finding: wages tend to increase with age until mid-60s, then decline. Higher education (advanced degree) associated with substantially higher wages.

### 2. Stock Market Data — Smarket (Classification)

Daily percentage returns for the S&P 500 index 2001–2005. Goal: predict whether market goes **Up or Down** on a given day.

```r
data(Smarket)
# Variables: Lag1...Lag5 (previous returns), Volume, Direction (Up/Down)
# Classification problem — no continuous response, just a category
```

Key finding: predicting market direction is difficult; returns have very low autocorrelation — the market is close to a random walk.

### 3. Gene Expression Data — NCI60 (Unsupervised)

Gene expression measurements on **6,830 genes** for **64 cancer cell lines** from the National Cancer Institute. No response variable — goal is to discover natural groupings among cell lines.

```r
data(NCI60)
# Unsupervised: cluster cell lines by expression profile
# PCA reveals structure corresponding to cancer type
```

Key finding: cell lines cluster naturally by tissue of origin (leukemia, colon, melanoma, etc.) even without any label information.

---

## Statistical Learning Framework

| Type | Response | Goal | Example |
|------|----------|------|---------|
| Regression | Quantitative (continuous) | Predict numeric value | Wage prediction |
| Classification | Qualitative (categorical) | Predict class label | Market Up/Down |
| Unsupervised | None | Discover structure | Gene clustering |

---

## A Brief History

- **1800s**: Legendre & Gauss developed least squares for astronomical orbit prediction
- **1900s**: Fisher introduced linear discriminant analysis
- **1970s**: Generalized linear models (Nelder & Wedderburn)
- **1980s**: Classification and Regression Trees (CART), neural networks
- **1990s**: SVMs, boosting, model selection advances
- **2000s+**: Statistical learning merges with machine learning; big data era

---

## Notation Used Throughout

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of observations |
| $p$ | Number of variables/predictors |
| $x_{ij}$ | Value of $j$th variable for $i$th observation |
| $y_i$ | Response for $i$th observation |
| $\mathbf{X}$ | $n \times p$ data matrix |

---

## Book Roadmap

| Chapter | Topic |
|---------|-------|
| 2 | Statistical Learning fundamentals |
| 3 | Linear Regression |
| 4 | Classification (logistic regression, LDA, QDA, KNN) |
| 5 | Resampling Methods (CV, Bootstrap) |
| 6 | Model Selection and Regularization (Ridge, Lasso) |
| 7 | Moving Beyond Linearity (Splines, GAMs) |
| 8 | Tree-Based Methods (Bagging, RF, Boosting) |
| 9 | Support Vector Machines |
| 10 | Unsupervised Learning (PCA, K-Means, Hierarchical) |

---

*Next: [Chapter 2 — Statistical Learning](ch02-statistical-learning.md)*
