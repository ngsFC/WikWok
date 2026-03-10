# Chapter 3: Linear Regression

> *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani

## 3.1 Simple Linear Regression

Assumes a linear relationship between a single predictor $X$ and response $Y$:

$$Y \approx \beta_0 + \beta_1 X$$

### Estimating the Coefficients

The **least squares** approach minimizes the residual sum of squares (RSS):

$$\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

The closed-form estimates:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**Example: Advertising data**

| Coefficient | Estimate | Std. Error | t-statistic | p-value |
|-------------|----------|------------|-------------|---------|
| Intercept   | 7.0325   | 0.4578     | 15.36       | <0.0001 |
| TV          | 0.0475   | 0.0027     | 17.67       | <0.0001 |

Interpretation: spending $1,000 extra on TV advertising is associated with ~47.5 additional units sold.

### Assessing Coefficient Accuracy

- **Standard errors**: $\text{SE}(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_i(x_i - \bar{x})^2}$
- **95% confidence interval**: $\hat{\beta}_1 \pm 2 \cdot \text{SE}(\hat{\beta}_1)$
- **Hypothesis test**: $H_0: \beta_1 = 0$ via t-statistic $t = \hat{\beta}_1 / \text{SE}(\hat{\beta}_1)$

### Model Fit: RSE and R²

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}}, \qquad R^2 = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}$$

For Advertising: RSE = 3.26 (units on the sales scale), $R^2 = 0.612$ (TV explains 61% of variance in sales).

---

## 3.2 Multiple Linear Regression

Extends to $p$ predictors:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

**Example: Advertising data with TV, radio, newspaper**

| Predictor | Coefficient | Std. Error | t-stat | p-value |
|-----------|-------------|------------|--------|---------|
| Intercept | 2.939       | 0.3119     | 9.42   | <0.0001 |
| TV        | 0.046       | 0.0014     | 32.81  | <0.0001 |
| Radio     | 0.189       | 0.0086     | 21.89  | <0.0001 |
| Newspaper | −0.001      | 0.0059     | −0.18  | 0.8599  |

Newspaper becomes **non-significant** in multiple regression (was significant alone) because it's a surrogate for radio. Multiple regression R² = 0.897.

### F-statistic (Overall Model Significance)

Tests $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$:

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)}$$

Large $F$ (small p-value) → reject $H_0$, some predictor is related to response.

### Variable Selection

Given $p$ predictors, $2^p$ possible models. Practical approaches:

| Method | Approach |
|--------|---------|
| **Forward selection** | Start empty; add most significant predictor one at a time |
| **Backward selection** | Start full; remove least significant predictor one at a time |
| **Mixed selection** | Combine both, re-checking significance after each addition |

Criteria: Adjusted $R^2$, AIC, BIC, Mallow's $C_p$.

---

## 3.3 Other Considerations

### Qualitative Predictors

For a two-level factor, create a **dummy variable**:

$$x_i = \begin{cases} 1 & \text{if female} \\ 0 & \text{if male} \end{cases}$$

**Credit dataset example** — balance vs gender:

| Coefficient | Estimate | Std. Error | t-stat | p-value |
|-------------|----------|------------|--------|---------|
| Intercept   | 509.80   | 33.13      | 15.389 | <0.0001 |
| Female      | 19.73    | 46.05      | 0.429  | 0.6690  |

For $k$-level factors: create $k-1$ dummy variables (one level is the **baseline**).

```r
# R automatically creates dummies for factor variables
lm(balance ~ gender + income + student, data = Credit)
```

### Interaction Terms

Relaxes the **additive assumption**: include $X_1 \times X_2$:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \epsilon$$

**Advertising with interaction (TV×radio):**

| Predictor | Coefficient | Std. Error | t-stat | p-value |
|-----------|-------------|------------|--------|---------|
| Intercept | 6.7502      | 0.248      | 27.23  | <0.0001 |
| TV        | 0.0191      | 0.002      | 12.70  | <0.0001 |
| Radio     | 0.0289      | 0.009      | 3.24   | 0.0014  |
| TV×Radio  | 0.0011      | 0.000      | 20.73  | <0.0001 |

$R^2$ jumps from 89.7% to **96.8%** — confirms synergy between TV and radio advertising.

> **Hierarchical principle**: if interaction is included, include main effects too, even if their p-values are large.

### Polynomial (Non-linear) Regression

Model non-linearity by including polynomial terms:

$$\text{mpg} = \beta_0 + \beta_1 \times \text{horsepower} + \beta_2 \times \text{horsepower}^2 + \epsilon$$

```r
lm(mpg ~ poly(horsepower, 2), data = Auto)
# R² improves from 0.606 (linear) to 0.688 (quadratic)
```

**Table 3.10** (Auto dataset):

| Predictor | Coefficient | Std. Error | t-stat | p-value |
|-----------|-------------|------------|--------|---------|
| Intercept | 56.90       | 1.800      | 31.6   | <0.0001 |
| horsepower | −0.4662    | 0.0311     | −15.0  | <0.0001 |
| horsepower² | 0.0012   | 0.0001     | 10.1   | <0.0001 |

---

## 3.3.3 Potential Problems

### 1. Non-linearity
**Detection**: residual plot — curved pattern suggests non-linearity.
**Fix**: transform predictors ($\log X$, $\sqrt{X}$, $X^2$).

### 2. Correlated Error Terms
Common in **time series** data — adjacent errors correlated.
**Detection**: plot residuals vs time; look for "tracking" patterns.
**Effect**: standard errors underestimated → false confidence.

### 3. Heteroscedasticity
Non-constant $\text{Var}(\epsilon_i)$.
**Detection**: funnel shape in residual plot.
**Fix**: transform response $\log Y$ or $\sqrt{Y}$; use weighted least squares.

### 4. Outliers
Points with unusually large residuals.
**Detection**: **studentized residuals** > 3 in absolute value.

### 5. High Leverage Points
Unusual $x_i$ values (not $y_i$). Can disproportionately influence fit.
**Measure**: leverage statistic $h_i = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_{j}(x_j-\bar{x})^2}$; average leverage = $(p+1)/n$.

### 6. Collinearity
Predictors correlated with each other → inflated SEs, reduced power.

**Detection**:
- Correlation matrix
- **Variance Inflation Factor (VIF)**: $\text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R^2_{X_j|X_{-j}}}$

| VIF | Interpretation |
|-----|---------------|
| 1 | No collinearity |
| 1–5 | Moderate |
| > 5–10 | Problematic |

**Example**: Credit data — `limit` and `rating` are collinear (VIFs ~160). SE of $\hat{\beta}_{\text{limit}}$ inflates 12-fold.

**Fix**: drop one collinear predictor, or combine them (e.g., average of standardized versions = "credit worthiness").

---

## 3.5 Linear Regression vs KNN Regression

KNN regression: $\hat{f}(x_0) = \frac{1}{K}\sum_{x_i \in \mathcal{N}_0} y_i$

- When true $f$ is linear → **linear regression beats KNN** (parametric wins)
- When true $f$ is non-linear → **KNN beats linear regression**
- In high dimensions ($p \geq 4$): KNN degrades (**curse of dimensionality**)

---

## 3.6 Lab: Linear Regression in R

### Boston Dataset

Predict `medv` (median house value) from 13 predictors.

```r
library(MASS)
library(ISLR)

# Simple regression
lm.fit <- lm(medv ~ lstat, data = Boston)
summary(lm.fit)
# Intercept: 34.55, lstat: -0.95, R² = 0.544

# Confidence intervals
confint(lm.fit)
#             2.5 %   97.5 %
# (Intercept) 33.45   35.659
# lstat       -1.03   -0.874

# Predictions
predict(lm.fit, data.frame(lstat=c(5,10,15)), interval="confidence")
#       fit   lwr   upr
# 1  29.80  29.01  30.60
# 2  25.05  24.47  25.63
# 3  20.30  19.73  20.87

# Residual plots
par(mfrow=c(2,2))
plot(lm.fit)
```

### Multiple Regression

```r
# All predictors
lm.fit <- lm(medv ~ ., data = Boston)
# R² = 0.7406, RSE = 4.745

# VIF to check collinearity
library(car)
vif(lm.fit)
# rad: 7.48, tax: 9.01 (moderate-high)

# Interaction term
summary(lm(medv ~ lstat * age, data = Boston))
# lstat:age interaction significant (p = 0.025)

# Non-linear transformation
lm.fit2 <- lm(medv ~ lstat + I(lstat^2), data = Boston)
# R² = 0.641 vs 0.544 (linear)
anova(lm.fit, lm.fit2)  # F = 135, p < 2e-16 → quadratic is better

# Polynomial fit
lm.fit5 <- lm(medv ~ poly(lstat, 5), data = Boston)
# R² = 0.682
```

### Carseats Dataset — Qualitative Predictors

```r
lm.fit <- lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
# R² = 0.876
# ShelveLocGood: +4.85, ShelveLocMedium: +1.95 (vs Bad baseline)
contrasts(Carseats$ShelveLoc)
```

---

## Summary

| Concept | Key Formula/Insight |
|---------|-------------------|
| Simple regression | $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$, minimize RSS |
| Model fit | $R^2 = 1 - \text{RSS/TSS}$; RSE = typical error magnitude |
| Overall significance | F-statistic tests all $\beta_j = 0$ simultaneously |
| Qualitative predictors | Dummy variables; $k$-level factor → $k-1$ dummies |
| Interaction | Include $X_1 X_2$ term; hierarchical principle |
| Collinearity | VIF > 5–10 is problematic; drop or combine predictors |
| Non-linearity | Polynomial or log transforms; check residual plots |

---

*Previous: [Chapter 2](ch02-statistical-learning.md) | Next: [Chapter 4 — Classification](ch04-classification.md)*
