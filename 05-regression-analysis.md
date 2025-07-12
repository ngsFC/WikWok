# Regression Analysis

Regression analysis examines relationships between variables, allowing prediction and understanding of how one variable affects another.

## Simple Linear Regression

### Model
**Y = β₀ + β₁X + ε**

Where:
- Y: Dependent (response) variable
- X: Independent (predictor) variable
- β₀: Y-intercept
- β₁: Slope (change in Y per unit change in X)
- ε: Error term (residuals)

### Assumptions
1. **Linearity**: Relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No outliers**: Extreme values don't unduly influence the model

### Parameter Estimation (Least Squares)

**Slope**: β̂₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

**Intercept**: β̂₀ = ȳ - β̂₁x̄

**Alternative formula**: β̂₁ = Cov(X,Y) / Var(X)

### Model Evaluation

#### R-squared (R²)
Proportion of variance in Y explained by X.

**R² = SSR / SST = 1 - SSE / SST**

Where:
- SST: Total Sum of Squares
- SSR: Regression Sum of Squares
- SSE: Error Sum of Squares

**Interpretation**:
- R² = 0: No linear relationship
- R² = 1: Perfect linear relationship
- Typical range: 0 ≤ R² ≤ 1

#### Correlation Coefficient (r)
**r = √R²** (for simple linear regression)

**Properties**:
- -1 ≤ r ≤ 1
- Sign indicates direction of relationship
- |r| indicates strength of linear relationship

#### Standard Error of Regression
**s = √(SSE / (n-2))**

### Hypothesis Testing

#### Testing Slope Significance
**H₀: β₁ = 0** (no linear relationship)
**H₁: β₁ ≠ 0** (linear relationship exists)

**Test statistic**: t = β̂₁ / SE(β̂₁)
**Degrees of freedom**: df = n - 2

#### Confidence Interval for Slope
**β̂₁ ± t_(α/2,n-2) × SE(β̂₁)**

### Prediction

#### Point Prediction
**Ŷ = β̂₀ + β̂₁X**

#### Confidence Interval for Mean Response
Interval for E[Y|X = x₀]

#### Prediction Interval for Individual Response
Interval for a new individual Y value at X = x₀
(Wider than confidence interval due to additional uncertainty)

## Multiple Linear Regression

### Model
**Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε**

### Matrix Notation
**Y = Xβ + ε**

Where:
- Y: n×1 response vector
- X: n×(p+1) design matrix
- β: (p+1)×1 parameter vector
- ε: n×1 error vector

### Parameter Estimation
**β̂ = (X'X)⁻¹X'Y**

### Model Evaluation

#### Adjusted R-squared
Accounts for number of predictors in the model.

**R²ₐdⱼ = 1 - [(1 - R²)(n - 1) / (n - p - 1)]**

**Properties**:
- Can decrease when adding irrelevant variables
- Better for model comparison than R²

#### F-test for Overall Significance
Tests whether at least one predictor is significant.

**H₀: β₁ = β₂ = ... = βₚ = 0**
**H₁: At least one βᵢ ≠ 0**

**F = MSR / MSE**

Where:
- MSR = SSR / p
- MSE = SSE / (n - p - 1)

### Individual Coefficient Testing
**H₀: βᵢ = 0**

**Test statistic**: t = β̂ᵢ / SE(β̂ᵢ)
**Degrees of freedom**: df = n - p - 1

### Variable Selection

#### Forward Selection
Start with no variables, add variables one by one based on significance.

#### Backward Elimination
Start with all variables, remove non-significant variables one by one.

#### Stepwise Selection
Combination of forward and backward, can add or remove variables at each step.

#### Best Subsets
Evaluate all possible combinations of variables.

**Selection Criteria**:
- **AIC (Akaike Information Criterion)**: Lower is better
- **BIC (Bayesian Information Criterion)**: Penalizes model complexity more
- **Cp (Mallows' Cp)**: Balances bias and variance

## Logistic Regression

### When to Use
- Binary outcome variable (success/failure, diseased/healthy)
- Probability modeling
- Classification problems

### Logit Model
**logit(p) = ln(p/(1-p)) = β₀ + β₁X₁ + ... + βₚXₚ**

Where p = P(Y = 1|X)

### Probability Calculation
**p = e^(β₀ + β₁X₁ + ... + βₚXₚ) / (1 + e^(β₀ + β₁X₁ + ... + βₚXₚ))**

### Odds and Odds Ratio

#### Odds
**Odds = p / (1-p)**

#### Odds Ratio (OR)
For one unit increase in Xᵢ:
**OR = e^βᵢ**

**Interpretation**:
- OR > 1: Positive association
- OR < 1: Negative association
- OR = 1: No association

### Parameter Estimation
Uses Maximum Likelihood Estimation (MLE) instead of least squares.

### Model Evaluation

#### Likelihood Ratio Test
Compares nested models.

**G = -2ln(L₀/L₁) = -2(ln L₀ - ln L₁)**

Where L₀ and L₁ are likelihoods of reduced and full models.

#### Hosmer-Lemeshow Test
Tests goodness of fit by grouping observations.

#### ROC Curve and AUC
- **ROC**: Receiver Operating Characteristic curve
- **AUC**: Area Under the Curve
- **Range**: 0.5 (no discrimination) to 1.0 (perfect discrimination)

#### Classification Metrics
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Accuracy**: Overall correct classification rate
- **Precision**: Positive predictive value

## Regression Diagnostics

### Residual Analysis

#### Types of Residuals
- **Raw residuals**: eᵢ = yᵢ - ŷᵢ
- **Standardized residuals**: eᵢ / s
- **Studentized residuals**: Account for leverage

#### Diagnostic Plots
1. **Residuals vs Fitted**: Check linearity and homoscedasticity
2. **Q-Q plot**: Check normality of residuals
3. **Scale-Location**: Check homoscedasticity
4. **Residuals vs Leverage**: Identify influential points

### Assumption Violations

#### Non-linearity
**Solutions**:
- Transform variables (log, square root, polynomial)
- Add interaction terms
- Use non-parametric regression

#### Heteroscedasticity
**Detection**: Breusch-Pagan test, White test
**Solutions**:
- Transform dependent variable
- Weighted least squares
- Robust standard errors

#### Multicollinearity
**Detection**: 
- **VIF (Variance Inflation Factor)**: VIF > 10 indicates problem
- **Condition Index**: > 30 indicates severe multicollinearity

**Solutions**:
- Remove correlated variables
- Ridge regression
- Principal component regression

#### Autocorrelation
**Detection**: Durbin-Watson test
**Solutions**:
- Include lagged variables
- Use time series methods

### Influential Observations

#### Leverage
Measures how far X values are from their mean.
**High leverage**: hᵢᵢ > 2(p+1)/n

#### Cook's Distance
Measures influence of observation on all fitted values.
**Influential**: D > 4/n

#### DFBETAS
Measures influence on individual coefficients.

## Model Selection and Validation

### Cross-Validation

#### K-fold Cross-Validation
1. Divide data into k folds
2. Use k-1 folds for training, 1 for testing
3. Repeat k times
4. Average performance metrics

#### Leave-One-Out Cross-Validation (LOOCV)
Special case where k = n.

### Information Criteria

#### AIC (Akaike Information Criterion)
**AIC = -2ln(L) + 2k**

Where k is number of parameters.

#### BIC (Bayesian Information Criterion)
**BIC = -2ln(L) + k ln(n)**

More penalty for model complexity than AIC.

### Train/Validation/Test Split
- **Training set**: Fit models
- **Validation set**: Select best model
- **Test set**: Estimate final performance

## Advanced Topics

### Interaction Terms
**Y = β₀ + β₁X₁ + β₂X₂ + β₃X₁X₂ + ε**

The effect of X₁ depends on the value of X₂.

### Polynomial Regression
**Y = β₀ + β₁X + β₂X² + ... + βₖXᵏ + ε**

Captures non-linear relationships using linear regression framework.

### Generalized Linear Models (GLMs)
Extension that allows:
- Different error distributions (binomial, Poisson, gamma)
- Link functions connecting linear predictor to mean

**Components**:
1. **Random component**: Error distribution
2. **Systematic component**: Linear predictor
3. **Link function**: Connects mean to linear predictor

### Robust Regression
Methods that are less sensitive to outliers:
- **Huber regression**: Uses Huber loss function
- **Quantile regression**: Models quantiles instead of mean
- **M-estimators**: Downweight outliers

## Bioinformatics Applications

### Gene Expression Analysis
- **Linear models**: Comparing expression across conditions
- **Mixed effects models**: Account for batch effects and random factors
- **Interaction terms**: Gene × treatment interactions

### GWAS (Genome-Wide Association Studies)
- **Logistic regression**: Case-control studies
- **Linear regression**: Quantitative traits
- **Population stratification**: Include principal components as covariates

### Phylogenetic Comparative Methods
- **PGLS (Phylogenetic Generalized Least Squares)**: Account for phylogenetic relationships
- **Independent contrasts**: Remove phylogenetic autocorrelation

### Dose-Response Analysis
- **Sigmoid models**: EC50 estimation
- **Hill equation**: Cooperative binding
- **Biphasic models**: Multiple binding sites

### Survival Analysis Extensions
- **Cox proportional hazards**: Semi-parametric regression
- **Accelerated failure time**: Parametric survival models

### Network Analysis
- **Regression on graph**: Predict node properties
- **Network regression**: Model network structure

## Common Mistakes and Best Practices

### Data Preparation
1. **Check for missing data**: Impute or remove appropriately
2. **Scale variables**: Standardize for better interpretation
3. **Outlier detection**: Identify and handle appropriately

### Model Building
1. **Start simple**: Begin with simple models, add complexity gradually
2. **Domain knowledge**: Use biological understanding to guide modeling
3. **Avoid overfitting**: Use cross-validation and regularization

### Interpretation
1. **Correlation ≠ Causation**: Regression shows association, not causation
2. **Extrapolation**: Be cautious predicting outside observed range
3. **Effect size**: Consider practical significance alongside statistical significance

### Reporting
1. **Report assumptions**: State and check model assumptions
2. **Confidence intervals**: Provide uncertainty estimates
3. **Model diagnostics**: Include residual analysis
4. **Reproducibility**: Provide code and data when possible