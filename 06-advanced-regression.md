# Advanced Regression Methods

Advanced regression techniques address limitations of standard linear regression, including overfitting, multicollinearity, and non-linear relationships.

## Regularized Regression

### The Bias-Variance Tradeoff
- **Bias**: Error from oversimplifying the model
- **Variance**: Error from sensitivity to small fluctuations in training data
- **Goal**: Minimize total error = Bias² + Variance + Irreducible Error

### Ridge Regression (L2 Regularization)

#### Objective Function
**Minimize: RSS + λΣβⱼ²**

Where:
- RSS: Residual Sum of Squares
- λ (lambda): Regularization parameter
- L2 penalty: Σβⱼ² (sum of squared coefficients)

#### Properties
- **Shrinks coefficients** toward zero but doesn't set them exactly to zero
- **Handles multicollinearity** by distributing coefficients among correlated variables
- **All variables retained** in the model
- **λ = 0**: Standard linear regression
- **λ → ∞**: All coefficients approach zero

#### Matrix Solution
**β̂ᵣᵢdgₑ = (X'X + λI)⁻¹X'Y**

#### Choosing λ
- **Cross-validation**: Select λ that minimizes CV error
- **Ridge trace**: Plot coefficients vs λ values
- **GCV (Generalized Cross-Validation)**: Efficient approximation to LOOCV

### Lasso Regression (L1 Regularization)

#### Objective Function
**Minimize: RSS + λΣ|βⱼ|**

Where:
- L1 penalty: Σ|βⱼ| (sum of absolute values of coefficients)

#### Properties
- **Variable selection**: Sets some coefficients exactly to zero
- **Sparse solutions**: Automatically selects relevant features
- **Handles high-dimensional data** (p > n)
- **Not differentiable**: Requires specialized algorithms

#### Algorithms
- **Coordinate descent**: Iteratively optimize one coefficient at a time
- **LARS (Least Angle Regression)**: Efficient path algorithm
- **Proximal gradient methods**: For large datasets

#### Geometric Interpretation
- **Ridge**: Circular constraint region
- **Lasso**: Diamond-shaped constraint region
- **Corners of diamond**: Allow coefficients to be exactly zero

### Elastic Net

#### Objective Function
**Minimize: RSS + λ₁Σ|βⱼ| + λ₂Σβⱼ²**

Or equivalently:
**Minimize: RSS + λ[(1-α)Σβⱼ² + αΣ|βⱼ|]**

Where α ∈ [0,1] is the mixing parameter.

#### Properties
- **Combines Ridge and Lasso**: α = 0 (Ridge), α = 1 (Lasso)
- **Handles correlated groups**: Selects groups of correlated variables
- **Stable selection**: Less sensitive to small data changes than Lasso
- **Two parameters**: λ (overall regularization) and α (mixing)

#### Advantages
- **Grouping effect**: Correlated variables have similar coefficients
- **Better than Lasso** when p > n and variables are highly correlated
- **Robust selection**: More reliable variable selection than Lasso

## Polynomial Regression

### Model
**Y = β₀ + β₁X + β₂X² + ... + βₖXᵏ + ε**

### Implementation
- **Orthogonal polynomials**: Reduce multicollinearity between X, X², X³, etc.
- **Raw polynomials**: Direct powers of X (can be numerically unstable)

### Choosing Polynomial Degree
- **Cross-validation**: Select degree that minimizes prediction error
- **Information criteria**: AIC, BIC penalize complexity
- **Statistical tests**: F-tests for nested models

### Piecewise Polynomials and Splines

#### Step Functions
Divide X range into bins, fit constant within each bin.

#### Piecewise Linear
Connect linear segments at breakpoints (knots).

#### Cubic Splines
- **Cubic polynomials** in each interval
- **Continuity constraints** at knots
- **Smoothness constraints**: First and second derivatives continuous

#### Natural Cubic Splines
- Additional constraint: Linear beyond boundary knots
- **Fewer parameters** than regular cubic splines
- **Better boundary behavior**

#### Smoothing Splines
Choose knot locations and smoothness automatically:
**Minimize: Σ(yᵢ - g(xᵢ))² + λ∫g''(t)²dt**

## Generalized Additive Models (GAMs)

### Model
**Y = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ) + ε**

Where fⱼ are smooth functions.

### Components
- **Smooth functions**: Estimated non-parametrically
- **Additive structure**: Effects of variables add up
- **Flexibility**: Each variable can have different smooth relationship

### Fitting Algorithms
- **Backfitting**: Iteratively estimate each smooth function
- **Penalized likelihood**: Add smoothness penalties
- **GAM algorithms**: Specialized software implementations

### Advantages
- **Interpretability**: Additive structure easier to understand than full interactions
- **Flexibility**: Can capture non-linear relationships
- **Visualization**: Easy to plot individual smooth functions

### Limitations
- **No interactions**: Unless explicitly included
- **Curse of dimensionality**: Performance degrades with many variables
- **Overfitting**: Need careful regularization

## Tree-Based Methods

### Regression Trees

#### Splitting Criteria
- **RSS reduction**: Choose split that maximally reduces RSS
- **Recursive binary splitting**: Top-down, greedy approach

#### Tree Structure
- **Internal nodes**: Splitting rules
- **Terminal nodes (leaves)**: Predictions
- **Depth**: Number of levels in tree

#### Pruning
- **Problem**: Full trees overfit
- **Cost complexity pruning**: Balance fit and complexity
- **Cross-validation**: Choose optimal tree size

### Random Forest

#### Algorithm
1. **Bootstrap sampling**: Create B bootstrap samples
2. **Random feature sampling**: At each split, consider random subset of features
3. **Tree building**: Build trees on bootstrap samples
4. **Prediction**: Average predictions from all trees

#### Advantages
- **Reduces overfitting**: Averaging reduces variance
- **Handles missing data**: Built-in missing value handling
- **Variable importance**: Measures feature importance
- **Out-of-bag error**: Built-in validation without separate test set

#### Variable Importance
- **Mean decrease in impurity**: Average improvement in node purity
- **Permutation importance**: Decrease in accuracy when variable is permuted

### Gradient Boosting

#### Concept
- **Sequential learning**: Each tree corrects errors of previous trees
- **Gradient descent**: Optimize loss function using gradients
- **Weak learners**: Use simple trees (often depth 2-6)

#### Algorithm
1. Initialize with simple model
2. Compute residuals
3. Fit new tree to residuals
4. Update model
5. Repeat

#### Hyperparameters
- **Learning rate**: Controls contribution of each tree
- **Number of trees**: More trees = more complex model
- **Tree depth**: Complexity of individual trees
- **Subsampling**: Fraction of data used for each tree

## Support Vector Regression (SVR)

### Linear SVR
Find function that has at most ε deviation from targets.

**Objective**: Minimize ||w||² subject to constraints

### Non-linear SVR
- **Kernel trick**: Map to higher-dimensional space
- **Common kernels**: Polynomial, RBF (Gaussian), sigmoid

### ε-insensitive Loss
- **No penalty** for predictions within ε of true value
- **Linear penalty** beyond ε-tube

## Neural Networks for Regression

### Single Hidden Layer
**Y = β₀ + Σβⱼσ(γ₀ⱼ + Σγᵢⱼxᵢ)**

Where σ is activation function (sigmoid, ReLU, tanh).

### Deep Learning
- **Multiple hidden layers**: Can capture complex patterns
- **Regularization**: Dropout, weight decay, early stopping
- **Optimization**: SGD, Adam, RMSprop

### Applications in Bioinformatics
- **Gene expression prediction**: From sequence features
- **Protein structure prediction**: From amino acid sequence
- **Drug discovery**: QSAR modeling

## Model Selection and Comparison

### Cross-Validation Strategies

#### K-fold Cross-Validation
Standard approach for model comparison.

#### Time Series Cross-Validation
For temporal data: use past to predict future.

#### Group Cross-Validation
For clustered data: ensure groups don't split across folds.

### Information Criteria
- **AIC**: -2log(L) + 2k
- **BIC**: -2log(L) + k log(n)
- **Extended IC**: Include complexity penalties for regularized models

### Prediction Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Adjusted R²**: Penalized R²

## Ensemble Methods

### Bagging (Bootstrap Aggregating)
- **Bootstrap samples**: Train models on different samples
- **Average predictions**: Reduce variance
- **Examples**: Random Forest, Extra Trees

### Boosting
- **Sequential training**: Each model corrects previous errors
- **Examples**: AdaBoost, Gradient Boosting, XGBoost

### Stacking
- **Meta-learning**: Train meta-model on predictions of base models
- **Cross-validation**: Prevent overfitting in meta-model

## Bioinformatics Applications

### Genomics

#### GWAS with Regularization
- **High-dimensional data**: Millions of SNPs, thousands of samples
- **Lasso regression**: Automatic SNP selection
- **Elastic Net**: Handle linkage disequilibrium

#### Gene Expression Analysis
- **Ridge regression**: Handle correlated genes
- **Sparse methods**: Identify key regulatory genes
- **Pathway analysis**: Group regularization

### Proteomics

#### Mass Spectrometry Data
- **High noise**: Robust regression methods
- **Many features**: Regularized approaches
- **Non-linear relationships**: GAMs, tree methods

### Drug Discovery

#### QSAR Modeling
- **Molecular descriptors**: High-dimensional chemical features
- **Random Forest**: Handle non-linear structure-activity relationships
- **SVR**: Robust to outliers

### Phylogenetics

#### Trait Evolution
- **Phylogenetic regression**: Account for evolutionary relationships
- **Non-linear trends**: GAMs for trait evolution
- **Tree-based methods**: Model complex evolutionary scenarios

## Implementation Considerations

### Software Packages

#### R
- **glmnet**: Ridge, Lasso, Elastic Net
- **mgcv**: GAMs
- **randomForest**: Random Forest implementation
- **gbm**: Gradient Boosting Machines

#### Python
- **scikit-learn**: Comprehensive machine learning library
- **statsmodels**: Statistical modeling
- **xgboost**: Extreme Gradient Boosting
- **keras/tensorflow**: Deep learning

### Computational Considerations
- **Scalability**: Some methods don't scale to very large datasets
- **Memory usage**: Consider memory requirements for large datasets
- **Parallel computing**: Many algorithms can be parallelized

### Hyperparameter Tuning
- **Grid search**: Exhaustive search over parameter grid
- **Random search**: Often more efficient than grid search
- **Bayesian optimization**: Intelligent parameter search

## Best Practices

### Data Preprocessing
1. **Handle missing data**: Imputation or removal
2. **Scale features**: Especially important for regularized methods
3. **Feature engineering**: Create relevant derived features

### Model Development
1. **Start simple**: Begin with linear models, add complexity gradually
2. **Use domain knowledge**: Incorporate biological understanding
3. **Validate thoroughly**: Use proper cross-validation schemes

### Model Interpretation
1. **Variable importance**: Understand which features matter
2. **Partial dependence plots**: Visualize feature effects
3. **SHAP values**: Modern approach to feature importance

### Reporting Results
1. **Model comparison**: Compare multiple approaches
2. **Uncertainty quantification**: Provide confidence/prediction intervals
3. **Reproducibility**: Share code and parameter settings