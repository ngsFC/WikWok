# Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset, providing simple summaries about the sample and observations.

## Measures of Central Tendency

### Mean (Arithmetic Average)
The sum of all values divided by the number of observations.

**Formula**: x̄ = (Σxi) / n

**Properties**:
- Sensitive to outliers
- Uses all data points
- Can be non-integer even for integer data

**Example**: Gene expression levels: [2.1, 3.4, 2.8, 15.2, 3.1]
Mean = (2.1 + 3.4 + 2.8 + 15.2 + 3.1) / 5 = 5.32

### Median
The middle value when data is ordered from least to greatest.

**Properties**:
- Robust to outliers
- Divides data into two equal halves
- Better than mean for skewed distributions

**Example**: From above data [2.1, 2.8, 3.1, 3.4, 15.2]
Median = 3.1 (middle value)

### Mode
The most frequently occurring value(s) in the dataset.

**Types**:
- **Unimodal**: One mode
- **Bimodal**: Two modes
- **Multimodal**: Multiple modes
- **No mode**: All values occur with equal frequency

## Measures of Variability (Dispersion)

### Range
The difference between the maximum and minimum values.

**Formula**: Range = max(x) - min(x)

**Limitations**: Only uses two values, sensitive to outliers

### Variance
The average of squared deviations from the mean.

**Population Variance**: σ² = Σ(xi - μ)² / N
**Sample Variance**: s² = Σ(xi - x̄)² / (n-1)

**Note**: Sample variance uses (n-1) for Bessel's correction to provide unbiased estimate.

### Standard Deviation
The square root of variance, expressed in original units.

**Population**: σ = √σ²
**Sample**: s = √s²

### Coefficient of Variation (CV)
Relative measure of variability, useful for comparing datasets with different units or scales.

**Formula**: CV = (s / x̄) × 100%

### Interquartile Range (IQR)
The range of the middle 50% of the data.

**Formula**: IQR = Q3 - Q1

**Advantages**: Robust to outliers, describes spread of middle data

## Measures of Position

### Percentiles
Values below which a certain percentage of data falls.

**Common Percentiles**:
- 25th percentile (Q1): First quartile
- 50th percentile (Q2): Median
- 75th percentile (Q3): Third quartile

### Quartiles
Values that divide the dataset into four equal parts.
- **Q1**: 25% of data below this value
- **Q2**: Median (50%)
- **Q3**: 75% of data below this value

### Z-scores (Standard Scores)
Number of standard deviations a value is from the mean.

**Formula**: z = (x - μ) / σ

**Interpretation**:
- z = 0: Value equals the mean
- z > 0: Value above the mean
- z < 0: Value below the mean
- |z| > 2: Considered unusual (beyond 2 standard deviations)

## Measures of Shape

### Skewness
Measures asymmetry of the distribution.

**Types**:
- **Right-skewed (positive)**: Tail extends to the right, mean > median
- **Left-skewed (negative)**: Tail extends to the left, mean < median
- **Symmetric**: Mean ≈ median

**Formula**: Skewness = E[(X - μ)³] / σ³

### Kurtosis
Measures the "tailedness" of the distribution.

**Types**:
- **Mesokurtic**: Normal distribution (kurtosis = 3)
- **Leptokurtic**: Heavy tails, sharp peak (kurtosis > 3)
- **Platykurtic**: Light tails, flat peak (kurtosis < 3)

## Data Visualization

### Numerical Summaries
- **Five-number summary**: Min, Q1, Median, Q3, Max
- **Summary statistics table**: Mean, median, mode, std dev, variance

### Graphical Methods

#### For Single Variables
- **Histogram**: Shows distribution shape and frequency
- **Box plot**: Displays five-number summary and outliers
- **Density plot**: Smooth estimate of distribution
- **Dot plot**: Each observation as a dot
- **Stem-and-leaf plot**: Retains actual data values

#### For Categorical Data
- **Bar chart**: Heights represent frequencies or percentages
- **Pie chart**: Shows proportions of a whole
- **Frequency table**: Counts and percentages

#### For Relationships
- **Scatter plot**: Relationship between two continuous variables
- **Side-by-side box plots**: Compare distributions across groups
- **Heat map**: Correlation matrix visualization

## Outlier Detection

### Methods for Identifying Outliers

#### Statistical Methods
1. **IQR Method**: 
   - Lower fence: Q1 - 1.5×IQR
   - Upper fence: Q3 + 1.5×IQR
   - Values beyond fences are outliers

2. **Z-score Method**: |z| > 2 or 3 (depending on threshold)

3. **Modified Z-score**: Uses median absolute deviation (MAD)

#### Visual Methods
- Box plots clearly show outliers as points beyond whiskers
- Scatter plots reveal outliers as isolated points

### Handling Outliers
1. **Investigation**: Determine if outlier is error or genuine observation
2. **Removal**: Only if confirmed to be error
3. **Transformation**: Log, square root to reduce impact
4. **Robust methods**: Use median, MAD instead of mean, std dev
5. **Separate analysis**: Analyze with and without outliers

## Bioinformatics Applications

### Gene Expression Analysis
- **Log transformation**: Convert fold-changes to normal-like distribution
- **Quantile normalization**: Make samples comparable
- **Outlier genes**: May indicate biological significance or technical issues

### Sequence Analysis
- **GC content**: Percentage of G and C nucleotides
- **Read depth distribution**: Coverage statistics in sequencing
- **Quality scores**: Distribution of sequencing quality

### Protein Analysis
- **Hydrophobicity indices**: Descriptive stats of amino acid properties
- **Secondary structure content**: Percentages of α-helix, β-sheet, coil

## Common Mistakes and Considerations

1. **Mean vs Median**: Use median for skewed data
2. **Sample vs Population**: Use appropriate formulas
3. **Outlier influence**: Consider robust statistics
4. **Scale differences**: Use coefficient of variation for comparison
5. **Missing data**: Decide how to handle before calculating statistics