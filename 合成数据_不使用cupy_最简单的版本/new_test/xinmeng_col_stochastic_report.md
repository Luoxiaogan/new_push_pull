# Xinmeng Column-Stochastic Matrices Analysis Report

## Overview
- Node count range tested: 4 to 104
- Matrix types: get_xinmeng_matrix (fixed) and get_xinmeng_like_matrix (random)
- Random matrix samples: 10 seeds averaged
- **Matrix Type**: Column-stochastic (normalized by column sums)

## Key Findings

### 1. Condition Number (Kappa) Analysis
- **Fixed Matrix**:
  - Growth exponent: 15.728 (O(n^15.73))
  - Value at n=64: 1.78e+16
  - Min value (n=4): 4.00
  - Max value (n=96): 1.10e+22

- **Random Matrix**:
  - Mean at n=64: 2.52e+27 ± 2.62e+28
  - Average variability: 923.3%

### 2. Beta Values Analysis
- **Fixed Matrix**:
  - Range: 0.7234 to 0.8969
  - Trend: Decreasing with n

- **Random Matrix**:
  - Mean range: 0.7429 to 0.9870
  - Average standard deviation: 0.0426

### 3. Spectral Gap (1-β) Analysis
- **Fixed Matrix**:
  - Min spectral gap (n=72): 0.103095
  - Max spectral gap (n=4): 0.276638
  - Convergence rate deterioration: nan%

### 4. Spectral Complexity (S_B) Analysis
- **Fixed Matrix**:
  - Growth exponent: 1.637 (O(n^1.64))
  - Value at n=64: 3.15e+03

### 5. Perron Vector Analysis
- **Fixed Matrix**:
  - Max Perron entry at n=64: 0.500000
  - Min Perron entry at n=64: 2.81e-17
  - Ratio (max/min) = Kappa

### 6. Fixed vs Random Comparison
- Kappa ratio (random/fixed): 41314042905944181964800.00x on average
- Beta difference: 0.1070
- Spectral gap ratio: 0.46x

## Conclusions

1. **Scalability**: The condition number grows as O(n^15.7), indicating poor scalability.

2. **Convergence**: Spectral gap decreases with n, suggesting slower convergence for larger networks.

3. **Randomization Impact**: Random matrices show higher condition numbers with significant variability.

4. **Practical Implications**: 
   - For n > 32, numerical stability becomes a concern (kappa > 10^9)
   - The tridiagonal structure creates highly imbalanced Perron vectors
   - Not recommended for distributed optimization algorithms requiring good conditioning

## Files Generated
- Numerical results: xinmeng_col_stochastic_analysis.csv
- Visualization: xinmeng_col_stochastic_properties.png
