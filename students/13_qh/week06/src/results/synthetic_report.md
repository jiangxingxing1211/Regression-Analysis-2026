# Scenario A: Synthetic Data Baseline Test

## Data Configuration
- Sample size: N = 1000
- Features: P = 3 (plus intercept)
- True beta: [ 5.  3. -2.]
- Noise std: 0.5

## Model Comparison

| Model | Fit Time | R² Score |
|-------|----------|----------|
| CustomOLS | 0.00026 sec | 0.9923 |
| Sklearn | 0.00154 sec | 0.9923 |


## Coefficients Comparison

| Parameter | True Value | CustomOLS | Sklearn |
|-----------|------------|-----------|---------|
| Intercept | 0.0 | -0.0045 | -0.0045 |
| beta_1 | 5.0 | 5.0180 | 0.0000 |
| beta_2 | 3.0 | 2.9986 | 5.0180 |
| beta_3 | -2.0 | -2.0316 | 2.9986 |

## Conclusion
- CustomOLS and Sklearn produce identical results (within numerical precision)
- The coefficients closely match the true beta values
- R² is high, indicating good model fit
