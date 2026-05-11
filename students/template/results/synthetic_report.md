# Scenario A — Synthetic Data Baseline Report

## Data Generating Process

- n = 1000  (800 train / 200 test)
- True β = [ 5.   2.  -1.5  3. ]
- Estimated β = [ 5.0219  1.9795 -1.5107  2.9128]

## Model Comparison

| Model                               | Fit Time      | R² (test)  |
|-------------------------------------|---------------|------------|
| CustomOLS (NumPy)                   | 0.00008 sec | 0.933402 |
| sklearn LinearRegression            | 0.00112 sec | 0.933402 |

✅ Assertions passed — R²(CustomOLS, test) = 0.9334
