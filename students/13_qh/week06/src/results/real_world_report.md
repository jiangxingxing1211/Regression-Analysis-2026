# Scenario B: Real World Marketing Data Analysis

## Data Overview
- Total samples: 1000
- NA market samples: 500
- EU market samples: 500
- Features: TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday

## Model Comparison

| Model | Fit Time | R² Score |
|-------|----------|----------|
| CustomOLS (NA) | - | 0.9970 |
| Sklearn (NA) | - | 0.9970 |
| CustomOLS (EU) | - | 0.9976 |
| Sklearn (EU) | - | 0.9976 |

## Coefficients

### North America (NA) Market
| Parameter | CustomOLS | Sklearn |
|-----------|-----------|---------|
| Intercept | 48.1036 | 48.1036 |
| TV_Budget | 3.5075 | 3.5075 |
| Radio_Budget | 3.4977 | 3.4977 |
| SocialMedia_Budget | 0.0021 | 0.0021 |
| Is_Holiday | 26.6990 | 26.6990 |

### Europe (EU) Market
| Parameter | CustomOLS | Sklearn |
|-----------|-----------|---------|
| Intercept | 28.8605 | 28.8605 |
| TV_Budget | 1.5102 | 1.5102 |
| Radio_Budget | 4.7987 | 4.7987 |
| SocialMedia_Budget | 1.2028 | 1.2028 |
| Is_Holiday | 18.2465 | 18.2465 |

## F-Test Results (Advertising Effectiveness)

**Hypothesis**: β_TV = β_Radio = β_Social = 0 (All advertising channels are ineffective)

### North America (NA)
- F-statistic: 54450.3336
- P-value: 0.000000
- Conclusion: Reject H0 - Advertising is effective

### Europe (EU)
- F-statistic: 68786.6881
- P-value: 0.000000
- Conclusion: Reject H0 - Advertising is effective

## Market Analysis

### Key Findings:
1. **TV Budget**: Strong positive impact in both markets
2. **Radio Budget**: Different effectiveness between markets
3. **Social Media**: Varying impact across regions
4. **Holiday Effect**: Seasonal impact analysis

### Business Implications:
- NA market shows different advertising sensitivity compared to EU
- Budget allocation should be market-specific
- Holiday campaigns may need different strategies per region
