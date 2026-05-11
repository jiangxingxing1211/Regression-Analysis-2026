# Scenario B — Real-World Market Analysis

Dataset: `q3_marketing.csv`  ·  NA rows: 500  ·  EU rows: 500

---

## North America (NA)

R² = **0.9970**

Coefficients:
  - Intercept: +48.1036
  - TV: +3.5075
  - Radio: +3.4977
  - SocialMedia: +0.0021
  - Holiday: +26.6990

F-Test — H₀: β_TV = β_Radio = β_SocialMedia = 0

- F-stat = 54450.3336,  p-value = 0.00e+00

---

## Europe (EU)

R² = **0.9976**

Coefficients:
  - Intercept: +28.8605
  - TV: +1.5102
  - Radio: +4.7987
  - SocialMedia: +1.2028
  - Holiday: +18.2465

F-Test — H₀: β_TV = β_Radio = β_SocialMedia = 0

- F-stat = 68786.6881,  p-value = 0.00e+00

---

## Conclusions

✅ NA: advertising channels are **EFFECTIVE**  (F = 54450.33, p = 0.00e+00)
✅ EU: advertising channels are **EFFECTIVE**  (F = 68786.69, p = 0.00e+00)

> Residual plots saved to `market_comparison.png`
