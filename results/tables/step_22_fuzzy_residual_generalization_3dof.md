# Step 22 - Generalisation flou/RL 3DDL spatiale

| Cible | Methode | Succes | Pas | Distance finale | Couple moyen | Coupure residu |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T1_train (0.95, 0.55, 0.50) | flou | 1 | 372 | 0.009218 | 15.465372 | - |
| T1_train (0.95, 0.55, 0.50) | flou + Q | 1 | 376 | 0.007928 | 15.502739 | - |
| T1_train (0.95, 0.55, 0.50) | flou + Q securise | 1 | 376 | 0.007928 | 15.502739 | 329 |
| T2_left (0.70, -0.75, 0.35) | flou | 1 | 389 | 0.009892 | 15.249992 | - |
| T2_left (0.70, -0.75, 0.35) | flou + Q | 1 | 386 | 0.009803 | 15.163284 | - |
| T2_left (0.70, -0.75, 0.35) | flou + Q securise | 1 | 386 | 0.009803 | 15.163284 | - |
| T3_high (0.55, 0.65, 0.90) | flou | 1 | 389 | 0.009925 | 15.325161 | - |
| T3_high (0.55, 0.65, 0.90) | flou + Q | 1 | 388 | 0.009980 | 15.188268 | - |
| T3_high (0.55, 0.65, 0.90) | flou + Q securise | 1 | 388 | 0.009980 | 15.188268 | - |
| T4_low_far (1.25, 0.35, 0.20) | flou | 1 | 363 | 0.007052 | 15.272914 | - |
| T4_low_far (1.25, 0.35, 0.20) | flou + Q | 1 | 365 | 0.007222 | 15.363112 | - |
| T4_low_far (1.25, 0.35, 0.20) | flou + Q securise | 1 | 365 | 0.007222 | 15.363112 | 324 |
| T5_side (0.25, 1.20, 0.45) | flou | 1 | 423 | 0.007445 | 17.292317 | - |
| T5_side (0.25, 1.20, 0.45) | flou + Q | 1 | 422 | 0.007501 | 17.234816 | - |
| T5_side (0.25, 1.20, 0.45) | flou + Q securise | 1 | 422 | 0.007501 | 17.234816 | 267 |

## Ecarts par rapport au flou seul

| Cible | Methode | Delta pas | Delta distance | Delta couple | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| T1_train | flou + Q | +4 | -0.001290 | +0.037366 | moins rapide que la base floue |
| T1_train | flou + Q securise | +4 | -0.001290 | +0.037366 | residu coupe par securite |
| T2_left | flou + Q | -3 | -0.000089 | -0.086708 | plus rapide |
| T2_left | flou + Q securise | -3 | -0.000089 | -0.086708 | plus rapide |
| T3_high | flou + Q | -1 | +0.000055 | -0.136893 | plus rapide |
| T3_high | flou + Q securise | -1 | +0.000055 | -0.136893 | plus rapide |
| T4_low_far | flou + Q | +2 | +0.000170 | +0.090198 | moins rapide que la base floue |
| T4_low_far | flou + Q securise | +2 | +0.000170 | +0.090198 | residu coupe par securite |
| T5_side | flou + Q | -1 | +0.000056 | -0.057501 | plus rapide |
| T5_side | flou + Q securise | -1 | +0.000056 | -0.057501 | residu coupe, convergence plus rapide que le flou seul |
