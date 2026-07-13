# Step 12 - Generalisation flou/RL securisee

| Cible | Methode | Succes | Pas | Distance finale | Couple moyen | Coupure residu |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T1_train (1.10, 0.55) | flou | 1 | 360 | 0.005227 | 13.853263 | - |
| T1_train (1.10, 0.55) | flou + Q | 1 | 269 | 0.005537 | 14.420214 | - |
| T1_train (1.10, 0.55) | flou + Q securise | 1 | 269 | 0.005537 | 14.420214 | - |
| T2_diag (0.85, 0.85) | flou | 1 | 361 | 0.006030 | 12.683325 | - |
| T2_diag (0.85, 0.85) | flou + Q | 1 | 271 | 0.004960 | 12.982476 | - |
| T2_diag (0.85, 0.85) | flou + Q securise | 1 | 363 | 0.006121 | 12.719937 | 145 |
| T3_low (1.25, 0.25) | flou | 1 | 358 | 0.004593 | 14.835777 | - |
| T3_low (1.25, 0.25) | flou + Q | 1 | 269 | 0.009732 | 15.800595 | - |
| T3_low (1.25, 0.25) | flou + Q securise | 1 | 269 | 0.009732 | 15.800595 | - |
| T4_high (0.65, 1.05) | flou | 1 | 400 | 0.002534 | 13.500462 | - |
| T4_high (0.65, 1.05) | flou + Q | 0 | 550 | 0.030002 | 12.991741 | - |
| T4_high (0.65, 1.05) | flou + Q securise | 1 | 389 | 0.009925 | 13.731230 | 248 |
| T5_far (1.35, 0.45) | flou | 1 | 349 | 0.002453 | 14.805810 | - |
| T5_far (1.35, 0.45) | flou + Q | 1 | 263 | 0.003264 | 15.287970 | - |
| T5_far (1.35, 0.45) | flou + Q securise | 1 | 263 | 0.003264 | 15.287970 | - |

## Ecarts par rapport au flou seul

| Cible | Methode | Delta pas | Delta distance | Delta couple | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| T1_train | flou + Q | -91 | +0.000311 | +0.566951 | plus rapide, effort plus eleve |
| T1_train | flou + Q securise | -91 | +0.000311 | +0.566951 | plus rapide, effort plus eleve |
| T2_diag | flou + Q | -90 | -0.001070 | +0.299151 | plus rapide, effort plus eleve |
| T2_diag | flou + Q securise | +2 | +0.000091 | +0.036612 | residu coupe par securite |
| T3_low | flou + Q | -89 | +0.005138 | +0.964818 | plus rapide, effort plus eleve |
| T3_low | flou + Q securise | -89 | +0.005138 | +0.964818 | plus rapide, effort plus eleve |
| T4_high | flou + Q | +150 | +0.027468 | -0.508721 | degradation: pas de convergence |
| T4_high | flou + Q securise | -11 | +0.007392 | +0.230767 | residu coupe, convergence plus rapide que le flou seul |
| T5_far | flou + Q | -86 | +0.000812 | +0.482159 | plus rapide, effort plus eleve |
| T5_far | flou + Q securise | -86 | +0.000812 | +0.482159 | plus rapide, effort plus eleve |
