# Step 11 - Generalisation flou/RL

| Cible | Methode | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T1_train (1.10, 0.55) | fuzzy_base | 1 | 360 | 0.005227 | 0.077840 | 13.853263 |
| T1_train (1.10, 0.55) | fuzzy_rl | 1 | 269 | 0.005537 | 0.075709 | 14.420214 |
| T2_diag (0.85, 0.85) | fuzzy_base | 1 | 361 | 0.006030 | 0.079733 | 12.683325 |
| T2_diag (0.85, 0.85) | fuzzy_rl | 1 | 271 | 0.004960 | 0.078132 | 12.982476 |
| T3_low (1.25, 0.25) | fuzzy_base | 1 | 358 | 0.004593 | 0.076931 | 14.835777 |
| T3_low (1.25, 0.25) | fuzzy_rl | 1 | 269 | 0.009732 | 0.065777 | 15.800595 |
| T4_high (0.65, 1.05) | fuzzy_base | 1 | 400 | 0.002534 | 0.077762 | 13.500462 |
| T4_high (0.65, 1.05) | fuzzy_rl | 0 | 550 | 0.030002 | 0.011906 | 12.991741 |
| T5_far (1.35, 0.45) | fuzzy_base | 1 | 349 | 0.002453 | 0.079276 | 14.805810 |
| T5_far (1.35, 0.45) | fuzzy_rl | 1 | 263 | 0.003264 | 0.072352 | 15.287970 |

## Ecarts flou/RL - flou seul

| Cible | Delta pas | Delta distance | Delta couple | Interpretation courte |
| --- | ---: | ---: | ---: | --- |
| T1_train | -91 | +0.000311 | +0.566951 | plus rapide, mais plus couteux en effort |
| T2_diag | -90 | -0.001070 | +0.299151 | plus rapide, mais plus couteux en effort |
| T3_low | -89 | +0.005138 | +0.964818 | plus rapide, mais plus couteux en effort |
| T4_high | +150 | +0.027468 | -0.508721 | degradation: la politique apprise ne converge pas |
| T5_far | -86 | +0.000812 | +0.482159 | plus rapide, mais plus couteux en effort |
