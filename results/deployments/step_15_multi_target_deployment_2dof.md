# Step 15 - Deploiement multi-cibles 2 DDL

La politique flou/RL est entrainee une fois, sauvegardee dans un artefact
`npz`, rechargee, puis deployee sur une sequence de cibles sans remise a
zero du robot entre deux cibles.

| Cible | Methode | Succes | Pas | Distance finale | Vitesse finale | Couple moyen | Coupure residu |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D1_train (1.10, 0.55) | flou seul | 1 | 360 | 0.005227 | 0.077840 | 13.853263 | - |
| D1_train (1.10, 0.55) | flou + Q securise deploye | 1 | 270 | 0.004881 | 0.079945 | 14.570781 | - |
| D2_diag (0.85, 0.85) | flou seul | 1 | 221 | 0.006878 | 0.076844 | 13.632396 | - |
| D2_diag (0.85, 0.85) | flou + Q securise deploye | 1 | 221 | 0.006666 | 0.075519 | 13.696226 | 118 |
| D3_low (1.25, 0.25) | flou seul | 1 | 259 | 0.005632 | 0.074208 | 15.692195 | - |
| D3_low (1.25, 0.25) | flou + Q securise deploye | 1 | 264 | 0.009827 | 0.079627 | 15.794948 | 189 |
| D4_high (0.65, 1.05) | flou seul | 1 | 267 | 0.009325 | 0.079046 | 16.391588 | - |
| D4_high (0.65, 1.05) | flou + Q securise deploye | 1 | 266 | 0.008018 | 0.075580 | 16.188886 | 127 |
| D5_far (1.35, 0.45) | flou seul | 1 | 259 | 0.005959 | 0.076387 | 17.620459 | - |
| D5_far (1.35, 0.45) | flou + Q securise deploye | 1 | 288 | 0.009247 | 0.069563 | 16.930252 | 222 |

## Ecarts du controleur deploye par rapport au flou seul

| Cible | Delta pas | Delta distance | Delta couple | Interpretation |
| --- | ---: | ---: | ---: | --- |
| D1_train | -90 | -0.000346 | +0.717518 | plus rapide, effort plus eleve |
| D2_diag | +0 | -0.000212 | +0.063830 | residu coupe par supervision |
| D3_low | +5 | +0.004195 | +0.102753 | residu coupe par supervision |
| D4_high | -1 | -0.001308 | -0.202702 | residu coupe par supervision |
| D5_far | +29 | +0.003288 | -0.690208 | residu coupe par supervision |
