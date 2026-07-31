# Benchmark redondance terminal_fold 5DDL

| Repli | Degres | Succes | Pas | Distance finale | Vitesse finale | Couple moyen | Norme posture |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold_m90 | -90.0 | True | 278 | 1.1082e-03 | 7.9899e-02 | 4.7957e+01 | 3.0068e+00 |
| fold_m60 | -60.0 | True | 274 | 8.2986e-04 | 7.7684e-02 | 4.7668e+01 | 3.0631e+00 |
| fold_m30 | -30.0 | True | 267 | 6.8795e-04 | 7.9065e-02 | 4.7209e+01 | 3.1886e+00 |
| fold_0 | 0.0 | True | 266 | 7.3641e-04 | 7.7064e-02 | 4.6361e+01 | 3.3352e+00 |
| fold_p30 | 30.0 | True | 264 | 4.8865e-04 | 7.8048e-02 | 4.6340e+01 | 3.4700e+00 |
| fold_p60 | 60.0 | True | 259 | 1.6717e-03 | 7.7009e-02 | 4.7320e+01 | 3.5820e+00 |
| fold_p90 | 90.0 | True | 252 | 4.7862e-03 | 7.8181e-02 | 4.9417e+01 | 3.6765e+00 |

Synthese :

- meilleure distance finale : `fold_p30` avec `4.8865e-04` ;
- effort moyen minimal : `fold_p30` avec `4.6340e+01 N.m` ;
- convergence la plus rapide : `fold_p90` en `252` pas.

Interpretation : le cinquieme DDL peut etre exploite comme un choix
de posture. Toutes les postures testees atteignent la cible, mais le
repli distal deplace le compromis entre distance finale, temps de
convergence, effort et norme articulaire.
