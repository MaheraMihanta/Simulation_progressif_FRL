# Benchmark multi-cibles 4DDL

| Cible | Controleur | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
|---|---:|---:|---:|---:|---:|---:|
| T1_reference | PID | True | 192 | 5.9738e-03 | 7.9083e-02 | 2.2504e+01 |
| T1_reference | PID_gains_flous | True | 204 | 1.9478e-03 | 7.9691e-02 | 2.2819e+01 |
| T2_lateral_bas | PID | True | 175 | 9.9573e-03 | 6.3815e-02 | 2.2352e+01 |
| T2_lateral_bas | PID_gains_flous | True | 199 | 3.0995e-03 | 7.8637e-02 | 2.2702e+01 |
| T3_haut_diagonal | PID | True | 163 | 8.4770e-03 | 7.5010e-02 | 3.3055e+01 |
| T3_haut_diagonal | PID_gains_flous | True | 192 | 9.8852e-03 | 5.0356e-02 | 3.1948e+01 |
| T4_avant_droit | PID | True | 146 | 5.1332e-03 | 7.5806e-02 | 2.4897e+01 |
| T4_avant_droit | PID_gains_flous | True | 232 | 2.6595e-03 | 7.8012e-02 | 3.8627e+01 |
| T5_arriere_haut | PID | True | 194 | 5.5417e-03 | 7.9725e-02 | 2.5152e+01 |
| T5_arriere_haut | PID_gains_flous | True | 205 | 2.1948e-03 | 7.9737e-02 | 2.5481e+01 |

Interpretation courte : le PID a gains flous conserve la meme
architecture de commande que le PID dynamique, mais module localement
`Kp`, `Ki` et `Kd` sans base floue globale.
