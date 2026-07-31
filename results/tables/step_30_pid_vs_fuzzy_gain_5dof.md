# Benchmark multi-cibles 5DDL

| Cible | Controleur | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
|---|---:|---:|---:|---:|---:|---:|
| T1_reference | PID | True | 242 | 9.7132e-03 | 4.9575e-02 | 4.4813e+01 |
| T1_reference | PID_gains_flous | True | 266 | 7.3641e-04 | 7.7064e-02 | 4.6361e+01 |
| T2_lateral_bas | PID | True | 165 | 9.8870e-03 | 5.3350e-02 | 2.9950e+01 |
| T2_lateral_bas | PID_gains_flous | True | 264 | 1.5590e-04 | 7.8952e-02 | 4.8537e+01 |
| T3_haut_diagonal | PID | True | 243 | 9.6592e-03 | 5.2056e-02 | 4.6728e+01 |
| T3_haut_diagonal | PID_gains_flous | True | 255 | 9.1601e-04 | 7.8611e-02 | 4.9853e+01 |
| T4_avant_droit | PID | True | 284 | 4.3623e-03 | 7.8540e-02 | 5.2014e+01 |
| T4_avant_droit | PID_gains_flous | True | 322 | 9.8299e-03 | 4.4285e-02 | 5.8650e+01 |
| T5_arriere_haut | PID | True | 240 | 9.9172e-03 | 4.8682e-02 | 4.5698e+01 |
| T5_arriere_haut | PID_gains_flous | True | 260 | 5.6841e-04 | 7.7980e-02 | 4.7341e+01 |

Interpretation courte : le passage au 5DDL garde le meme schema
PID couple calcule + adaptation locale des gains. La logique floue
reste lineaire en nombre d'articulations : 45 evaluations locales,
sans base floue globale.
