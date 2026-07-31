# RL residuel 4DDL sous perturbation externe

Couple externe applique : `(0.0, -4.0, 0.0, 0.0)` N.m.
Mode residuel : couple moteur borne, action apprise `q1_res+`.

| Controleur | Action | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
|---|---:|---:|---:|---:|---:|---:|
| PID adapte + RL residuel | q1_res+ | True | 204 | 1.5262e-03 | 7.8301e-02 | 2.6549e+01 |
| PID adapte seul | base | False | 500 | 1.0812e-01 | 1.3180e-05 | 2.2930e+01 |

Valeurs Q finales du bandit residuel :

| Action | Valeur Q | Essais |
|---|---:|---:|
| base | -9.3765e+01 | 1 |
| q0_res+ | -9.3866e+01 | 1 |
| q1_res+ | 5.8386e+01 | 16 |
| q2_res+ | -1.2720e+02 | 1 |
| q3_res+ | -5.6874e+02 | 1 |
| q0_res- | -9.3748e+01 | 1 |
| q1_res- | -1.4434e+02 | 1 |
| q2_res- | -1.2468e+02 | 1 |
| q3_res- | -3.6591e+02 | 1 |

Interpretation : le PID adapte seul garde une erreur statique sous
perturbation constante. Le residu RL en couple apprend l'action qui
annule le biais moteur dominant et ramene la trajectoire dans la
tolerance.
