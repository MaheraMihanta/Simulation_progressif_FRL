# RL residuel 5DDL sous perturbation externe

Couple externe applique : `(0.0, -4.0, 0.0, 0.0, 0.0)` N.m.
Mode residuel : couple moteur borne, action apprise `q1_res+`.

| Controleur | Action | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
|---|---:|---:|---:|---:|---:|---:|
| PID adapte + RL residuel | q1_res+ | True | 255 | 1.6626e-04 | 7.8867e-02 | 4.8308e+01 |
| PID adapte seul | base | False | 550 | 1.4159e-02 | 1.5333e-05 | 3.4826e+01 |

Valeurs Q finales du bandit residuel :

| Action | Valeur Q | Essais |
|---|---:|---:|
| base | -2.0709e+02 | 1 |
| q0_res+ | -2.0731e+02 | 1 |
| q1_res+ | -1.0127e+02 | 15 |
| q2_res+ | -2.2269e+02 | 1 |
| q3_res+ | -3.0261e+02 | 1 |
| q4_res+ | -5.3502e+02 | 1 |
| q0_res- | -2.0731e+02 | 1 |
| q1_res- | -2.1891e+02 | 1 |
| q2_res- | -2.4515e+02 | 2 |
| q3_res- | -2.2538e+02 | 1 |
| q4_res- | -5.8898e+02 | 1 |

Interpretation : le PID adapte seul garde une erreur statique sous
perturbation constante. Le residu RL en couple apprend l'action qui
annule le biais moteur dominant et ramene la trajectoire dans la
tolerance.
