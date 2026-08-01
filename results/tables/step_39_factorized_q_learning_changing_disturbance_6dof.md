# Q-learning factorise 6DDL sous perturbations changeantes

Profil de perturbation par segments de `120` pas :

- segment 1 / `single_q1` : `(0.0, -4.0, 0.0, 0.0, 0.0, 0.0)`
- segment 2 / `multi_q1_q2` : `(0.0, -4.0, -3.0, 0.0, 0.0, 0.0)`
- segment 3 / `multi_q1_q3_q5` : `(0.0, -4.0, 0.0, -1.0, 0.0, -0.8)`

| Profil | Controleur | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |
|---|---|---:|---:|---:|---:|---:|
| single_q1 | pid_adapte | False | 420 | 1.1052e-01 | 6.7460e-03 | 5.1397e+01 |
| single_q1 | pid_adapte_q_factorise | False | 420 | 5.2882e-01 | 7.7065e-01 | 4.6985e+01 |
| multi_q1_q2 | pid_adapte | False | 420 | 7.7933e-02 | 7.1239e-02 | 5.4910e+01 |
| multi_q1_q2 | pid_adapte_q_factorise | False | 420 | 3.8088e-02 | 7.0925e+00 | 3.9288e+01 |
| multi_q1_q3_q5 | pid_adapte | False | 420 | 5.1825e-01 | 1.8938e-03 | 5.6794e+01 |
| multi_q1_q3_q5 | pid_adapte_q_factorise | False | 420 | 1.7753e-01 | 2.8493e+00 | 5.0542e+01 |
| changing_schedule | pid_adapte | False | 420 | 5.1844e-01 | 7.6165e-02 | 5.2171e+01 |
| changing_schedule | pid_adapte_q_factorise | False | 420 | 5.1997e-01 | 3.7272e-01 | 4.6813e+01 |

Taux de succes des 15 derniers episodes : `0.000`.

Actions apprises dominantes :

- `base` : `92` pas
- `q0_res-,q2_res-,q3_res-,q5_res+` : `46` pas
- `q0_res+,q1_res-,q3_res-,q4_res+,q5_res+` : `38` pas
- `q0_res+,q1_res-,q2_res-,q3_res+,q4_res+,q5_res+` : `28` pas
- `q1_res+,q3_res+` : `27` pas
- `q0_res-,q1_res-,q2_res-,q3_res-,q4_res-` : `27` pas
- `q0_res-,q1_res+,q4_res-,q5_res-` : `26` pas
- `q3_res-,q4_res-` : `18` pas

Interpretation : cette experience retire le prior explicite du benchmark
factorise. La politique doit choisir les signes locaux a partir de
l'erreur articulaire et de la vitesse. Les perturbations changent
d'un episode a l'autre pendant l'apprentissage, puis la politique est
testee sur chaque profil et sur un profil temporel changeant.

Resultat actuel : le probleme n'est pas encore resolu par cette
variante tabulaire compacte. La politique reduit parfois la distance
finale sur des perturbations multi-axes statiques, mais elle garde des
vitesses finales trop elevees et ne produit aucun episode reussi dans
la fenetre finale. Sur le profil temporel changeant, elle ne depasse
pas le PID adapte. Cette etape sert donc de diagnostic avant de passer
a une politique continue plus expressive.
