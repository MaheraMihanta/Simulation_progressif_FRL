# Actions residuelles factorisees 6DDL

Recherche locale : `6 articulations x 3 choix x 1 passes = 18` evaluations par scenario, sans enumeration du produit cartesien `3^6 = 729`.
La recherche est initialisee par un prior de compensation construit a
partir du couple perturbateur connu dans ce benchmark synthetique.

| Scenario | Perturbation | Controleur | Action | Succes | Pas | Distance finale | Couple moyen |
|---|---:|---|---:|---:|---:|---:|---:|
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0, 0.0)` | action_axis_alignee | base | False | 360 | 1.1339e-01 | 5.6011e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0, 0.0)` | action_axis_alignee | q1_res+ | True | 343 | 8.1792e-03 | 5.7486e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0, 0.0)` | action_factorisee | q1_res+ | True | 343 | 8.1792e-03 | 5.7486e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0, 0.0)` | action_axis_alignee | base | False | 360 | 7.2886e-02 | 6.0127e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0, 0.0)` | action_axis_alignee | q2_res+ | False | 360 | 1.1160e-01 | 4.3883e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0, 0.0)` | action_factorisee | q1_res+,q2_res+ | True | 272 | 2.9518e-03 | 5.0068e+01 |
| multi_q1_q3_q5 | `(0.0, -4.0, 0.0, -1.0, 0.0, -0.8)` | action_axis_alignee | base | False | 360 | 5.1818e-01 | 6.1854e+01 |
| multi_q1_q3_q5 | `(0.0, -4.0, 0.0, -1.0, 0.0, -0.8)` | action_axis_alignee | q5_res+ | False | 360 | 6.1090e-02 | 5.5354e+01 |
| multi_q1_q3_q5 | `(0.0, -4.0, 0.0, -1.0, 0.0, -0.8)` | action_factorisee | q1_res+,q3_res+,q5_res+ | True | 350 | 6.8530e-03 | 5.6724e+01 |

Interpretation : l'action factorisee choisit un signe local par
articulation. Elle conserve une complexite lineaire en nombre
d'articulations tout en autorisant des corrections simultanees, ce
qui supprime la limite observee avec l'action axis-alignee unique.
