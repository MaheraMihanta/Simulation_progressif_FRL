# Actions residuelles factorisees 5DDL

Recherche locale : `5 articulations x 3 choix x 2 passes = 30` evaluations par scenario, sans enumeration du produit cartesien `3^5 = 243`.

| Scenario | Perturbation | Controleur | Action | Succes | Pas | Distance finale | Couple moyen |
|---|---:|---|---:|---:|---:|---:|---:|
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | action_axis_alignee | base | False | 350 | 1.4313e-02 | 4.1516e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | action_axis_alignee | q1_res+ | True | 255 | 1.6626e-04 | 4.8308e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | action_factorisee | q1_res+ | True | 255 | 1.6626e-04 | 4.8308e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | action_axis_alignee | base | False | 350 | 8.3970e-02 | 4.2240e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | action_axis_alignee | q2_res+ | True | 270 | 5.0342e-04 | 4.6463e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | action_factorisee | q2_res+ | True | 270 | 5.0342e-04 | 4.6463e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | action_axis_alignee | base | False | 350 | 1.1282e-01 | 4.5248e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | action_axis_alignee | q2_res+ | False | 350 | 1.4433e-02 | 4.2908e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | action_factorisee | q1_res+,q2_res+ | True | 266 | 3.4184e-04 | 4.9291e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | action_axis_alignee | base | False | 350 | 7.9562e-02 | 3.9440e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | action_axis_alignee | q1_res+ | False | 350 | 6.2067e-02 | 3.9352e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | action_factorisee | q1_res+,q3_res+ | True | 251 | 1.8663e-04 | 4.7731e+01 |

Interpretation : l'action factorisee choisit un signe local par
articulation. Elle conserve une complexite lineaire en nombre
d'articulations tout en autorisant des corrections simultanees, ce
qui supprime la limite observee avec l'action axis-alignee unique.
