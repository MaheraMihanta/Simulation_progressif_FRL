# Perturbations multiples 5DDL

| Scenario | Perturbation | Controleur | Action | Succes | Pas | Distance finale | Couple moyen |
|---|---:|---|---:|---:|---:|---:|---:|
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | action_axis_alignee | base | False | 550 | 1.4159e-02 | 3.4826e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | action_axis_alignee | q1_res+ | True | 255 | 1.6626e-04 | 4.8308e+01 |
| single_q1 | `(0.0, -4.0, 0.0, 0.0, 0.0)` | compensation_multi_axes | multi_axis_reference | True | 255 | 1.6626e-04 | 4.8308e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | action_axis_alignee | base | False | 550 | 8.3670e-02 | 3.4050e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | action_axis_alignee | q2_res+ | True | 270 | 5.0342e-04 | 4.6463e+01 |
| single_q2 | `(0.0, 0.0, -3.0, 0.0, 0.0)` | compensation_multi_axes | multi_axis_reference | True | 270 | 5.0342e-04 | 4.6463e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | action_axis_alignee | base | False | 550 | 1.1085e-01 | 3.7492e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | action_axis_alignee | q2_res+ | False | 550 | 1.4159e-02 | 3.5643e+01 |
| multi_q1_q2 | `(0.0, -4.0, -3.0, 0.0, 0.0)` | compensation_multi_axes | multi_axis_reference | True | 266 | 3.4184e-04 | 4.9291e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | action_axis_alignee | base | False | 550 | 7.9406e-02 | 3.3580e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | action_axis_alignee | q1_res+ | False | 550 | 6.2140e-02 | 3.3413e+01 |
| multi_q1_q3 | `(0.0, -4.0, 0.0, -1.0, 0.0)` | compensation_multi_axes | multi_axis_reference | True | 251 | 1.8663e-04 | 4.7731e+01 |

Interpretation : les perturbations mono-articulaires sont corrigees
par une action axis-alignee unique. Les perturbations simultanees
restent hors tolerance avec une seule action, alors qu'une compensation
multi-axes de reference reussit. L'espace d'actions `1 + 2n` est donc
suffisant pour des biais dominants isoles, mais limite pour plusieurs
biais independants appliques en meme temps.
