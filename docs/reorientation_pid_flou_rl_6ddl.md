# Passage PID flou-RL vers 6DDL

## Objectif

Le passage 6DDL reprend la conclusion du 5DDL : le residu RL doit rester borne
autour d'un controleur PID dynamique a gains flous, mais l'action residuelle ne
doit plus etre limitee a un seul axe actif. Le modele conserve donc :

- PID dynamique en couple calcule comme controleur principal ;
- adaptation floue locale des gains `Kp`, `Ki`, `Kd` ;
- residu RL borne, avec action factorisee par articulation ;
- validation CoppeliaSim 6 articulations via la scene NiryoOne.

## Modele 6DDL retenu

Le bras Python 6DDL est l'extension directe du 5DDL :

- `q0` : lacet de base autour de l'axe vertical ;
- `q1, q2, q3, q4, q5` : chaine planaire 5R dans le plan radial-z ;
- cible : position 3D de l'effecteur.

L'IK analytique conserve une structure simple : les trois derniers segments
sont remplaces par un lien distal virtuel. La posture redondante est choisie par
un pitch distal et deux replis :

- `wrist_fold` : repli entre les deux premiers liens distaux ;
- `terminal_fold` : repli terminal ;
- par defaut, les deux replis valent zero, donc les liens distaux sont alignes.

## Complexite conservee

- adaptation floue : `9n`, donc `54` evaluations locales pour 6DDL ;
- actions residuelles axis-alignees : `1 + 2n`, donc `13` actions ;
- etat RL compact : signes des 6 erreurs articulaires + niveau de vitesse,
  donc `3^7 = 2187` etats ;
- action factorisee : `6 x 3` choix locaux, sans enumeration des `3^6 = 729`
  actions globales.

## Implementation ajoutee

- Modele 6DDL :
  - `src/robot/kinematics_6dof.py`
  - `src/robot/dynamics_6dof.py`
  - `src/robot/arm_6dof.py`
- Environnements 6DDL :
  - `src/envs/arm_6dof_env.py`
  - `src/envs/arm_6dof_dynamic_env.py`
- RL residuel 6DDL :
  - `src/rl/pid_residual_q_learning_6dof.py`
- Visualisation :
  - `plot_arm_6dof`
  - `plot_control_simulation_6dof`
- Simulations :
  - `experiments/run_pid_dynamic_6dof.py`
  - `experiments/run_pid_fuzzy_gain_dynamic_6dof.py`
  - `experiments/benchmark_pid_factorized_residual_multi_disturbance_6dof.py`
  - `experiments/run_coppelia_tracking_6dof.py`
- Tests :
  - `tests/test_kinematics_6dof.py`
  - `tests/test_dynamics_6dof.py`
  - `tests/test_pid_residual_q_learning_6dof.py`

## Resultats obtenus

### PID dynamique 6DDL

Script :

`python experiments/run_pid_dynamic_6dof.py`

Resultat :

- succes : oui ;
- pas : `265` ;
- distance finale : `5.2608e-03` ;
- vitesse finale : `7.8199e-02` ;
- couple moyen : `4.1707e+01 N.m`.

Figure :

- `results/figures/step_36_pid_dynamic_6dof.png`

### PID a gains flous 6DDL

Script :

`python experiments/run_pid_fuzzy_gain_dynamic_6dof.py`

Resultat :

- succes : oui ;
- pas : `342` ;
- distance finale : `5.9302e-03` ;
- vitesse finale : `7.9542e-02` ;
- couple moyen : `5.2218e+01 N.m` ;
- base floue globale : non utilisee ;
- regles locales : `9` par articulation, donc `54` evaluations locales.

Figure :

- `results/figures/step_37_pid_fuzzy_gain_dynamic_6dof.png`

### Actions residuelles factorisees 6DDL

Script :

`python experiments/benchmark_pid_factorized_residual_multi_disturbance_6dof.py`

Scenario :

- `single_q1` : perturbation mono-articulaire ;
- `multi_q1_q2` : perturbations simultanees ;
- `multi_q1_q3_q5` : perturbations simultanees incluant l'axe terminal.

Sorties :

- `results/tables/step_38_factorized_residual_6dof.csv`
- `results/tables/step_38_factorized_residual_6dof.md`
- `results/figures/step_38_factorized_residual_6dof.png`

Resultats synthetiques :

- meilleure action axis-alignee : `1/3` succes ;
- action factorisee : `3/3` succes ;
- evaluations factorisees : `18` par scenario ;
- produit cartesien evite : `729` actions globales.

Interpretation : le 6DDL confirme la limite de l'action residuelle unique. Une
commande factorisee peut activer plusieurs compensations en meme temps, par
exemple `q1_res+,q2_res+` ou `q1_res+,q3_res+,q5_res+`, tout en gardant une
complexite lineaire. Le benchmark utilise un prior de compensation construit a
partir du couple perturbateur connu ; la suite naturelle est de remplacer ce
prior par une politique apprise a partir des recompenses.

### Validation CoppeliaSim 6DDL

Script :

`python experiments/run_coppelia_tracking_6dof.py --controller fuzzy-pid --duration 1 --dt 0.05 --no-plots`

Scene :

- `bras_manipulateur_niryoOne.ttt`
- backend ZeroMQ Remote API ;
- chemins articulaires NiryoOne deja definis dans `fuzzy_drl_sim/config.py`.

Resultat court obtenu avec CoppeliaSim ouvert :

- backend : `coppeliasim` ;
- controleur : `fuzzy-pid` ;
- RMSE articulaire : `6.6180e-03` ;
- erreur articulaire maximale : `1.2633e-02` ;
- erreur finale : `1.4174e-02` ;
- violations de contraintes : `0`.

Sortie :

- `results/coppelia_6dof/20260801_091755_coppelia_nominal_fuzzy-pid/`

## Suite conseillee

1. Entrainer la variante Q-learning factorisee 6DDL sur perturbations changeantes,
   pour apprendre le prior de compensation au lieu de le fournir.
2. Etendre `FuzzyGuidedTrackingTask` vers un vrai entrainement continu SAC/TD3
   dans CoppeliaSim, en gardant l'action normalisee par articulation.
3. Ajouter des trajectoires de tache cartesiens pour comparer suivi articulaire
   CoppeliaSim et suivi position 3D du modele Python.
