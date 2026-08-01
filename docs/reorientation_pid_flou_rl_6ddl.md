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
  - schedules de perturbations constantes, par episode, ou par segments temporels
- Visualisation :
  - `plot_arm_6dof`
  - `plot_control_simulation_6dof`
- Simulations :
  - `experiments/run_pid_dynamic_6dof.py`
  - `experiments/run_pid_fuzzy_gain_dynamic_6dof.py`
  - `experiments/benchmark_pid_factorized_residual_multi_disturbance_6dof.py`
  - `experiments/run_pid_factorized_residual_q_learning_6dof_changing_disturbance.py`
  - `experiments/run_coppelia_tracking_6dof.py`
  - `experiments/train_sac_td3_fuzzy_guided_6dof.py`
  - `experiments/compare_cartesian_tracking_6dof.py`
- Trajectoires :
  - `multi_sine`
  - `point_to_point`
  - `cartesian_loop`
  - `cartesian_point_to_point`
- Passerelle DRL continue :
  - `fuzzy_drl_sim/gym_env.py`
  - wrapper Gymnasium de `FuzzyGuidedTrackingTask`
- Tests :
  - `tests/test_kinematics_6dof.py`
  - `tests/test_dynamics_6dof.py`
  - `tests/test_pid_residual_q_learning_6dof.py`
  - `tests/test_cartesian_trajectory_6dof.py`

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

### Q-learning factorise appris sous perturbations changeantes

Script :

`python experiments/run_pid_factorized_residual_q_learning_6dof_changing_disturbance.py`

Scenario :

- apprentissage sans prior de compensation explicite ;
- perturbation choisie par episode pendant l'apprentissage ;
- evaluation sur les trois profils statiques et sur un profil temporel
  changeant par segments de `120` pas.

Sorties :

- `results/tables/step_39_factorized_q_learning_changing_disturbance_6dof.csv`
- `results/tables/step_39_factorized_q_learning_changing_disturbance_6dof.md`
- `results/figures/step_39_factorized_q_learning_changing_disturbance_6dof.png`
- `results/figures/step_39_factorized_q_learning_changing_disturbance_6dof_learning.png`

Resultats synthetiques :

- etats : `2187` ;
- episodes : `50` ;
- taux de succes des 15 derniers episodes : `0.000` ;
- succes statiques PID adapte : `0/3` ;
- succes statiques PID adapte + Q factorise : `0/3` ;
- profil temporel changeant : distance finale PID adapte `5.1844e-01`,
  distance finale PID adapte + Q factorise `5.1997e-01`.

Interpretation : cette tentative retire bien le prior fourni au benchmark, mais
le Q-learning tabulaire compact ne suffit pas encore. La politique apprise
ameliore parfois la distance finale sur les profils multi-axes statiques, mais
avec des vitesses finales trop elevees, donc sans stabilisation acceptable. Sur
la perturbation temporelle changeante, elle ne depasse pas le PID adapte. Cette
etape valide surtout la difficulte du probleme sans prior et motive une
politique continue plus expressive.

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

### Passerelle SAC/TD3

Fichiers :

- `fuzzy_drl_sim/gym_env.py`
- `fuzzy_drl_sim/rl_task.py`
- `experiments/train_sac_td3_fuzzy_guided_6dof.py`

Le wrapper Gymnasium expose `FuzzyGuidedTrackingTask` avec :

- observation continue de la tache floue ;
- action continue normalisee par articulation dans `[-1, 1]` ;
- mode `residual` par defaut : la politique apprend un residu borne autour du
  PID flou expert, au lieu de remplacer directement la consigne ;
- mode `direct` conserve pour diagnostic ;
- reset fixe a `q=0` par defaut pour rendre les episodes comparables ;
- backend hors-ligne par defaut ;
- backend CoppeliaSim via `--coppelia`.

Commande prevue :

`python experiments/train_sac_td3_fuzzy_guided_6dof.py --algo sac --trajectory cartesian_loop --action-mode residual --residual-scale 0.05 --timesteps 300 --learning-starts 50 --batch-size 32 --eval-episodes 2 --duration 2 --dt 0.05`

Etat actuel : les dependances optionnelles sont installees et le pipeline
d'entrainement fonctionne (`stable-baselines3=True`, `gymnasium=True`,
`torch=True`).

Resultat SAC hors-ligne court :

- sortie : `results/drl_6dof/20260801_110708_offline_sac_cartesian_loop/`
- timesteps : `300` ;
- duree episode : `2 s` ;
- mode : `residual`, `residual_scale=0.05` ;
- evaluation : `2` episodes ;
- reward moyen : `-7.3072` ;
- erreur articulaire finale moyenne : `5.5657e-03` ;
- erreur cartesienne finale moyenne : `6.9387e-03` ;
- violations de contraintes : `0`.

Resultat TD3 hors-ligne court :

- sortie : `results/drl_6dof/20260801_110759_offline_td3_cartesian_loop/`
- timesteps : `300` ;
- reward moyen : `-7.2823` ;
- erreur articulaire finale moyenne : `1.0280e-02` ;
- erreur cartesienne finale moyenne : `1.0569e-02` ;
- violations de contraintes : `0`.

Smoke-test SAC CoppeliaSim :

- commande : `python experiments/train_sac_td3_fuzzy_guided_6dof.py --algo sac --trajectory cartesian_loop --action-mode residual --residual-scale 0.05 --timesteps 20 --learning-starts 5 --batch-size 8 --eval-episodes 1 --duration 0.5 --dt 0.05 --seed 29 --coppelia`
- sortie : `results/drl_6dof/20260801_110759_coppelia_sac_cartesian_loop/`
- erreur cartesienne finale moyenne : `3.6884e-01` ;
- baseline CoppeliaSim fuzzy-PID meme duree : `3.7409e-01` ;
- violations de contraintes : `0`.

Synthese :

- `results/tables/step_40_continuous_drl_residual_6dof.csv`
- `results/tables/step_40_continuous_drl_residual_6dof.md`

Interpretation : l'entrainement SAC/TD3 continu est maintenant operationnel.
Sur un budget de `300` pas, la politique n'est pas encore optimale, mais le
mode residuel preserve le comportement PID flou et produit des evaluations
stables hors-ligne. Le smoke-test CoppeliaSim valide l'integration technique ;
un vrai apprentissage CoppeliaSim demandera des episodes plus longs et beaucoup
plus de pas d'entrainement.

### Trajectoires cartesiennes 6DDL

Les trajectoires de tache ont ete ajoutees dans `fuzzy_drl_sim/trajectory.py` :

- `cartesian_loop` : boucle 3D douce relative a la position initiale de
  l'effecteur ;
- `cartesian_point_to_point` : transfert quintique 3D relatif a la position
  initiale de l'effecteur.

Chaque reference cartesienne est transformee en consigne articulaire par IK
6DDL, puis le suivi est evalue a deux niveaux :

- erreur articulaire `q_ref - q`, compatible avec les simulations CoppeliaSim ;
- erreur position 3D `p_ref - p`, calculee par FK Python a partir des positions
  articulaires mesurees, ou depuis `tip_path` CoppeliaSim si ce chemin est
  renseigne dans `RobotConfig`.

Script :

`python experiments/compare_cartesian_tracking_6dof.py --mode coppelia --controller fuzzy-pid --trajectory cartesian_loop --duration 4 --dt 0.05 --no-plots`

Resultat dry-run Python :

- sortie : `results/cartesian_6dof/20260801_101601_offline_nominal_fuzzy-pid/`
- RMSE articulaire : `7.4961e-02` ;
- erreur articulaire finale : `1.0846e-02` ;
- RMSE cartesienne : `9.7994e-02` ;
- erreur cartesienne finale : `6.3495e-03`.

Resultat CoppeliaSim :

- sortie : `results/cartesian_6dof/20260801_101611_coppelia_nominal_fuzzy-pid/`
- RMSE articulaire : `1.8094e-01` ;
- erreur articulaire finale : `1.0388e-02` ;
- RMSE cartesienne : `9.5613e-02` ;
- erreur cartesienne finale : `5.9011e-03`.

Interpretation : la trajectoire cartesienne relative donne maintenant une base
commune pour comparer le suivi articulaire CoppeliaSim et le suivi position 3D
du modele Python. Les erreurs finales sont faibles sur les deux backends ; les
RMSE restent plus eleves parce que la trajectoire commence par une reorientation
depuis la posture initiale.

## Suite conseillee

1. L'etape Q-learning factorisee 6DDL sur perturbations changeantes est faite
   comme diagnostic : sans prior, la variante tabulaire compacte ne resout pas
   encore la stabilisation.
2. Augmenter progressivement le budget SAC/TD3 hors-ligne sur `cartesian_loop`
   et `cartesian_point_to_point`, puis conserver les meilleurs checkpoints.
3. Lancer une campagne CoppeliaSim plus longue avec le mode residuel, en
   comparant systematiquement fuzzy-PID seul, SAC residuel et TD3 residuel.
4. Utiliser les trajectoires cartesiennes comme objectif SAC/TD3 et renseigner
   `RobotConfig.tip_path` si l'on veut comparer la position reelle du tip
   CoppeliaSim au lieu de la FK Python.
