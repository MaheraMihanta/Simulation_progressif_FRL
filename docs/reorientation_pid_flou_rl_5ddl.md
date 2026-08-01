# Passage PID flou-RL vers 5DDL

## Objectif

Le document 4DDL montrait que la base floue globale n'est pas extensible :
avec 5 articulations, un controleur flou qui observe erreur et vitesse sur
toutes les articulations demanderait :

`3^(2*5) = 59049 regles`

Le passage au bras 5DDL conserve donc la reorientation retenue :

- PID dynamique en couple calcule comme structure principale ;
- adaptation floue locale des gains `Kp`, `Ki`, `Kd` ;
- RL residuel borne autour du PID adapte, sans remplacer le controleur de base.

## Modele 5DDL retenu

Le bras 5DDL est construit comme extension directe du 4DDL :

- `q0` : lacet de base autour de l'axe vertical ;
- `q1, q2, q3, q4` : chaine planaire 4R dans le plan radial-z ;
- cible : position 3D de l'effecteur.

La cinematique inverse resout la redondance en choisissant un pitch terminal et
un repli terminal entre les deux derniers segments. Par defaut, les deux
segments distaux sont alignes, ce qui donne une posture stable et comparable au
4DDL tout en gardant le cinquieme degre de liberte dans le modele dynamique.

## Complexite conservee

La decomposition reste lineaire en nombre d'articulations :

- adaptation floue : `9n`, donc `45` evaluations locales pour 5DDL ;
- actions RL residuelles : `1 + 2n`, donc `11` actions ;
- etat RL compact : signes des 5 erreurs articulaires + niveau de vitesse,
  donc `3^6 = 729` etats.

On evite ainsi la base floue globale a `59049` regles.

## Implementation ajoutee

- Modele 5DDL :
  - `src/robot/kinematics_5dof.py`
  - `src/robot/dynamics_5dof.py`
  - `src/robot/arm_5dof.py`
- Environnements 5DDL :
  - `src/envs/arm_5dof_env.py`
  - `src/envs/arm_5dof_dynamic_env.py`
- RL residuel 5DDL :
  - `src/rl/pid_residual_q_learning_5dof.py`
- Visualisation :
  - `plot_arm_5dof`
  - `plot_control_simulation_5dof`
- Simulations :
  - `experiments/run_pid_dynamic_5dof.py`
  - `experiments/run_pid_fuzzy_gain_dynamic_5dof.py`
  - `experiments/benchmark_pid_fuzzy_gain_multi_target_5dof.py`
  - `experiments/run_pid_residual_q_learning_5dof.py`
  - `experiments/run_pid_residual_q_learning_5dof_disturbance.py`
  - `experiments/benchmark_terminal_fold_5dof.py`
  - `experiments/benchmark_pid_residual_multi_disturbance_5dof.py`
- Tests :
  - `tests/test_kinematics_5dof.py`
  - `tests/test_dynamics_5dof.py`
  - `tests/test_pid_residual_q_learning_5dof.py`

## Resultats obtenus

### PID dynamique 5DDL

Script :

`python experiments/run_pid_dynamic_5dof.py`

Resultat :

- succes : oui ;
- pas : `242` ;
- distance finale : `9.7132e-03` ;
- vitesse finale : `4.9575e-02` ;
- couple moyen : `4.4813e+01 N.m`.

Figure :

- `results/figures/step_28_pid_dynamic_5dof.png`

### PID a gains flous 5DDL

Script :

`python experiments/run_pid_fuzzy_gain_dynamic_5dof.py`

Resultat :

- succes : oui ;
- pas : `266` ;
- distance finale : `7.3641e-04` ;
- vitesse finale : `7.7064e-02` ;
- couple moyen : `4.6361e+01 N.m` ;
- base floue globale : non utilisee ;
- regles locales : `9` par articulation, donc `45` evaluations locales.

Figure :

- `results/figures/step_29_pid_fuzzy_gain_dynamic_5dof.png`

### Benchmark multi-cibles 5DDL

Script :

`python experiments/benchmark_pid_fuzzy_gain_multi_target_5dof.py`

Sorties :

- `results/tables/step_30_pid_vs_fuzzy_gain_5dof.csv`
- `results/tables/step_30_pid_vs_fuzzy_gain_5dof.md`
- `results/figures/step_30_pid_vs_fuzzy_gain_5dof.png`

Resultats synthetiques :

- PID dynamique : `5/5` succes, distance finale moyenne `8.7078e-03`,
  moyenne `234.8` pas ;
- PID a gains flous : `5/5` succes, distance finale moyenne `2.4413e-03`,
  moyenne `273.4` pas.

Le PID a gains flous est en moyenne plus lent, mais il termine plus pres de la
cible. Cela confirme l'interet de l'adaptation locale des gains sans explosion
combinatoire.

### RL residuel 5DDL

Script :

`python experiments/run_pid_residual_q_learning_5dof.py`

Resultat :

- etats : `729` ;
- actions : `11` ;
- episodes : `45` ;
- taux de succes sur les 15 derniers episodes : `1.000` ;
- rollout appris : succes en `284` pas, distance finale `9.9680e-03` ;
- rollout PID adapte seul : succes en `266` pas, distance finale `7.3641e-04` ;
- superviseur : residu desactive au pas `259`.

Figures :

- `results/figures/step_31_pid_residual_q_learning_5dof.png`
- `results/figures/step_31_pid_residual_q_learning_5dof_learning.png`

Interpretation : sur le modele nominal 5DDL, le residu RL n'ameliore pas encore
le PID adapte. Le superviseur remplit bien son role : si le residu degrade la
progression, il revient au PID a gains flous. Ce resultat est coherent avec le
4DDL : le RL residuel devient surtout utile lorsqu'il compense une perturbation
ou une erreur non modelisee.

### RL residuel 5DDL sous perturbation externe

Script :

`python experiments/run_pid_residual_q_learning_5dof_disturbance.py`

Scenario :

- couple externe non modelise : `(0.0, -4.0, 0.0, 0.0, 0.0) N.m` ;
- residu RL : couple moteur borne ;
- actions : `11`, soit `base`, puis `+/-` sur chaque articulation.

Resultat :

- action apprise : `q1_res+` ;
- PID adapte + RL residuel : succes en `255` pas, distance finale
  `1.6626e-04`, vitesse finale `7.8867e-02` ;
- PID adapte seul : echec a `550` pas, distance finale `1.4159e-02`,
  vitesse finale quasi nulle ;
- taux de succes pendant l'apprentissage bandit : `0.577`.

Sorties :

- `results/tables/step_32_pid_residual_disturbance_5dof.csv`
- `results/tables/step_32_pid_residual_disturbance_5dof.md`
- `results/figures/step_32_pid_residual_disturbance_5dof.png`

Interpretation : la perturbation cree une erreur statique que le PID adapte ne
supprime pas totalement. Le residu RL en couple apprend le biais moteur qui
compense cette erreur, comme dans le scenario 4DDL. Cela confirme que le role le
plus pertinent du RL residuel est la compensation des erreurs non modelisees.

### Redondance du 5DDL par `terminal_fold`

Script :

`python experiments/benchmark_terminal_fold_5dof.py`

Scenario :

- cible conservee : `(1.25, 0.45, 0.60)` ;
- controleur : PID a gains flous en couple calcule ;
- replis testes : `-90`, `-60`, `-30`, `0`, `+30`, `+60`, `+90` degres.

Sorties :

- `results/tables/step_33_terminal_fold_5dof.csv`
- `results/tables/step_33_terminal_fold_5dof.md`
- `results/figures/step_33_terminal_fold_5dof.png`

Resultats synthetiques :

- succes : `7/7` postures ;
- meilleure distance finale : `fold_p30`, distance `4.8865e-04` ;
- effort moyen minimal : `fold_p30`, couple moyen `4.6340e+01 N.m` ;
- convergence la plus rapide : `fold_p90`, `252` pas.

Interpretation : le cinquieme degre de liberte est bien exploitable comme
variable de posture. Toutes les postures atteignent la cible, mais le repli
terminal modifie le compromis entre precision finale, vitesse de convergence,
effort moteur et norme articulaire. Sur cette cible, `terminal_fold = +30 deg`
donne le meilleur compromis precision/effort.

### Perturbations simultanees et limite des actions axis-alignees

Script :

`python experiments/benchmark_pid_residual_multi_disturbance_5dof.py`

Scenario :

- deux perturbations mono-articulaires : `single_q1`, `single_q2` ;
- deux perturbations simultanees : `multi_q1_q2`, `multi_q1_q3` ;
- comparaison entre PID adapte seul, meilleure action residuelle axis-alignee,
  et compensation multi-axes de reference.

Sorties :

- `results/tables/step_34_multi_disturbance_5dof.csv`
- `results/tables/step_34_multi_disturbance_5dof.md`
- `results/figures/step_34_multi_disturbance_5dof.png`

Resultats synthetiques :

- meilleure action axis-alignee : `2/4` succes ;
- compensation multi-axes de reference : `4/4` succes ;
- `single_q1` : `q1_res+` reussit, distance `1.6626e-04` ;
- `single_q2` : `q2_res+` reussit, distance `5.0342e-04` ;
- `multi_q1_q2` : meilleure action unique `q2_res+`, echec,
  distance `1.4159e-02` ; compensation multi-axes, succes, distance
  `3.4184e-04` ;
- `multi_q1_q3` : meilleure action unique `q1_res+`, echec,
  distance `6.2140e-02` ; compensation multi-axes, succes, distance
  `1.8663e-04`.

Interpretation : l'espace d'actions `1 + 2n` est efficace quand un biais moteur
dominant est isole. En revanche, il devient trop limite lorsque plusieurs
biais independants apparaissent simultanement : une seule action axis-alignee
ne peut pas compenser deux articulations a la fois. Pour le passage au 6DDL, il
faudra donc eviter une enumeration combinatoire tout en autorisant des residus
multi-axes, par exemple avec une politique factorisee par articulation.

### Actions residuelles factorisees 5DDL

Script :

`python experiments/benchmark_pid_factorized_residual_multi_disturbance_5dof.py`

Implementation :

- utilitaires factorises generiques :
  - `factorized_residual_action_vector`
  - `factorized_residual_action_label`
- variante Q-learning factorisee :
  - `train_pid_factorized_residual_q_learning_5dof`
  - `rollout_pid_factorized_residual_q_policy_5dof`
- benchmark local sans enumeration cartesienne :
  - `5 articulations x 3 choix x 2 passes = 30` evaluations par scenario ;
  - produit cartesien evite : `3^5 = 243` actions globales.

Sorties :

- `results/tables/step_35_factorized_residual_5dof.csv`
- `results/tables/step_35_factorized_residual_5dof.md`
- `results/figures/step_35_factorized_residual_5dof.png`

Resultats synthetiques :

- meilleure action axis-alignee : `2/4` succes ;
- action factorisee : `4/4` succes ;
- `single_q1` : action factorisee `q1_res+`, succes, distance
  `1.6626e-04` ;
- `single_q2` : action factorisee `q2_res+`, succes, distance
  `5.0342e-04` ;
- `multi_q1_q2` : action factorisee `q1_res+,q2_res+`, succes, distance
  `3.4184e-04` ;
- `multi_q1_q3` : action factorisee `q1_res+,q3_res+`, succes, distance
  `1.8663e-04`.

Interpretation : l'action factorisee garde une complexite lineaire en nombre
d'articulations, car chaque articulation choisit localement entre `base`,
`res+` et `res-`. Elle corrige cependant plusieurs biais simultanes, ce que
l'action residuelle unique axis-alignee ne permettait pas.

## Suite conseillee

1. Passer ensuite au 6DDL avec la meme architecture : PID vectoriel, gains flous
   locaux, choix de posture redondant et RL residuel borne/factorise.
2. Entrainer et comparer la variante Q-learning factorisee sur des perturbations
   non constantes ou changeantes, afin de depasser le benchmark local a residu
   fixe.

La premiere partie de cette suite est maintenant documentee dans
`docs/reorientation_pid_flou_rl_6ddl.md`.
