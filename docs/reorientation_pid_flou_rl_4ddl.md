# Reorientation PID flou-RL vers 4DDL

## Constat

La base floue globale devient rapidement non traitable. Avec 3 termes par
variable, un controleur flou qui observe erreur et vitesse sur `n` articulations
demande `3^(2n)` regles :

- 2DDL : 81 regles.
- 3DDL : 729 regles.
- 4DDL : 6561 regles.
- 6DDL : 531441 regles.

Cette croissance confirme que la logique floue pleine echelle n'est pas une
bonne base pour les extensions prevues.

## Option retenue

La piste PID a gains adaptes est plus raisonnable. Le PID conserve une structure
interpretable et robuste, tandis que la logique floue ne sert plus a enumerer
tous les etats multi-articulaires. Elle ajuste localement `Kp`, `Ki` et `Kd`
pour chaque articulation avec une petite table `3 x 3` sur :

- magnitude de l'erreur articulaire ;
- magnitude de la derivee de l'erreur.

Le cout devient lineaire : 9 petites regles par articulation, donc 36 evaluations
locales pour 4DDL et 54 pour 6DDL, au lieu de centaines de milliers de regles.

## Role du RL

Le RL est introduit comme residu de commande, pas comme remplacement du PID. La
commande appliquee est :

`q_ddot_cmd = q_ddot_PID_flou + q_ddot_RL_residuel`

Le premier prototype 4DDL utilise une table Q compacte :

- etat : signes des 4 erreurs articulaires + niveau de vitesse ;
- nombre d'etats : `3^5 = 243` ;
- actions : 9 residus, soit aucune action puis `+/-` sur chaque articulation.

Cette formulation garde le probleme manipulable et permet d'etudier si le RL
ameliore la convergence ou l'effort sans destabiliser le controleur classique.

## Implementation ajoutee

- Modele 4DDL : `src/robot/kinematics_4dof.py`, `src/robot/dynamics_4dof.py`.
- Environnements 4DDL : `src/envs/arm_4dof_env.py`,
  `src/envs/arm_4dof_dynamic_env.py`.
- PID a gains flous : `FuzzyGainScheduledPIDController`.
- RL residuel 4DDL : `src/rl/pid_residual_q_learning_4dof.py`.
- Simulations :
  - `experiments/run_pid_dynamic_4dof.py`
  - `experiments/run_pid_fuzzy_gain_dynamic_4dof.py`
  - `experiments/run_pid_residual_q_learning_4dof.py`

## Suite conseillee

1. Comparer PID dynamique et PID a gains flous sur plusieurs cibles 4DDL.
2. Garder le residu RL borne et actionne autour du PID adapte.
3. Ajouter des perturbations externes pour tester si le RL apprend surtout a
   compenser les erreurs non modelisees.
4. Generaliser ensuite vers 5DDL/6DDL avec la meme decomposition, sans revenir a
   une base floue globale.

## Avancement de la suite conseillee

### 1. Benchmark multi-cibles 4DDL

Le script `experiments/benchmark_pid_fuzzy_gain_multi_target_4dof.py` compare
le PID dynamique et le PID a gains flous sur cinq cibles 4DDL.

Sorties :

- `results/tables/step_26_pid_vs_fuzzy_gain_4dof.csv`
- `results/tables/step_26_pid_vs_fuzzy_gain_4dof.md`
- `results/figures/step_26_pid_vs_fuzzy_gain_4dof.png`

Resultats synthetiques :

- PID dynamique : 5 succes sur 5, distance finale moyenne `7.0166e-03`.
- PID a gains flous : 5 succes sur 5, distance finale moyenne `3.9574e-03`.

Le PID a gains flous est parfois plus lent, mais il termine en moyenne plus pres
de la cible. C'est coherent avec son role : ameliorer l'adaptation locale des
gains sans remplacer la structure PID.

### 2. Residu RL borne autour du PID adapte

Le residu RL reste borne par construction. Les actions residuelles sont
axis-alignees :

`base, q0_res+, ..., q(n-1)_res+, q0_res-, ..., q(n-1)_res-`

Le nombre d'actions est donc `1 + 2n`, ce qui reste traitable :

- 4DDL : 9 actions.
- 5DDL : 11 actions.
- 6DDL : 13 actions.

Cette logique est maintenant factorisee dans `src/rl/residual_actions.py`.

### 3. Perturbation externe non modelisee

Le script `experiments/run_pid_residual_q_learning_4dof_disturbance.py` applique
un couple externe constant non modelise :

`(0.0, -4.0, 0.0, 0.0) N.m`

Dans ce cas, un residu exprime en acceleration n'est pas assez adapte : il peut
atteindre la cible pendant l'exploration, mais la politique gloutonne n'apprend
pas une compensation stable. Le scenario perturbation utilise donc un residu de
couple moteur borne, appris par un bandit RL episodique.

Resultats :

- action apprise : `q1_res+` ;
- PID adapte + RL residuel : succes en 204 pas, distance finale `1.5262e-03` ;
- PID adapte seul : echec a 500 pas, distance finale `1.0812e-01`.

Sorties :

- `results/tables/step_27_pid_residual_disturbance_4dof.csv`
- `results/tables/step_27_pid_residual_disturbance_4dof.md`
- `results/figures/step_27_pid_residual_disturbance_4dof.png`

Cette experience confirme que le residu RL est utile surtout quand il compense
une erreur non modelisee, ici un biais de couple moteur.

### 4. Generalisation 5DDL/6DDL

La decomposition retenue est directement extensible :

- PID vectoriel : deja compatible avec une taille quelconque.
- Adaptation floue des gains : 9 regles locales par articulation, donc `9n`.
- Actions RL residuelles : `1 + 2n` actions axis-alignees.
- Etat RL conseille : garder une representation compacte, par exemple signes
  d'erreur articulaire + niveau global de vitesse, ou bien une selection des
  articulations dominantes au lieu d'un produit cartesien complet.

Pour 6DDL, l'objectif est donc de rester sur :

- 54 evaluations floues locales pour les gains PID ;
- 13 actions residuelles bornees ;
- pas de base floue globale a `3^12 = 531441` regles.

Le passage intermediaire vers 5DDL est maintenant realise et documente dans
`docs/reorientation_pid_flou_rl_5ddl.md`. Il conserve la meme decomposition :
PID dynamique, adaptation floue locale des gains et RL residuel borne.
