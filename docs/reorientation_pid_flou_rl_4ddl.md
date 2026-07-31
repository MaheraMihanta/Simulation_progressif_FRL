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
