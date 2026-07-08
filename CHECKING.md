# Journal de checking

Ce fichier garde la trace des controles effectues a chaque etape implementee.

## Etape 0 - Environnement Python

Statut : OK

Controle effectue :

```text
python --version
```

Resultat :

```text
Python 3.12.4
```

Bibliotheques disponibles :

- numpy : OK
- matplotlib : OK
- scipy : OK
- torch : OK
- pyro : OK

Bibliotheques absentes :

- skfuzzy
- gymnasium
- pyro4

Decision :

- Demarrer avec `numpy` et `matplotlib`.
- Eviter une dependance obligatoire a `skfuzzy` ou `gymnasium` pour les premieres etapes.

## Etape 1 - Cinematique 2 DDL

Statut : OK

Controles effectues :

```text
python -m unittest discover -s tests
python experiments/run_kinematics_2dof.py
```

Resultat des tests :

```text
Ran 6 tests in 2.600s
OK
```

Resultat de l'experience :

```text
joint_angles_rad=[-0.241865  1.650568]
target=[1.1  0.55]
end_effector=[1.1  0.55]
error=0.000000000000e+00
figure=E:\THESE\RL\Simuation_FRL\results\figures\step_01_kinematics_2dof.png
```

Fichiers principaux ajoutes :

- `docs/modele_2ddl.md`
- `src/robot/kinematics.py`
- `src/robot/arm_2dof.py`
- `src/visualization/plots.py`
- `experiments/run_kinematics_2dof.py`
- `tests/test_kinematics.py`

Decision :

- La base cinematique 2 DDL est validee.
- L'erreur nulle est normale ici : la cinematique inverse analytique est
  revalidee avec la meme cinematique directe. C'est un controle de coherence
  numerique, pas une preuve de performance realiste d'un controleur.
- La prochaine etape peut ajouter une cible dynamique, une erreur de suivi et un premier controleur PID.

## Etape 2 - Environnement 2 DDL avec cible et reward

Statut : OK

Controles effectues :

```text
python -m unittest discover -s tests
```

Resultat :

```text
Ran 10 tests in 0.240s
OK
```

Fichiers principaux ajoutes :

- `src/envs/arm_2dof_env.py`
- `src/envs/__init__.py`
- `tests/test_env_pid.py`

Comportement valide :

- L'environnement expose `q`, `q_dot`, `end_effector`, `target`, `error`, `distance`.
- Une action est interpretee comme vitesse articulaire commandee.
- L'action est bornee par `max_joint_speed`.
- Le reward penalise la distance a la cible et l'effort de commande.
- Une cible hors espace atteignable est rejetee.

## Etape 3 - Controle PID articulaire

Statut : OK

Controles effectues :

```text
python -m unittest discover -s tests
python experiments/run_pid_2dof.py
```

Resultat de l'experience PID :

```text
steps=32
done=True
truncated=False
desired_joint_angles_rad=[-0.241865  1.650568]
final_joint_angles_rad=[-0.241247  1.639437]
final_distance=8.478859065988e-03
figure=E:\THESE\RL\Simuation_FRL\results\figures\step_02_pid_2dof.png
```

Fichiers principaux ajoutes :

- `src/controllers/pid.py`
- `src/controllers/__init__.py`
- `experiments/run_pid_2dof.py`

Decision :

- Le PID est valide comme premiere commande classique de reference.
- La figure PID montre maintenant une simulation plus lisible : trajectoire de
  l'effecteur, poses successives du bras, erreur de position et commande.
- Les courbes d'erreur et de commande sont les indicateurs quantitatifs les plus
  importants ; la visualisation du bras sert a verifier et expliquer le mouvement.
- La prochaine etape logique est d'ajouter un controleur flou simple sur le meme environnement et la meme cible.

## Etape 4 - Controle flou articulaire simple

Statut : OK

Controles effectues :

```text
python -m unittest discover -s tests
python experiments/run_fuzzy_2dof.py
```

Resultat des tests :

```text
Ran 12 tests in 0.170s
OK
```

Resultat de l'experience floue :

```text
steps=58
done=True
truncated=False
desired_joint_angles_rad=[-0.241865  1.650568]
final_joint_angles_rad=[-0.241842  1.638886]
final_distance=9.329197091651e-03
figure=E:\THESE\RL\Simuation_FRL\results\figures\step_03_fuzzy_2dof.png
```

Fichiers principaux ajoutes ou modifies :

- `src/controllers/fuzzy.py`
- `experiments/run_fuzzy_2dof.py`
- `tests/test_env_pid.py`
- `src/visualization/plots.py`
- `experiments/run_pid_2dof.py`

Decision :

- Le controleur flou converge vers la meme cible que le PID dans le meme
  environnement.
- La convergence floue est plus lente sur ce reglage initial : 58 iterations
  contre 32 pour le PID.
- Pour la these, il faut presenter les courbes et les metriques comme preuves
  principales. Les captures/animations de simulation sont utiles en appui
  visuel, mais elles ne remplacent pas l'erreur, le temps de convergence et
  l'effort de commande.
- La prochaine etape logique est de creer une comparaison PID/flou sur plusieurs
  cibles, puis d'ajouter le premier Q-learning discretise.

## Etape 5 - RL elementaire avec Bellman sur MDP discret

Statut : OK

Objectif :

- Mettre en evidence les notions fondamentales de l'apprentissage par
  renforcement avant de passer a Q-learning :
  `state`, `action`, `policy`, `reward`, `return`, `discounted return`,
  `state value`, `action value`, equation de Bellman et equation
  d'optimalite de Bellman.

Controles effectues :

```text
python -m unittest discover -s tests
python experiments/run_rl_bellman_2dof.py
```

Resultat des tests :

```text
Ran 17 tests in 1.361s
OK
```

Resultat de l'experience RL/Bellman :

```text
state_count=961
action_count=9
gamma=0.950
start_state=480
random_policy_value_start=-2.234846793579e+01
optimal_value_start=4.416102132407e+00
value_iteration_iterations=37
rollout_steps=8
done=True
return=7.120809279466e+00
discounted_return=4.416102132407e+00
actions=['q1+/q2+', 'q2+', 'q2+', 'q1-/q2+', 'q2+', 'q2+', 'q1-/q2+', 'q2+']
final_distance=5.369575198995e-02
figure=E:\THESE\RL\Simuation_FRL\results\figures\step_04_rl_bellman_2dof.png
```

Fichiers principaux ajoutes :

- `docs/formulation_rl.md`
- `src/rl/discrete_arm_mdp.py`
- `src/rl/dynamic_programming.py`
- `src/rl/__init__.py`
- `experiments/run_rl_bellman_2dof.py`
- `tests/test_rl_bellman.py`

Comportement valide :

- L'etat discret est une case articulaire `(indice_q1, indice_q2)`.
- L'action discrete est un petit deplacement articulaire.
- La politique aleatoire est evaluee par l'equation de Bellman.
- La politique optimale est obtenue par value iteration, donc par l'equation
  d'optimalite de Bellman.
- Le rollout optimal atteint la cible discrete en 8 actions.
- La valeur optimale de l'etat initial correspond au discounted return du
  rollout optimal, ce qui valide la coherence entre `V*`, politique et retour.

Decision :

- Cette etape montre les briques RL de base de maniere explicite et
  interpretable.
- La prochaine etape naturelle est Q-learning : remplacer le calcul exact par
  une estimation apprise par interaction avec l'environnement.
- Ensuite, comparer PID, flou, value iteration et Q-learning sur les memes
  cibles.
