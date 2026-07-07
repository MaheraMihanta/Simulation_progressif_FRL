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
- La prochaine etape logique est d'ajouter un controleur flou simple sur le meme environnement et la meme cible.
