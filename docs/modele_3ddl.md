# Modele spatial du bras 3 DDL

Ce document decrit l'extension du bras planaire `2 DDL` vers un bras spatial
`3 DDL`. La configuration retenue suit la demande du projet :

- une base rotative autour de l'axe vertical `z` ;
- un bras `2 DDL` superpose sur cette base ;
- les deux articulations du bras travaillent dans le plan radial-vertical choisi
  par l'angle de base.

Le vecteur articulaire devient :

```text
q = [q0, q1, q2]^T
```

avec :

- `q0` : rotation de base, ou lacet ;
- `q1` : articulation d'epaule dans le plan vertical ;
- `q2` : articulation de coude.

## Cinematique directe

On commence par calculer la cinematique du bras `2R` dans le plan radial-z :

```text
rho = l1 cos(q1) + l2 cos(q1 + q2)
z   = l1 sin(q1) + l2 sin(q1 + q2)
```

La base projette ensuite ce point dans l'espace :

```text
x = rho cos(q0)
y = rho sin(q0)
z = z
```

La position du coude est :

```text
rho1 = l1 cos(q1)
z1   = l1 sin(q1)

x1 = rho1 cos(q0)
y1 = rho1 sin(q0)
z1 = z1
```

## Cinematique inverse

Pour une cible spatiale `(x, y, z)`, la base choisit d'abord l'azimut :

```text
q0 = atan2(y, x)
rho = sqrt(x^2 + y^2)
```

Le probleme restant est une cinematique inverse planaire dans le plan
`(rho, z)` :

```text
r2 = rho^2 + z^2
c2 = (r2 - l1^2 - l2^2) / (2 l1 l2)
q2 = atan2(s2, c2)
q1 = atan2(z, rho) - atan2(l2 s2, l1 + l2 c2)
```

Le signe de `s2` donne les deux configurations coude haut / coude bas.

## Domaine atteignable

Comme la base peut pivoter librement, l'espace atteignable est une coque
spherique :

```text
abs(l1 - l2) <= sqrt(x^2 + y^2 + z^2) <= l1 + l2
```

Cette formulation garde la meme intuition que le bras `2 DDL`, mais remplace le
disque/anneau planaire par un volume spatial obtenu par revolution.

## Jacobienne

Avec :

```text
rho = l1 cos(q1) + l2 cos(q1 + q2)
drho/dq1 = -l1 sin(q1) - l2 sin(q1 + q2)
drho/dq2 = -l2 sin(q1 + q2)
dz/dq1 = rho
dz/dq2 = l2 cos(q1 + q2)
```

la Jacobienne geometrique est :

```text
J(q) =
[ -rho sin(q0)   drho/dq1 cos(q0)   drho/dq2 cos(q0) ]
[  rho cos(q0)   drho/dq1 sin(q0)   drho/dq2 sin(q0) ]
[       0              dz/dq1             dz/dq2     ]
```

Elle relie les vitesses articulaires a la vitesse cartesienne de l'effecteur :

```text
p_dot = J(q) q_dot
```

## Dynamique simplifiee

Le modele dynamique garde l'equation utilisee dans le projet :

```text
M(q) q_ddot + C(q, q_dot) q_dot + G(q) + F q_dot = tau
```

Le sous-systeme `q1, q2` reprend la dynamique du bras planaire vertical. La base
`q0` ajoute une inertie de lacet qui depend de la posture :

```text
I_yaw(q1, q2) = I_base + m1 rho_c1^2 + m2 rho_c2^2
```

avec :

```text
rho_c1 = r1 cos(q1)
rho_c2 = l1 cos(q1) + r2 cos(q1 + q2)
```

La matrice d'inertie prend la forme :

```text
M_3DDL(q) =
[ I_yaw(q)    0          0     ]
[    0       M11        M12    ]
[    0       M12        M22    ]
```

Les termes de gravite n'agissent pas directement sur le lacet de base :

```text
G_3DDL(q) = [0, G1(q1,q2), G2(q1,q2)]^T
```

Le modele inclut aussi :

- le frottement visqueux sur les trois articulations ;
- le terme `d(I_yaw)/dt q0_dot` sur la base ;
- les effets centrifuges de `q0_dot` sur les articulations du bras via le
  gradient de `I_yaw`.

Cette dynamique est volontairement intermediaire : elle est plus riche qu'une
simple copie du modele `2 DDL`, mais reste assez compacte pour comparer PID,
flou et flou/RL avant le passage vers CoppeliaSim.

## Environnements et experiences

Les nouveaux modules ajoutent les memes niveaux que le pipeline `2 DDL` :

- `src/robot/kinematics_3dof.py` ;
- `src/robot/dynamics_3dof.py` ;
- `src/envs/arm_3dof_env.py` ;
- `src/envs/arm_3dof_dynamic_env.py` ;
- `src/rl/fuzzy_residual_q_learning_3dof.py`.

Les experiences produisent les livrables suivants :

- `step_16_kinematics_3dof.png` ;
- `step_17_pid_3dof.png` ;
- `step_18_fuzzy_3dof.png` ;
- `step_19_pid_dynamic_3dof.png` ;
- `step_20_fuzzy_dynamic_3dof.png` ;
- `step_21_fuzzy_residual_q_learning_3dof.png` ;
- `step_22_fuzzy_residual_generalization_3dof.csv` ;
- `step_22_fuzzy_residual_generalization_3dof.md` ;
- `step_22_fuzzy_residual_generalization_3dof.png`.

## Extension flou/RL

L'etat flou 3DDL est :

```text
x = (erreur_q0, erreur_q1, erreur_q2, q0_dot, q1_dot, q2_dot)
```

Chaque variable utilise trois termes :

```text
negative, zero, positive
```

Le nombre de regles devient donc :

```text
3^6 = 729 regles floues
```

L'action residuelle est une acceleration articulaire discrete dans les trois
axes :

```text
q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel
```

Les directions possibles sont les combinaisons `-1, 0, +1` sur les trois
articulations, avec l'action nulle placee en premier. On obtient :

```text
3^3 = 27 actions residuelles
```

Cette extension illustre l'effet direct de la montee en dimension : la logique
floue garde l'espace d'etat interpretable, mais la base de regles et l'espace
d'action augmentent fortement.
