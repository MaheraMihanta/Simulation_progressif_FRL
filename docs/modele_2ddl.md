# Modele cinematique du bras 2 DDL

Ce document pose la premiere version du modele 2 DDL utilisee dans les simulations Python.

## Hypotheses initiales

- Bras planaire a deux articulations rotatives.
- Base fixe en `(0, 0)`.
- Longueurs des segments : `l1` et `l2`.
- Angles articulaires : `q1` et `q2`.
- La premiere etape etait cinematique. L'extension dynamique ajoute les masses,
  inerties, frottements, gravite et couples moteurs.

## Cinematique directe

La position de l'effecteur est :

```text
x = l1 cos(q1) + l2 cos(q1 + q2)
y = l1 sin(q1) + l2 sin(q1 + q2)
```

La position du coude est :

```text
x1 = l1 cos(q1)
y1 = l1 sin(q1)
```

## Jacobienne

La relation entre vitesses articulaires et vitesse de l'effecteur est :

```text
[x_dot]   [ -l1 sin(q1) - l2 sin(q1 + q2)    -l2 sin(q1 + q2) ] [q1_dot]
[y_dot] = [  l1 cos(q1) + l2 cos(q1 + q2)     l2 cos(q1 + q2) ] [q2_dot]
```

## Cinematique inverse

Pour une cible `(x, y)`, on calcule :

```text
r2 = x^2 + y^2
c2 = (r2 - l1^2 - l2^2) / (2 l1 l2)
q2 = atan2(s2, c2)
q1 = atan2(y, x) - atan2(l2 s2, l1 + l2 c2)
```

avec :

```text
s2 = sqrt(1 - c2^2)
```

Le signe de `s2` permet d'obtenir deux configurations possibles : coude haut ou coude bas.

## Domaine atteignable

Une cible est atteignable si sa distance `r` a la base respecte :

```text
abs(l1 - l2) <= r <= l1 + l2
```

## Dynamique articulaire

Le modele dynamique utilise dans la simulation a couples est :

```text
M(q) q_ddot + C(q, q_dot) q_dot + G(q) + F q_dot = tau
```

avec :

- `q = [q1, q2]` : angles articulaires ;
- `q_dot` : vitesses articulaires ;
- `q_ddot` : accelerations articulaires ;
- `tau = [tau1, tau2]` : couples moteurs ;
- `M(q)` : matrice d'inertie ;
- `C(q, q_dot) q_dot` : termes Coriolis et centrifuges ;
- `G(q)` : couples de gravite ;
- `F q_dot` : frottement visqueux.

Les centres de masse sont places a :

```text
r1 = ratio1 l1
r2 = ratio2 l2
```

Par defaut, `ratio1 = ratio2 = 0.5`. Si les inerties ne sont pas fournies, on
utilise l'approximation d'une tige uniforme :

```text
I1 = m1 l1^2 / 12
I2 = m2 l2^2 / 12
```

La matrice d'inertie est :

```text
M11 = I1 + I2 + m1 r1^2 + m2 (l1^2 + r2^2 + 2 l1 r2 cos(q2))
M12 = I2 + m2 (r2^2 + l1 r2 cos(q2))
M22 = I2 + m2 r2^2

M(q) = [ M11  M12 ]
       [ M12  M22 ]
```

Les termes Coriolis et centrifuges sont regroupes sous forme vectorielle :

```text
h = m2 l1 r2 sin(q2)

C(q, q_dot) q_dot =
[ -h (2 q1_dot q2_dot + q2_dot^2) ]
[  h q1_dot^2                         ]
```

Les couples de gravite sont :

```text
G1 = (m1 r1 + m2 l1) g cos(q1) + m2 r2 g cos(q1 + q2)
G2 = m2 r2 g cos(q1 + q2)
```

Le frottement visqueux est :

```text
F q_dot = [ f1 q1_dot, f2 q2_dot ]
```

## Simulation dynamique

L'environnement dynamique interprete l'action comme un couple moteur. A chaque
pas :

```text
q_ddot = M(q)^-1 (tau - C(q, q_dot)q_dot - G(q) - Fq_dot)
q_dot <- q_dot + q_ddot dt
q     <- q + q_dot dt
```

L'integration est semi-implicite : la nouvelle vitesse est utilisee pour mettre
a jour la position. Les couples, vitesses et angles sont bornes par les limites
du modele.

## Commande dynamique PID et floue

Les experiences dynamiques utilisent une commande a couple calcule :

```text
tau = M(q) q_ddot_cmd + C(q, q_dot)q_dot + G(q) + Fq_dot
```

Pour le PID, `q_ddot_cmd` est donne par un PID articulaire sur l'erreur
`q_desire - q`.

Pour le controleur flou, la meme base de regles que la commande floue initiale
est interpretee comme une commande d'acceleration articulaire. Le modele
dynamique transforme ensuite cette acceleration desiree en couple moteur.

## Prochaine extension

La prochaine etape peut maintenant utiliser cette dynamique dans la formulation
RL : l'action pourra etre un couple discret ou continu, et l'etat devra inclure
les vitesses articulaires.
