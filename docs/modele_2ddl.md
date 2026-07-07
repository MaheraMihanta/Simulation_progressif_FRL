# Modele cinematique du bras 2 DDL

Ce document pose la premiere version du modele 2 DDL utilisee dans les simulations Python.

## Hypotheses initiales

- Bras planaire a deux articulations rotatives.
- Base fixe en `(0, 0)`.
- Longueurs des segments : `l1` et `l2`.
- Angles articulaires : `q1` et `q2`.
- Pas encore de dynamique, de masse ni de couple moteur dans cette premiere etape.

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

## Prochaine extension

La prochaine etape ajoutera un environnement de simulation avec cible, erreur, action discrete ou continue, puis les premiers controleurs.

