# Formulation RL elementaire pour le bras 2 DDL

Ce document fixe la premiere formulation d'apprentissage par renforcement
utilisee dans le projet. L'objectif n'est pas encore d'obtenir le meilleur
agent possible, mais de rendre visibles les notions de base.

## Probleme de decision de Markov

Le bras est transforme en MDP discret :

```text
MDP = (S, A, P, R, gamma)
```

- `S` : ensemble des etats.
- `A` : ensemble des actions.
- `P(s' | s, a)` : transition entre etats.
- `R(s, a, s')` : reward obtenu apres une transition.
- `gamma` : facteur d'actualisation.

## State

Dans cette premiere version, un etat est une case de la grille articulaire :

```text
s = (indice_q1, indice_q2)
```

Chaque etat correspond a un couple d'angles :

```text
q = [q1, q2]
```

La position de l'effecteur est calculee avec la cinematique directe.

## Action

Une action est un petit deplacement discret dans l'espace articulaire :

```text
a in {stay, q1-, q1+, q2-, q2+, q1-/q2-, q1-/q2+, q1+/q2-, q1+/q2+}
```

L'action ne donne pas directement la position finale de l'effecteur. Elle
modifie l'etat articulaire courant, puis le modele calcule la nouvelle position.

## Reward

Le reward penalise la distance a la cible et l'effort de commande :

```text
r = - distance(effecteur, cible) - penalite_action
```

Un bonus positif est ajoute lorsque l'etat suivant atteint la cible :

```text
r = r + bonus_cible
```

## Policy

Une politique indique quelle action choisir dans chaque etat :

```text
pi(s) = a
```

Dans cette etape, on utilise une politique deterministe obtenue par value
iteration.

## Return

Le return non actualise d'une trajectoire est :

```text
G = r0 + r1 + r2 + ... + rT
```

Il mesure la somme brute des rewards accumules.

## Discounted return

Le return actualise est :

```text
G_gamma = r0 + gamma r1 + gamma^2 r2 + ... + gamma^T rT
```

Il donne plus d'importance aux rewards proches dans le temps.

## State value

La valeur d'etat sous une politique `pi` est :

```text
V_pi(s) = E_pi[G_gamma | S0 = s]
```

Elle mesure le return actualise attendu si l'on part de l'etat `s` et que l'on
suit la politique `pi`.

## Action value

La valeur action-etat est :

```text
Q_pi(s, a) = E_pi[G_gamma | S0 = s, A0 = a]
```

Elle mesure l'interet de choisir l'action `a` dans l'etat `s`, puis de suivre
la politique.

## Equation de Bellman

Pour evaluer une politique donnee :

```text
V_pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a,s') + gamma V_pi(s')]
```

Dans notre MDP, la transition est deterministe, donc l'equation devient :

```text
V_pi(s) = R(s, pi(s), s') + gamma V_pi(s')
```

## Equation d'optimalite de Bellman

Pour chercher la meilleure politique :

```text
V*(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V*(s')]
```

Dans notre MDP deterministe :

```text
V*(s) = max_a [R(s,a,s') + gamma V*(s')]
```

La politique optimale est ensuite :

```text
pi*(s) = argmax_a [R(s,a,s') + gamma V*(s')]
```

## Q-learning tabulaire

Apres la programmation dynamique, on ajoute une version apprise par interaction.
La table `Q(s, a)` est initialisee, puis l'agent repete des episodes depuis un
etat initial. A chaque transition observee, la mise a jour est :

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

avec :

- `alpha` : taux d'apprentissage ;
- `epsilon` : probabilite d'explorer une action aleatoire ;
- `gamma` : facteur d'actualisation.

La politique apprise est ensuite la politique gloutonne :

```text
pi_Q(s) = argmax_a Q(s,a)
```

Cette etape est importante car elle remplace le calcul exact de Bellman par une
estimation construite a partir des experiences de l'agent. Dans le MDP discret
actuel, on peut comparer directement la trajectoire Q-learning avec la solution
obtenue par value iteration.

## Q-learning dynamique residuel

Pour passer vers le modele dynamique sans perdre immediatement la stabilite, on
introduit une formulation hybride :

```text
q_ddot_cmd = q_ddot_PID + q_ddot_RL_residuel
tau = M(q) q_ddot_cmd + C(q,q_dot)q_dot + G(q) + F q_dot
```

Le PID a couple calcule fournit une politique stabilisante de base. Le
Q-learning ne choisit pas encore tout le couple moteur ; il choisit une
correction discrete d'acceleration parmi :

```text
{0, +/-q1, +/-q2, combinaisons diagonales}
```

L'etat tabulaire utilise :

```text
s = (erreur_q1, erreur_q2, q1_dot, q2_dot)
```

apres discretisation. Le reward combine :

- distance a la cible ;
- vitesse articulaire ;
- effort moteur ;
- norme du residu RL ;
- progres instantane vers la cible ;
- bonus de succes.

Cette formulation est volontairement prudente. Elle valide le passage de
l'apprentissage tabulaire vers le simulateur dynamique a couples, tout en
gardant une politique de base stable. L'etape suivante consistera a reduire
progressivement l'aide du PID ou a apprendre directement des couples discrets.

## Role de cette etape

Cette etape sert de pont entre la commande classique et le RL :

- le bras fournit les etats et transitions ;
- le reward formule l'objectif ;
- Bellman permet de calculer une valeur d'etat ;
- Bellman optimality fournit une politique optimale sur l'espace discret ;
- la trajectoire obtenue peut etre comparee aux trajectoires PID et floues.
