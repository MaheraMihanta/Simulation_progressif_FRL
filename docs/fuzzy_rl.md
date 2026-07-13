# Hybridation logique floue et apprentissage par renforcement

Ce document precise le sens de la combinaison flou/RL ajoutee apres le
Q-learning dynamique residuel autour du PID.

## Idee principale

La logique floue et le RL ne jouent pas le meme role :

- la logique floue apporte une commande de base stable et interpretable ;
- les ensembles flous transforment l'etat continu en regles linguistiques ;
- le Q-learning apprend une correction discrete a appliquer dans ces regles.

On ne demande donc pas au RL d'apprendre tout le couple moteur depuis zero. Il
apprend un residu autour d'une politique floue deja capable d'atteindre la
cible.

## Architecture utilisee

La commande finale est :

```text
q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel
tau = M(q) q_ddot_cmd + C(q,q_dot)q_dot + G(q) + F q_dot
```

Le controleur flou calcule `q_ddot_flou` a partir de l'erreur articulaire et de
sa variation. Le modele dynamique inverse transforme ensuite l'acceleration
commandee en couple moteur.

## Etat flou pour le RL

L'etat dynamique continu est :

```text
x = (erreur_q1, erreur_q2, q1_dot, q2_dot)
```

Chaque variable est normalisee, puis projetee sur trois termes :

```text
negative, zero, positive
```

Le nombre de regles floues est donc :

```text
3^4 = 81 regles
```

Chaque observation active plusieurs regles avec des poids. La valeur d'une
action n'est plus lue dans une seule case discrete ; elle est agregee :

```text
Q_flou(x, a) = somme_i w_i(x) Q(regle_i, a)
```

ou `w_i(x)` est le degre d'activation de la regle `i`.

## Mise a jour Q-learning

L'action est choisie par epsilon-greedy sur les valeurs agregees :

```text
a = argmax_a Q_flou(x, a)
```

Apres la transition, l'erreur temporelle est :

```text
delta = r + gamma max_a' Q_flou(x', a') - Q_flou(x, a)
```

Les regles actives sont mises a jour proportionnellement a leur activation :

```text
Q(regle_i, a) <- Q(regle_i, a) + alpha w_i(x) delta
```

Cette formule donne une interpretation directe : une correction apprise dans un
etat influence aussi les etats voisins qui activent les memes regles floues.

## Reward

Le reward conserve les memes criteres que le Q-learning dynamique residuel :

- distance a la cible ;
- vitesse articulaire ;
- effort moteur ;
- norme du residu RL ;
- progres instantane vers la cible ;
- bonus de succes.

La penalisation du residu est importante : elle force le RL a utiliser la
correction uniquement quand elle apporte quelque chose par rapport au controleur
flou seul.

## Resultat de l'experience actuelle

Sur la cible `(1.1, 0.55)`, l'experience `step_10` donne :

```text
flou seul      : 360 pas, distance finale 5.23e-03, couple moyen 13.85 N.m
flou + Q flou  : 269 pas, distance finale 5.54e-03, couple moyen 14.42 N.m
```

Le resultat ne montre pas encore une superiorite globale. Il montre plutot un
compromis :

- l'hybride atteint la cible plus vite ;
- la precision finale reste comparable ;
- l'effort moyen augmente legerement.

Ce compromis donne un sens scientifique a l'hybridation : le RL exploite les
regles floues pour accelerer la convergence, mais il faut encore regler le
reward si l'objectif prioritaire devient la reduction d'energie ou la douceur du
mouvement.

## Interpretation

Cette etape correspond a deux pistes du plan initial :

- Option B : la logique floue structure l'espace d'etat RL ;
- Option C : le controleur flou sert de politique de depart que le RL ameliore.

La prochaine comparaison doit donc mesurer l'apport de cette architecture sur
plusieurs cibles et pas seulement sur une cible unique.
