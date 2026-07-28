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

## Generalisation sur plusieurs cibles

L'experience `step_11` entraine la table Q floue sur la cible de reference
`(1.10, 0.55)`, puis reutilise la meme table sur cinq cibles. Comme l'etat est
exprime en erreur articulaire relative a la cible, cette experience teste si les
regles apprises sont reutilisables.

Resume :

```text
flou seul     : succes 5/5
flou + Q flou : succes 4/5
delta pas moyen flou/RL - flou seul = -41.2 pas
delta couple moyen = +0.36 N.m
```

Sur quatre cibles, le residu appris accelere la convergence. Sur la cible
`T4_high`, il degrade le comportement et n'atteint pas la tolerance. Cette
degradation est importante : elle montre que la politique apprise sur une cible
ne doit pas encore etre consideree comme globalement robuste.

La conclusion actuelle est donc :

- la representation floue donne une certaine capacite de generalisation ;
- le residu RL peut accelerer le mouvement ;
- l'agent doit encore apprendre avec plusieurs cibles, ou etre protege par un
  mecanisme de selection qui revient au controleur flou si le residu est
  incertain.

Pour le rapport final, cette experience justifie la prochaine etape : entrainer
la politique hybride sur une distribution de cibles, et non plus sur une cible
unique.

## Supervision de securite du residu RL

L'experience de generalisation brute montre un risque important : un residu RL
utile sur certaines cibles peut degrader le controleur flou sur une autre cible.
Pour limiter ce risque, une supervision simple a ete ajoutee au rollout.

Principe :

```text
si la distance ne s'ameliore plus pendant N pas :
    couper q_ddot_RL_residuel
    revenir a q_ddot_cmd = q_ddot_flou
```

Dans l'experience actuelle :

```text
N = 100 pas
progres minimal = 1e-4
```

La comparaison `step_12` donne :

```text
flou seul          : succes 5/5
flou + Q brut      : succes 4/5
flou + Q securise  : succes 5/5
delta pas moyen du flou + Q securise = -55.0 pas
delta couple moyen = +0.46 N.m
```

Le superviseur restaure la convergence sur `T4_high`, la cible qui echouait
avec le residu brut. Il coupe aussi le residu sur `T2_diag`, ou la politique
securisee devient proche du flou seul. Cette observation est utile : le
superviseur augmente la robustesse, mais il peut reduire certains gains de
vitesse lorsque la coupure est trop conservatrice.

Cette etape donne une architecture plus defendable :

- le controleur flou reste la politique de secours ;
- le RL accelere la trajectoire lorsque son residu reste utile ;
- la supervision limite les degradations hors cible d'entrainement.

La prochaine amelioration consistera a remplacer cette coupure heuristique par
une decision apprise ou par une estimation explicite de confiance du residu.

## Deploiement sequentiel sur plusieurs cibles

L'etape `step_15` transforme l'experience en artefact deployable. La table Q
floue est entrainee une fois, sauvegardee dans
`results/policies/step_15_fuzzy_residual_policy_2dof.npz`, rechargee, puis
utilisee dans la boucle live sur une sequence de cinq cibles. Contrairement aux
rollouts precedents, le robot n'est pas remis a zero entre deux cibles : chaque
cible commence depuis l'etat atteint a la cible precedente.

Resultat synthetique :

```text
flou seul deploye          : succes 5/5, 1366 pas au total
flou + Q securise deploye  : succes 5/5, 1309 pas au total
gain total                 : -57 pas
couple moyen global        : quasi identique au flou seul
```

La supervision coupe le residu sur quatre cibles de la sequence. Ce resultat
montre que l'architecture est deployable de maniere prudente : le residu RL
apporte un gain net sur la mission complete, mais le controleur flou reste le
garant principal de la convergence hors cible d'entrainement.

Les livrables de deploiement sont :

- `results/policies/step_15_fuzzy_residual_policy_2dof.npz` ;
- `results/deployments/step_15_multi_target_deployment_2dof.csv` ;
- `results/deployments/step_15_multi_target_deployment_2dof.md` ;
- `results/deployments/step_15_multi_target_deployment_2dof.json` ;
- `results/figures/step_15_multi_target_deployment_2dof.png`.

## Extension vers le bras spatial 3 DDL

L'extension `3 DDL` conserve la meme architecture, mais ajoute une rotation de
base autour de l'axe vertical. Le bras `2 DDL` initial devient alors un bras
spatial obtenu par revolution du plan de travail.

La commande reste :

```text
q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel
tau = M_3DDL(q) q_ddot_cmd + C_3DDL(q,q_dot)q_dot + G_3DDL(q) + F q_dot
```

L'etat flou devient :

```text
x = (erreur_q0, erreur_q1, erreur_q2, q0_dot, q1_dot, q2_dot)
```

Chaque variable garde les trois termes linguistiques `negative`, `zero`,
`positive`. Le nombre de regles passe donc de :

```text
3^4 = 81 regles pour le 2 DDL
```

a :

```text
3^6 = 729 regles pour le 3 DDL
```

Le residu RL agit maintenant sur trois accelerations articulaires. Les actions
sont les combinaisons discretes `-1, 0, +1` sur les trois axes, avec l'action
nulle en premiere position :

```text
3^3 = 27 actions residuelles
```

Les experiences `step_16` a `step_22` reproduisent le mode operatoire du
`2 DDL` : cinematique inverse, PID, flou, dynamique, flou/RL sur cible de
reference, puis synthese multi-cibles. Cette phase sert surtout a mesurer
l'effet de la montee en dimension : la structure floue reste lisible, mais le
nombre de regles et d'actions augmente fortement, ce qui justifie ensuite un
entrainement multi-cibles ou un passage vers des methodes RL continues.
