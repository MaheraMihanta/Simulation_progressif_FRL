# Plan de travail propose - Simulation RL, logique floue et bras robotique

## 1. Objectif general

Demontrer progressivement, par simulation, le principe et l'applicabilite de l'apprentissage par renforcement pour la commande de bras robotiques.

La demarche partira d'un bras simple a 2 DDL, puis evoluera vers 3 DDL, et enfin vers un bras 6 DDL. Les premiers essais seront faits en Python, avant une validation plus realiste dans CoppeliaSim.

L'objectif scientifique est de comparer les approches classiques de commande, la logique floue, puis une approche hybride combinant apprentissage par renforcement et logique floue.

## 2. Questions de recherche

1. Comment modeliser correctement un bras robotique a 2 DDL pour tester les concepts fondamentaux de l'apprentissage par renforcement ?
2. Comment comparer une commande classique, une commande floue et une commande par apprentissage par renforcement sur les memes scenarios ?
3. Comment integrer la logique floue dans une architecture d'apprentissage par renforcement ?
4. A partir de quel niveau de complexite l'approche hybride flou/RL devient-elle plus interessante que les methodes classiques ?
5. Comment transferer progressivement les resultats de Python vers CoppeliaSim, puis vers des bras a plus grand nombre de DDL ?

## 3. Phase 0 - Cadrage et preparation

### Objectifs

- Identifier les bibliotheques Python disponibles.
- Verifier la presence eventuelle de Pyro, NumPy, SciPy, Matplotlib, Gymnasium, scikit-fuzzy ou equivalents.
- Definir les scenarios de test communs a toutes les methodes.
- Fixer les criteres de comparaison.

### Livrables

- Liste des outils disponibles.
- Specification des scenarios de simulation.
- Definition des indicateurs de performance.

### Indicateurs possibles

- Erreur finale de position.
- Temps de convergence.
- Depassement maximal.
- Energie ou effort de commande.
- Stabilite.
- Robustesse face aux perturbations.
- Capacite de generalisation vers de nouvelles cibles.

## 4. Phase 1 - Modelisation mathematique du bras 2 DDL

### Objectifs

Formuler le modele mathematique du bras robotique planaire a 2 degres de liberte.

### Elements a modeliser

- Cinematique directe.
- Cinematique inverse.
- Jacobienne.
- Dynamique du bras.
- Masses des segments.
- Longueurs des segments.
- Inerties.
- Couples articulaires.
- Eventuel modele simplifie de moteur pas a pas.
- Contraintes articulaires.

### Equations attendues

- Position de l'effecteur en fonction des angles articulaires.
- Relation entre vitesses articulaires et vitesse de l'effecteur.
- Equation dynamique sous la forme :

```text
M(q) q_ddot + C(q, q_dot) q_dot + G(q) = tau
```

avec :

- `q` : vecteur des angles articulaires.
- `q_dot` : vitesses articulaires.
- `q_ddot` : accelerations articulaires.
- `tau` : couples de commande.
- `M(q)` : matrice d'inertie.
- `C(q, q_dot)` : termes de Coriolis et centrifuges.
- `G(q)` : termes gravitationnels.

### Livrables

- Document mathematique du modele 2 DDL.
- Script Python minimal de cinematique directe et inverse.
- Visualisation simple du bras et de l'effecteur.

## 5. Phase 2 - Simulateur Python du bras 2 DDL

### Objectifs

Construire un environnement de simulation simple, clair et reutilisable.

### Fonctionnalites

- Etat du robot : angles, vitesses, position de l'effecteur.
- Actions : variations d'angles, vitesses articulaires ou couples.
- Transition d'etat selon le modele choisi.
- Gestion des limites articulaires.
- Generation de cibles.
- Affichage de la trajectoire.
- Enregistrement des resultats.

### Livrables

- Module Python du bras 2 DDL.
- Script de simulation autonome.
- Graphiques de trajectoire, erreur et commande.

## 6. Phase 3 - Commande classique

### Objectifs

Tester des methodes classiques de commande sur les memes scenarios que ceux qui seront utilises ensuite avec la logique floue et le RL.

### Methodes

- PID articulaire.
- PID dans l'espace operationnel.
- Commande robuste simple.
- Eventuellement commande par couple calcule si le modele dynamique est suffisamment etabli.

### Scenarios

- Atteinte d'un point fixe.
- Suivi d'une trajectoire simple.
- Perturbation externe ponctuelle.
- Variation de masse ou d'inertie.

### Livrables

- Controleur PID.
- Controleur robuste simple.
- Courbes de performance.
- Premiere base de comparaison.

## 7. Phase 4 - Commande par logique floue

### Objectifs

Reproduire les memes scenarios avec un controleur flou.

### Variables floues possibles

- Erreur de position.
- Variation de l'erreur.
- Distance a la cible.
- Vitesse articulaire.
- Effort de commande.

### Sorties possibles

- Correction angulaire.
- Vitesse articulaire commandee.
- Couple articulaire.
- Gain PID adapte dynamiquement.

### Approches possibles

- Controleur flou direct.
- PID flou, ou les gains `Kp`, `Ki`, `Kd` sont ajustes par logique floue.
- Controleur flou hierarchique pour separer decision globale et correction locale.

### Livrables

- Definition des ensembles flous.
- Base de regles floues.
- Controleur flou Python.
- Comparaison avec la commande classique.

## 8. Phase 5 - Apprentissage par renforcement sur bras 2 DDL

### Objectifs

Introduire les concepts fondamentaux de l'apprentissage par renforcement dans le cas du bras 2 DDL.

### Concepts a illustrer

- Etat.
- Action.
- Reward.
- Return.
- Discounted return.
- Fonction de valeur d'etat.
- Fonction de valeur action-etat.
- Politique.
- Equation de Bellman.
- Recherche de politique optimale.

### Formulation RL initiale

#### Etat

Exemples possibles :

- Angles articulaires.
- Vitesses articulaires.
- Position de l'effecteur.
- Position de la cible.
- Erreur entre effecteur et cible.

#### Action

Exemples possibles :

- Increment d'angle.
- Vitesse articulaire discrete.
- Couple discret.

#### Reward

Exemples possibles :

- Reward positif si la cible est atteinte.
- Penalisation de la distance a la cible.
- Penalisation de l'effort de commande.
- Penalisation des collisions ou limites articulaires.
- Bonus pour trajectoire courte et stable.

### Algorithmes a tester progressivement

1. Programmation dynamique sur espace discret.
2. Monte Carlo.
3. SARSA.
4. Q-learning.
5. Deep Q-Network si l'espace devient trop grand.
6. Actor-Critic ou PPO pour les actions continues.

### Livrables

- Environnement RL simple compatible avec les algorithmes choisis.
- Implementation Q-learning ou SARSA pour 2 DDL discretise.
- Visualisation de la politique apprise.
- Courbes reward/episode.
- Comparaison avec PID et logique floue.

## 9. Phase 6 - Combinaison apprentissage par renforcement et logique floue

### Objectifs

Explorer une approche hybride combinant RL et logique floue.

### Pistes d'hybridation

#### Option A - RL pour regler un controleur flou

Le systeme flou fournit la commande, tandis que le RL ajuste :

- Les fonctions d'appartenance.
- Les poids des regles.
- Les gains de sortie.
- Les seuils linguistiques.

#### Option B - Logique floue pour structurer l'espace d'etat RL

La logique floue transforme les etats continus en variables linguistiques :

- proche / moyen / loin.
- lent / rapide.
- erreur negative / nulle / positive.

Le RL apprend ensuite une politique sur cet espace flou plus interpretable.

#### Option C - Controleur flou comme politique initiale

Le controleur flou sert de politique de depart. Le RL ameliore progressivement cette politique par interaction avec l'environnement.

#### Option D - Reward shaping flou

La logique floue sert a construire une recompense plus riche, par exemple en combinant :

- Distance a la cible.
- Stabilite.
- Effort.
- Douceur du mouvement.
- Respect des limites articulaires.

### Approche recommandee pour commencer

Commencer par l'option B, car elle permet de rendre l'espace d'etat plus interpretable et de relier directement les notions de logique floue aux concepts RL.

Ensuite, tester l'option A pour obtenir une contribution plus innovante : un agent RL qui optimise automatiquement un controleur flou.

### Livrables

- Architecture hybride flou/RL.
- Comparaison avec RL pur.
- Analyse de l'interpretabilite.
- Analyse de la stabilite.
- Discussion sur l'apport scientifique.

### Implementation actuelle dans le projet

Une premiere architecture hybride est maintenant disponible :

```text
q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel
```

Le controleur flou fournit la commande de base. Le Q-learning apprend une
correction discrete d'acceleration sur une base de `81` regles floues issues de
`(erreur_q1, erreur_q2, q1_dot, q2_dot)`. Cette implementation correspond aux
options B et C :

- structuration floue de l'espace d'etat RL ;
- controleur flou utilise comme politique initiale stabilisante.

Le detail est documente dans `docs/fuzzy_rl.md` et l'experience correspondante
est `experiments/run_fuzzy_residual_q_learning_2dof.py`.

## 10. Phase 7 - Comparaison globale

### Objectifs

Comparer toutes les methodes dans un cadre commun.

### Methodes a comparer

- PID.
- Commande robuste simple.
- Controleur flou.
- RL pur.
- Controleur hybride flou/RL.

### Scenarios de comparaison

- Atteinte d'une cible fixe.
- Atteinte de plusieurs cibles aleatoires.
- Suivi de trajectoire.
- Perturbations externes.
- Changement des parametres physiques.
- Bruit de mesure.

### Resultats attendus

- Tableaux comparatifs.
- Graphiques de trajectoire.
- Courbes d'erreur.
- Courbes d'effort de commande.
- Courbes de reward.
- Analyse qualitative et quantitative.

## 11. Phase 8 - Extension vers 3 DDL puis 6 DDL

### Objectifs

Augmenter progressivement la complexite du robot.

### Progression

1. Bras planaire 2 DDL.
2. Bras planaire ou spatial 3 DDL.
3. Bras 4 a 5 DDL si necessaire.
4. Bras 6 DDL.

### Points a surveiller

- Explosion de la taille de l'espace d'etat.
- Difficultes de discretisation.
- Complexite de la cinematique inverse.
- Besoin d'algorithmes RL continus.
- Stabilisabilite de la commande.
- Temps d'apprentissage.

### Livrables

- Generalisation du simulateur.
- Adaptation des controleurs.
- Resultats comparatifs 2 DDL / 3 DDL / 6 DDL.

## 12. Phase 9 - Passage vers CoppeliaSim

### Objectifs

Valider les approches dans un environnement de simulation robotique plus realiste.

### Etapes

1. Reproduire le bras 2 DDL dans CoppeliaSim.
2. Connecter Python a CoppeliaSim.
3. Rejouer les scenarios simples.
4. Integrer les controleurs PID, flou, RL et flou/RL.
5. Comparer les resultats Python pur et CoppeliaSim.
6. Etendre vers un bras 6 DDL.

### Livrables

- Scene CoppeliaSim 2 DDL.
- Interface Python/CoppeliaSim.
- Validation experimentale des controleurs.
- Resultats comparatifs.

## 13. Organisation conseillee du projet

```text
Simuation_FRL/
  docs/
    modele_2ddl.md
    formulation_rl.md
    comparaison_methodes.md
  src/
    robot/
      arm_2dof.py
      kinematics.py
      dynamics.py
    controllers/
      pid.py
      robust.py
      fuzzy.py
      rl.py
      fuzzy_rl.py
    envs/
      arm_2dof_env.py
    experiments/
      run_pid.py
      run_fuzzy.py
      run_rl.py
      run_fuzzy_rl.py
    visualization/
      plots.py
  results/
    figures/
    logs/
    tables/
  coppeliasim/
    scenes/
    scripts/
```

## 14. Plan d'execution court terme

### Etape 1

Creer le modele cinematique 2 DDL et visualiser le bras dans Python.

### Etape 2

Ajouter une cible et mesurer l'erreur entre l'effecteur et la cible.

### Etape 3

Implementer un PID simple pour atteindre une cible fixe.

### Etape 4

Implementer un controleur flou sur le meme scenario.

### Etape 5

Creer un environnement RL discretise et tester Q-learning.

### Etape 6

Comparer PID, logique floue et Q-learning sur les memes cibles.

### Etape 7

Introduire une premiere architecture hybride flou/RL.

## 15. Contribution scientifique visee

La contribution principale peut etre formulee ainsi :

> Proposition et evaluation progressive d'une architecture hybride combinant logique floue et apprentissage par renforcement pour la commande de bras robotiques, avec validation comparative face aux commandes classiques sur des bras de complexite croissante.

Cette contribution est interessante car elle combine :

- L'interpretabilite de la logique floue.
- La capacite d'adaptation du RL.
- Une comparaison claire avec des methodes classiques.
- Une validation progressive du 2 DDL vers le 6 DDL.

## 16. Prochaine action recommandee

Commencer par le simulateur Python du bras 2 DDL, car il permettra de tester rapidement tous les concepts fondamentaux : etat, action, reward, politique, valeur d'etat, controle classique, logique floue et apprentissage.
