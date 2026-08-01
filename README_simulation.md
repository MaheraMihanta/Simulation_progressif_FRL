# Base de simulation Python-CoppeliaSim

Cette base suit le protocole du brouillon `article_fuzzy_guided_drl_6dof.pdf`:
trajectoires de reference, superviseur flou, controleur de comparaison, journalisation
CSV, resume JSON et figures pour les metriques.

Avant de poursuivre la redaction finale, lire aussi `SIMULATION_VALIDATION.md`.
Ce fichier explique pourquoi les anciens resultats vibraient et donne le protocole
de validation PID, fuzzy-PID et FRL/DRL.

## Verification sans CoppeliaSim

```powershell
python run_nominal_tracking.py --dry-run --controller fuzzy-pid --duration 3
```

Cette commande utilise un petit modele articulaire offline. Elle sert a verifier le
pipeline de donnees avant de lancer la scene.

## Lancement avec la scene CoppeliaSim

1. Ouvrir `bras_manipulateur_niryoOne.ttt` dans CoppeliaSim.
2. Verifier que l'add-on ZeroMQ Remote API est actif.
3. Laisser la simulation arretee dans CoppeliaSim.
4. Lancer:

```powershell
python run_nominal_tracking.py --controller fuzzy-pid --duration 12 --dt 0.05
```

Les resultats sont crees dans `results/<date>_coppelia_fuzzy-pid/`.

## Experiences utiles pour l'article

```powershell
python run_nominal_tracking.py --controller reference --duration 12
python run_nominal_tracking.py --controller pid --duration 12
python run_nominal_tracking.py --controller fuzzy-pid --duration 12
python run_nominal_tracking.py --controller pid --scenario sensor_noise --duration 12
python run_nominal_tracking.py --controller fuzzy-pid --scenario combined_uncertainty --duration 12
python run_nominal_tracking.py --controller fuzzy-pid --trajectory point_to_point --duration 10
```

Scenarios disponibles:

- `nominal`: trajectoire nominale sans perturbation ajoutee.
- `sensor_noise`: bruit gaussien sur la position et la vitesse observees par le controleur.
- `observation_delay`: retard discret sur l'etat observe par le controleur.
- `trajectory_step`: changement lisse de trajectoire pendant l'essai.
- `combined_uncertainty`: bruit, retard et changement de trajectoire combines.

Pour lancer une campagne comparative:

```powershell
python run_validation_campaign.py --duration 12 --dt 0.05
python run_validation_campaign.py --dry-run --duration 12 --dt 0.05
python run_validation_campaign.py --scenarios combined_uncertainty --duration 12 --dt 0.05
```

Les metriques actuellement produites sont: RMSE articulaire, erreur maximale,
erreur finale, energie de correction, lissage de l'action, violations de contraintes,
temps d'etablissement, moyenne/ecart-type de la norme d'erreur, ratio de changement
de signe des corrections et indice haute frequence d'erreur. Le superviseur flou
journalise aussi les poids de reward prevus pour la future phase DRL.

Les journaux CSV contiennent maintenant:

- `q1..q6`: etat articulaire reel mesure dans le backend.
- `q_obs1..q_obs6`: etat vu par le controleur apres bruit/retard eventuel.
- `q_ref1..q_ref6`: reference articulaire.
- `target1..target6`: consigne envoyee au backend.

Pour diagnostiquer les vibrations dans un ou plusieurs runs:

```powershell
python diagnose_tracking_results.py results\<run_1> results\<run_2>
```

## Point d'entree DRL

`fuzzy_drl_sim.rl_task.FuzzyGuidedTrackingTask` fournit deja une interface de type
Gym: `reset()`, `step(action)`, observation vectorielle, reward floue et action
continue normalisee dans `[-1, 1]`. Elle peut etre enveloppee ensuite par SAC ou TD3
sans modifier la scene CoppeliaSim.

Avant d'installer un framework RL complet, on peut verifier l'interface FRL/DRL avec
des politiques simples:

```powershell
python evaluate_frl_task.py --dry-run --policy zero --duration 6
python evaluate_frl_task.py --dry-run --policy proportional --duration 6
python evaluate_frl_task.py --dry-run --policy fuzzy_expert --duration 6
python evaluate_frl_task.py --policy fuzzy_expert --duration 6
```

Ces commandes ne sont pas un entrainement DRL. Elles valident seulement l'espace
d'action, la reward floue et le comportement ferme de l'environnement.
