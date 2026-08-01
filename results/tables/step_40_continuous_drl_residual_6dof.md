# SAC/TD3 residuel continu 6DDL

Cette etape transforme `FuzzyGuidedTrackingTask` en tache Gymnasium utilisable
par SAC/TD3. Le mode par defaut est maintenant `residual` : la politique
continue apprend une petite correction bornee autour du PID flou expert.

| Backend | Controleur | Algo | Duree | Timesteps | Erreur finale q | Erreur moyenne q | Erreur finale 3D | Erreur moyenne 3D | Violations |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| offline | fuzzy-pid | baseline | 2.0 | - | 5.2600e-03 | 1.7815e-01 | 6.5774e-03 | 1.5207e-01 | 0 |
| offline | sac-residual | sac | 2.0 | 300 | 5.5657e-03 | 1.8257e-01 | 6.9387e-03 | 1.5575e-01 | 0 |
| offline | td3-residual | td3 | 2.0 | 300 | 1.0280e-02 | 1.8266e-01 | 1.0569e-02 | 1.5607e-01 | 0 |
| coppeliasim | fuzzy-pid | baseline | 0.5 | - | 1.2345e+00 | 7.9053e-01 | 3.7409e-01 | 2.0688e-01 | 0 |
| coppeliasim | sac-residual | sac | 0.5 | 20 | 1.0592e+00 | 8.0805e-01 | 3.6884e-01 | 2.7954e-01 | 0 |

Sorties principales :

- `results/drl_6dof/20260801_110708_offline_sac_cartesian_loop/`
- `results/drl_6dof/20260801_110759_offline_td3_cartesian_loop/`
- `results/drl_6dof/20260801_110759_coppelia_sac_cartesian_loop/`

Interpretation : sur `300` pas, SAC et TD3 ne cherchent pas encore une commande
optimale ; ils valident surtout le pipeline continu et le mode residuel. SAC
hors-ligne reste proche du PID flou expert en erreur finale, tandis que TD3 est
un peu moins bon sur ce budget court. Le smoke-test CoppeliaSim de `20` pas
confirme que l'entrainement et l'evaluation SAC peuvent tourner sur la scene
ouverte, mais il faudra des episodes plus longs pour juger la performance.
