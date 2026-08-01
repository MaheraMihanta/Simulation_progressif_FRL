# Simulation validation notes

This file records the current simulation diagnosis before final article writing.

## Why the previous CoppeliaSim curves vibrated

The latest archived CoppeliaSim runs showed two problems:

1. The `multi_sine` reference did not start from the measured robot pose.
   At `t=0`, the initial joint-error norm was `0.236804 rad`, with a maximum
   absolute joint error of `0.155378 rad`.
2. The external PID correction changed sign almost every sample.
   The measured correction sign-flip ratio was about `0.95` for both archived
   `pid` and `fuzzy-pid` runs. This is consistent with the visible vibration in
   the curves and with the robot vibrating during simulation.

The likely mechanism is an outer Python PID loop fighting the internal
CoppeliaSim joint-position controller, amplified by a moving reference that starts
with a discontinuity and by raw velocity feedback in the derivative term.

## Corrections now implemented

- `multi_sine` now starts exactly from the current robot pose:
  `q_ref(0)=q(0)` and `qdot_ref(0)=0`.
- A smooth quintic start envelope ramps the multi-sine trajectory during the first
  second.
- The outer PID gains and correction limit were reduced for position-target
  control.
- The derivative term is low-pass filtered.
- The commanded target positions are now logged as `target1` to `target6`.
- The JSON summary now includes vibration indicators:
  `correction_sign_flip_ratio` and `high_frequency_error_index`.
- A `reference` controller was added to send only the desired joint trajectory.
  Use it to check whether CoppeliaSim vibrates without the external PID layer.
- The FRL/DRL task now aligns the next observation with the next reference sample.

## Validation commands

Offline smoke tests:

```powershell
python -m unittest discover -s tests
python run_nominal_tracking.py --dry-run --controller reference --duration 12 --dt 0.05
python run_nominal_tracking.py --dry-run --controller pid --duration 12 --dt 0.05
python run_nominal_tracking.py --dry-run --controller fuzzy-pid --duration 12 --dt 0.05
```

CoppeliaSim validation sequence:

```powershell
python run_nominal_tracking.py --controller reference --duration 12 --dt 0.05
python run_nominal_tracking.py --controller pid --duration 12 --dt 0.05
python run_nominal_tracking.py --controller fuzzy-pid --duration 12 --dt 0.05
```

Diagnose any run directory:

```powershell
python diagnose_tracking_results.py results\<run_reference> results\<run_pid> results\<run_fuzzy_pid>
```

## Acceptance criteria before final article writing

- Initial error norm should be close to zero for `multi_sine`.
- `reference` should not visibly vibrate in CoppeliaSim.
- `pid` and `fuzzy-pid` should have much lower sign-flip ratios than the archived
  runs, ideally below `0.10`.
- `high_frequency_error_index` should be close to the offline order of magnitude
  unless CoppeliaSim dynamics introduce expected physical oscillations.
- The FRL/DRL stage should only be trained after the reference, PID and fuzzy-PID
  baselines are stable.

## Latest validated CoppeliaSim results

Runs produced on 2026-06-16 after the corrections:

| Controller | Run directory | RMSE (rad) | Max error (rad) | Final error norm (rad) | Sign-flip ratio | High-frequency index |
|---|---|---:|---:|---:|---:|---:|
| Reference | `results_coppelia_validation/20260616_075415_coppelia_reference` | 0.007320 | 0.017286 | 0.023295 | 0.000000 | 0.000148 |
| PID | `results_coppelia_validation/20260616_075436_coppelia_pid` | 0.005089 | 0.011899 | 0.016058 | 0.009722 | 0.000156 |
| Fuzzy-PID | `results_coppelia_validation/20260616_075455_coppelia_fuzzy-pid` | 0.005401 | 0.012633 | 0.017062 | 0.009722 | 0.000153 |

Interpretation:

- The `reference` controller validates that the CoppeliaSim scene can track the
  smoothed trajectory without external PID corrections and without command
  vibration.
- The corrected PID improves tracking relative to direct reference tracking.
- The corrected fuzzy-PID is stable and smoother in correction energy, but it is
  not yet more accurate than PID on the nominal trajectory.
- FRL/DRL is currently validated at the task-interface level only. A training
  campaign is still required before using FRL/DRL results in the article.

## Scenario campaign support

Implemented scenarios:

- `nominal`: smoothed multi-sine or point-to-point reference without added
  uncertainty.
- `sensor_noise`: Gaussian noise added only to the controller observation. True
  robot state remains the basis for tracking metrics.
- `observation_delay`: the controller receives a delayed state sample.
- `trajectory_step`: a smooth mid-run reference offset tests trajectory-change
  handling.
- `combined_uncertainty`: sensor noise, observation delay and trajectory step.

Campaign command:

```powershell
python run_validation_campaign.py --duration 12 --dt 0.05
```

Short CoppeliaSim campaign produced on 2026-06-16 with `duration=6 s`:

| Scenario | Controller | RMSE (rad) | Energy | Smoothness | Sign-flip ratio | High-frequency index |
|---|---|---:|---:|---:|---:|---:|
| nominal | PID | 0.004495 | 0.000155 | 0.000024 | 0.011111 | 0.000221 |
| nominal | Fuzzy-PID | 0.004773 | 0.000116 | 0.000018 | 0.011111 | 0.000217 |
| sensor_noise | PID | 0.004538 | 0.000194 | 0.001978 | 0.176389 | 0.001009 |
| sensor_noise | Fuzzy-PID | 0.004803 | 0.000140 | 0.001221 | 0.159722 | 0.000803 |
| trajectory_step | PID | 0.004965 | 0.000191 | 0.000052 | 0.016667 | 0.000265 |
| trajectory_step | Fuzzy-PID | 0.005271 | 0.000142 | 0.000038 | 0.016667 | 0.000265 |
| combined_uncertainty | PID | 0.003601 | 0.000657 | 0.000896 | 0.079167 | 0.000639 |
| combined_uncertainty | Fuzzy-PID | 0.004067 | 0.000478 | 0.000584 | 0.076389 | 0.000532 |

Current interpretation:

- PID remains more accurate in RMSE on these short tests.
- Fuzzy-PID consistently reduces control energy and correction roughness.
- Under sensor noise and combined uncertainty, fuzzy-PID reduces the sign-flip
  and high-frequency indices relative to PID.
- These are simulation-engineering results only. They justify further tuning and
  FRL/DRL training, not final article claims yet.

## FRL/DRL task smoke evaluation

The environment `FuzzyGuidedTrackingTask` is validated with simple non-trained
policies via:

```powershell
python evaluate_frl_task.py --policy fuzzy_expert --duration 6
```

CoppeliaSim smoke results produced on 2026-06-16:

| Policy | RMSE (rad) | Max error (rad) | Final error norm (rad) | Energy | Sign-flip ratio | High-frequency index |
|---|---:|---:|---:|---:|---:|---:|
| proportional | 0.004185 | 0.010735 | 0.011115 | 0.000077 | 0.011204 | 0.000092 |
| fuzzy_expert | 0.004096 | 0.010113 | 0.010903 | 0.000089 | 0.011111 | 0.000078 |

Interpretation:

- The FRL/DRL action convention works in CoppeliaSim.
- A deterministic fuzzy expert policy is stable and slightly improves tracking
  and high-frequency behavior relative to a plain proportional policy in this
  short smoke test.
- This is still not a learned FRL/DRL result. The next simulation milestone is
  to wrap this task in a training loop once the RL dependencies are installed.
