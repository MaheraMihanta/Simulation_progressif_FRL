"""Train a continuous SAC/TD3 policy on the fuzzy-guided 6-DOF task."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzy_drl_sim import (
    RobotConfig,
    SimulationConfig,
    SUPPORTED_TRAJECTORIES,
    make_fuzzy_guided_gym_env,
)
from fuzzy_drl_sim.coppelia_env import CoppeliaArmEnv, CoppeliaConnectionError
from fuzzy_drl_sim.offline_env import OfflineArmEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC or TD3 on the 6-DOF fuzzy-guided tracking task.",
    )
    parser.add_argument("--algo", choices=["sac", "td3"], default="sac")
    parser.add_argument("--trajectory", choices=SUPPORTED_TRAJECTORIES, default="cartesian_loop")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--action-mode", choices=["residual", "direct"], default="residual")
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/drl_6dof"))
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logs.")
    parser.add_argument(
        "--start-from-current-state",
        action="store_true",
        help="Build each episode from the backend current state instead of q=0.",
    )
    parser.add_argument(
        "--coppelia",
        action="store_true",
        help="Use the opened CoppeliaSim scene instead of the offline plant.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("timesteps must be strictly positive.")
    if args.learning_starts < 0:
        raise ValueError("learning-starts must be zero or positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be strictly positive.")
    if args.eval_episodes < 0:
        raise ValueError("eval-episodes must be zero or positive.")
    if args.residual_scale < 0.0:
        raise ValueError("residual-scale must be non-negative.")

    try:
        from stable_baselines3 import SAC, TD3
    except ImportError:
        print(
            "stable-baselines3 and gymnasium are required for this training script.\n"
            "Install them with: pip install stable-baselines3 gymnasium",
            file=sys.stderr,
        )
        return 2

    robot = RobotConfig()
    simulation = SimulationConfig(
        dt=args.dt,
        duration=args.duration,
        output_dir=args.output_dir,
        make_plots=False,
    )
    backend = (
        CoppeliaArmEnv(robot, simulation)
        if args.coppelia
        else OfflineArmEnv(robot, simulation)
    )

    try:
        env = make_fuzzy_guided_gym_env(
            backend,
            robot,
            simulation,
            trajectory_name=args.trajectory,
            action_mode=args.action_mode,
            residual_scale=args.residual_scale,
            initial_q=None if args.start_from_current_state else np.zeros(robot.dof),
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_name = "coppelia" if args.coppelia else "offline"
    run_dir = args.output_dir / f"{stamp}_{backend_name}_{args.algo}_{args.trajectory}"
    run_dir.mkdir(parents=True, exist_ok=True)

    algorithm = SAC if args.algo == "sac" else TD3
    model = algorithm(
        "MlpPolicy",
        env,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tensorboard") if args.tensorboard else None,
    )
    try:
        model.learn(total_timesteps=args.timesteps)
        evaluation = (
            _evaluate_model(model, env, args.eval_episodes)
            if args.eval_episodes > 0
            else None
        )
    except CoppeliaConnectionError as exc:
        print(exc, file=sys.stderr)
        env.close()
        return 2
    finally:
        env.close()

    model_path = run_dir / f"{args.algo}_fuzzy_guided_6dof"
    model.save(model_path)
    summary_path = run_dir / "training_summary.json"
    summary = {
        "backend": backend_name,
        "algorithm": args.algo,
        "trajectory": args.trajectory,
        "timesteps": args.timesteps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "action_mode": args.action_mode,
        "residual_scale": args.residual_scale,
        "duration": args.duration,
        "dt": args.dt,
        "seed": args.seed,
        "tensorboard": args.tensorboard,
        "start_from_current_state": args.start_from_current_state,
        "model_path": str(model_path.with_suffix(".zip")),
        "evaluation": evaluation,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Run directory : {run_dir}")
    print(f"Model         : {model_path.with_suffix('.zip')}")
    print(f"Summary       : {summary_path}")
    if evaluation is not None:
        print("Evaluation:")
        for key, value in evaluation.items():
            print(f"  {key}: {value}")
    return 0


def _evaluate_model(model, env, episodes: int) -> dict[str, float | int]:
    episode_returns: list[float] = []
    final_errors: list[float] = []
    mean_errors: list[float] = []
    max_errors: list[float] = []
    final_cartesian_errors: list[float] = []
    mean_cartesian_errors: list[float] = []
    constraint_violations = 0
    total_steps = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        done = False
        episode_return = 0.0
        errors: list[float] = []
        cartesian_errors: list[float] = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            total_steps += 1
            errors.append(float(info.get("error_norm", 0.0)))
            cartesian_errors.append(float(info.get("cartesian_error_norm", 0.0)))
            constraint_violations += int(info.get("constraint_violations", 0))
        episode_returns.append(episode_return)
        final_errors.append(errors[-1] if errors else 0.0)
        mean_errors.append(float(np.mean(errors)) if errors else 0.0)
        max_errors.append(float(np.max(errors)) if errors else 0.0)
        final_cartesian_errors.append(cartesian_errors[-1] if cartesian_errors else 0.0)
        mean_cartesian_errors.append(float(np.mean(cartesian_errors)) if cartesian_errors else 0.0)

    return {
        "episodes": episodes,
        "steps": total_steps,
        "mean_return": float(np.mean(episode_returns)),
        "mean_final_error_norm": float(np.mean(final_errors)),
        "mean_error_norm": float(np.mean(mean_errors)),
        "max_error_norm": float(np.max(max_errors)),
        "mean_final_cartesian_error_norm": float(np.mean(final_cartesian_errors)),
        "mean_cartesian_error_norm": float(np.mean(mean_cartesian_errors)),
        "constraint_violations": constraint_violations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
