"""Launch a live interactive Python simulation of the dynamic 2-DOF arm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs import Arm2DOFDynamicEnvConfig
from interactive import CONTROLLER_MODES, LiveArm2DOFConfig, LiveArm2DOFSimulation
from robot import joint_positions, workspace_radius
from rl import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    train_fuzzy_residual_q_learning,
)


DEFAULT_POLICY_PATH = ROOT / "results" / "policies" / "fuzzy_residual_live_q_table.npz"
MODE_LABELS = {
    "pid": "PID",
    "fuzzy": "Flou",
    "fuzzy_rl_safe": "Flou+Q safe",
}
LABEL_TO_MODE = {label: mode for mode, label in MODE_LABELS.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live interactive simulation for the 2-DOF FRL arm.",
    )
    parser.add_argument(
        "--mode",
        choices=CONTROLLER_MODES,
        default="fuzzy",
        help="Controller used at startup.",
    )
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=(1.1, 0.55),
        help="Initial Cartesian target.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=3,
        help="Physics/control steps performed at each UI frame.",
    )
    parser.add_argument(
        "--train-rl",
        action="store_true",
        help="Train and cache a fuzzy residual Q table before launching.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=220,
        help="Training episodes used when --train-rl is passed.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the cached fuzzy residual Q table.",
    )
    parser.add_argument(
        "--headless-smoke-test",
        action="store_true",
        help="Run a short simulation without opening the UI.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of steps for --headless-smoke-test.",
    )
    return parser


def _learning_config(episodes: int) -> FuzzyResidualQLearningConfig:
    return FuzzyResidualQLearningConfig(
        episodes=episodes,
        max_steps_per_episode=550,
        alpha=0.35,
        gamma=0.97,
        epsilon_start=0.75,
        epsilon_end=0.04,
        epsilon_decay=0.985,
        residual_acceleration_scale=(1.5, 1.5),
        initial_q_value=-0.5,
        seed=17,
    )


def _load_policy(path: Path, encoder: FuzzyDynamicStateEncoder) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        q_value = np.asarray(data["q_value"], dtype=float)
    if q_value.shape != (encoder.n_rules, 9):
        raise ValueError(f"policy at {path} has invalid shape {q_value.shape}.")
    print(f"policy_loaded={path}")
    return q_value


def _train_policy(
    path: Path,
    target: tuple[float, float],
    encoder: FuzzyDynamicStateEncoder,
    learning_config: FuzzyResidualQLearningConfig,
) -> np.ndarray:
    env_config = Arm2DOFDynamicEnvConfig(
        target=target,
        dt=0.01,
        max_torque=(60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=550,
    )
    result = train_fuzzy_residual_q_learning(
        env_config,
        encoder=encoder,
        config=learning_config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q_value=result.q_value,
        target=np.asarray(target, dtype=float),
        episodes=np.asarray([learning_config.episodes], dtype=int),
    )
    success_rate = float(np.mean(result.episode_success[-60:]))
    print(f"policy_trained={path}")
    print(f"success_rate_last_60={success_rate:.3f}")
    return result.q_value


def _policy_for_args(
    args: argparse.Namespace,
    encoder: FuzzyDynamicStateEncoder,
    learning_config: FuzzyResidualQLearningConfig,
) -> np.ndarray | None:
    target = (float(args.target[0]), float(args.target[1]))
    if args.train_rl:
        return _train_policy(args.policy, target, encoder, learning_config)
    return _load_policy(args.policy, encoder)


def _run_headless_smoke_test(
    sim: LiveArm2DOFSimulation,
    steps: int,
) -> int:
    for _ in range(steps):
        sim.step()
    summary = sim.summary()
    print(f"mode={summary['mode']}")
    print(f"steps={summary['step']}")
    print(f"distance={summary['distance']:.12e}")
    print(f"speed={summary['speed']:.12e}")
    print(f"torque_norm={summary['torque_norm']:.12e}")
    print(f"message={summary['message']}")
    return 0 if np.isfinite(float(summary["distance"])) else 1


class LiveMatplotlibApp:
    """Matplotlib cockpit for the live 2-DOF arm simulation."""

    def __init__(
        self,
        sim: LiveArm2DOFSimulation,
        steps_per_frame: int = 3,
    ) -> None:
        if steps_per_frame <= 0:
            raise ValueError("steps_per_frame must be strictly positive.")
        self.sim = sim
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self._message = sim.status_message
        self._build_figure()

    def show(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self._animation = FuncAnimation(  # noqa: B020 - kept alive by the app.
            self.fig,
            self._update,
            interval=33,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def _build_figure(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.fig.canvas.manager.set_window_title("FRL live - bras 2DDL")
        grid = self.fig.add_gridspec(
            3,
            4,
            height_ratios=(3.1, 1.4, 0.52),
            width_ratios=(2.1, 2.1, 1.7, 1.1),
        )
        self.ax_arm = self.fig.add_subplot(grid[:2, :2])
        self.ax_distance = self.fig.add_subplot(grid[0, 2:])
        self.ax_torque = self.fig.add_subplot(grid[1, 2:])
        self.ax_status = self.fig.add_subplot(grid[2, :2])
        self.ax_status.axis("off")

        self._configure_arm_axis()
        (self.arm_line,) = self.ax_arm.plot([], [], "-o", linewidth=4, markersize=8)
        (self.path_line,) = self.ax_arm.plot([], [], linewidth=1.8, alpha=0.8)
        self.target_artist = self.ax_arm.scatter([], [], s=110, marker="x", color="tab:red")
        self.ee_artist = self.ax_arm.scatter([], [], s=70, color="tab:green")
        (self.distance_line,) = self.ax_distance.plot([], [], color="tab:blue")
        (self.torque_line,) = self.ax_torque.plot([], [], color="tab:orange")
        self.status_text = self.ax_status.text(
            0.01,
            0.55,
            "",
            va="center",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        radio_ax = self.fig.add_axes([0.72, 0.035, 0.14, 0.12])
        labels = [MODE_LABELS[mode] for mode in CONTROLLER_MODES]
        active = CONTROLLER_MODES.index(self.sim.mode)
        self.mode_radio = RadioButtons(radio_ax, labels, active=active)
        self.mode_radio.on_clicked(self._on_mode_clicked)

        pause_ax = self.fig.add_axes([0.88, 0.112, 0.08, 0.042])
        reset_ax = self.fig.add_axes([0.88, 0.064, 0.08, 0.042])
        kick_ax = self.fig.add_axes([0.88, 0.016, 0.08, 0.042])
        self.pause_button = Button(pause_ax, "Pause")
        self.reset_button = Button(reset_ax, "Reset")
        self.kick_button = Button(kick_ax, "Kick")
        self.pause_button.on_clicked(lambda _event: self._toggle_pause())
        self.reset_button.on_clicked(lambda _event: self._reset())
        self.kick_button.on_clicked(lambda _event: self._kick())

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.92,
            bottom=0.20,
            wspace=0.55,
            hspace=0.55,
        )

    def _configure_arm_axis(self) -> None:
        import matplotlib.pyplot as plt

        r_min, r_max = workspace_radius(self.sim.link_lengths)
        reach = r_max
        margin = 0.15 * reach
        self.ax_arm.set_xlim(-reach - margin, reach + margin)
        self.ax_arm.set_ylim(-reach - margin, reach + margin)
        self.ax_arm.set_aspect("equal", adjustable="box")
        self.ax_arm.grid(True, alpha=0.28)
        self.ax_arm.set_xlabel("x")
        self.ax_arm.set_ylabel("y")
        self.ax_arm.set_title("Simulation dynamique interactive")
        outer = plt.Circle(
            (0.0, 0.0),
            r_max,
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.35,
        )
        inner = plt.Circle(
            (0.0, 0.0),
            r_min,
            fill=False,
            linestyle=":",
            linewidth=1,
            alpha=0.25,
        )
        self.ax_arm.add_patch(outer)
        self.ax_arm.add_patch(inner)

    def _update(self, _frame: int) -> list[object]:
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.sim.step()

        summary = self.sim.summary()
        history = self.sim.history_arrays()
        positions = joint_positions(summary["q"], self.sim.link_lengths)
        self.arm_line.set_data(positions[:, 0], positions[:, 1])
        self.path_line.set_data(
            history["end_effector"][:, 0],
            history["end_effector"][:, 1],
        )
        self.target_artist.set_offsets(np.asarray([summary["target"]], dtype=float))
        self.ee_artist.set_offsets(np.asarray([summary["end_effector"]], dtype=float))

        distance = history["distance"]
        x_distance = np.arange(distance.size)
        self.distance_line.set_data(x_distance, distance)
        self.ax_distance.set_xlim(0, max(50, distance.size))
        self.ax_distance.set_ylim(0.0, max(0.05, float(np.max(distance)) * 1.15))
        self.ax_distance.set_title("Distance cible")
        self.ax_distance.grid(True, alpha=0.28)

        torque_norm = np.linalg.norm(history["torque"], axis=1)
        x_torque = np.arange(torque_norm.size)
        self.torque_line.set_data(x_torque, torque_norm)
        self.ax_torque.set_xlim(0, max(50, torque_norm.size))
        self.ax_torque.set_ylim(0.0, max(1.0, float(np.max(torque_norm)) * 1.15))
        self.ax_torque.set_title("Norme couple moteur")
        self.ax_torque.grid(True, alpha=0.28)

        residual = "off" if summary["residual_disabled"] else summary["action_name"]
        pause = "paused" if self.paused else "running"
        self.status_text.set_text(
            " | ".join(
                [
                    pause,
                    f"mode={MODE_LABELS[summary['mode']]}",
                    f"step={summary['step']}",
                    f"d={summary['distance']:.4f}",
                    f"speed={summary['speed']:.4f}",
                    f"tau={summary['torque_norm']:.2f}",
                    f"res={residual}",
                    str(summary["message"]),
                ]
            )
        )
        return [
            self.arm_line,
            self.path_line,
            self.target_artist,
            self.ee_artist,
            self.distance_line,
            self.torque_line,
            self.status_text,
        ]

    def _on_click(self, event) -> None:  # noqa: ANN001 - Matplotlib callback.
        if event.inaxes is not self.ax_arm or event.xdata is None or event.ydata is None:
            return
        try:
            self.sim.set_target((float(event.xdata), float(event.ydata)))
        except ValueError as exc:
            self.sim.status_message = str(exc)

    def _on_key(self, event) -> None:  # noqa: ANN001 - Matplotlib callback.
        if event.key == " ":
            self._toggle_pause()
        elif event.key == "r":
            self._reset()
        elif event.key == "p":
            self._kick()
        elif event.key in ("1", "2", "3"):
            self.mode_radio.set_active(int(event.key) - 1)

    def _on_mode_clicked(self, label: str) -> None:
        self.sim.set_mode(LABEL_TO_MODE[label])

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.label.set_text("Run" if self.paused else "Pause")

    def _reset(self) -> None:
        self.sim.reset_robot(clear_history=True)

    def _kick(self) -> None:
        self.sim.apply_disturbance((14.0, -9.0))


def main() -> int:
    args = _parser().parse_args()
    if args.headless_smoke_test:
        import matplotlib

        matplotlib.use("Agg")

    target = (float(args.target[0]), float(args.target[1]))
    encoder = FuzzyDynamicStateEncoder(
        error_scale=(0.9, 1.2),
        velocity_scale=(6.0, 6.0),
    )
    learning_config = _learning_config(args.episodes)
    q_value = _policy_for_args(args, encoder, learning_config)
    if q_value is None and args.mode == "fuzzy_rl_safe":
        print("policy_loaded=none; fuzzy_rl_safe starts with zero residuals")

    sim = LiveArm2DOFSimulation(
        config=LiveArm2DOFConfig(target=target),
        mode=args.mode,
        q_value=q_value,
        encoder=encoder,
        learning_config=learning_config,
    )
    if args.headless_smoke_test:
        return _run_headless_smoke_test(sim, args.steps)

    app = LiveMatplotlibApp(sim, steps_per_frame=args.steps_per_frame)
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
