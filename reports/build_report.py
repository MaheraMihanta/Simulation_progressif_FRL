"""Build a preliminary PDF report from the current FRL simulation results."""

from __future__ import annotations

import csv
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
REPORT_DIR = RESULTS / "report"
REPORT_PATH = REPORT_DIR / "rapport_simulation_frl_preliminaire.pdf"


def _add_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    *,
    footer: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")

    y = 1.0
    ax.text(0.0, y, title, fontsize=18, weight="bold", va="top")
    y -= 0.07

    for paragraph in paragraphs:
        if not paragraph:
            y -= 0.03
            continue
        for line in textwrap.wrap(paragraph, width=88):
            ax.text(0.0, y, line, fontsize=10.5, va="top")
            y -= 0.025
        y -= 0.025
        if y < 0.08:
            break

    if footer:
        fig.text(0.08, 0.04, footer, fontsize=8, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    notes: list[str],
) -> None:
    if not image_path.exists():
        _add_text_page(
            pdf,
            title,
            [f"Figure absente: {image_path.relative_to(ROOT)}"],
        )
        return

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold", va="top")

    image = mpimg.imread(image_path)
    ax_image = fig.add_axes([0.05, 0.14, 0.62, 0.74])
    ax_image.imshow(image)
    ax_image.axis("off")

    ax_notes = fig.add_axes([0.71, 0.14, 0.24, 0.74])
    ax_notes.axis("off")
    y = 1.0
    for note in notes:
        for line in textwrap.wrap(note, width=38):
            ax_notes.text(0.0, y, line, fontsize=10, va="top")
            y -= 0.045
        y -= 0.045
        if y < 0.05:
            break

    fig.text(
        0.05,
        0.05,
        str(image_path.relative_to(ROOT)),
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _read_generalization_rows() -> list[dict[str, str]]:
    csv_path = TABLES / "step_11_fuzzy_residual_generalization_2dof.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _generalization_summary() -> list[str]:
    rows = _read_generalization_rows()
    if not rows:
        return ["Le tableau de generalisation step_11 n'est pas encore disponible."]

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["target_id"], {})[row["method"]] = row

    lines = [
        "Resultats de generalisation flou/RL :",
        "",
    ]
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        learned = methods["fuzzy_rl"]
        delta_steps = int(learned["steps"]) - int(base["steps"])
        delta_distance = float(learned["final_distance"]) - float(base["final_distance"])
        delta_torque = float(learned["mean_torque_norm"]) - float(base["mean_torque_norm"])
        lines.append(
            f"{target_id}: delta pas={delta_steps:+d}, "
            f"delta distance={delta_distance:+.4e}, "
            f"delta couple={delta_torque:+.4e}, "
            f"succes flou/RL={learned['done']}"
        )
    return lines


def _safe_generalization_summary() -> list[str]:
    csv_path = TABLES / "step_12_fuzzy_residual_safe_generalization_2dof.csv"
    if not csv_path.exists():
        return ["Le tableau de generalisation securisee step_12 n'est pas encore disponible."]
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["target_id"], {})[row["method"]] = row

    lines = [
        "Resultats avec supervision de securite :",
        "",
    ]
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        safe = methods["fuzzy_rl_safe"]
        delta_steps = int(safe["steps"]) - int(base["steps"])
        delta_distance = float(safe["final_distance"]) - float(base["final_distance"])
        delta_torque = float(safe["mean_torque_norm"]) - float(base["mean_torque_norm"])
        switch = safe["residual_switch_step"] or "-"
        lines.append(
            f"{target_id}: delta pas={delta_steps:+d}, "
            f"delta distance={delta_distance:+.4e}, "
            f"delta couple={delta_torque:+.4e}, "
            f"coupure residu={switch}"
        )
    return lines


def _generalization_3dof_summary() -> list[str]:
    csv_path = TABLES / "step_22_fuzzy_residual_generalization_3dof.csv"
    if not csv_path.exists():
        return ["Le tableau de generalisation 3 DDL step_22 n'est pas encore disponible."]
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["target_id"], {})[row["method"]] = row

    lines = [
        "Resultats de generalisation flou/RL 3 DDL :",
        "",
    ]
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        raw = methods["fuzzy_rl"]
        safe = methods["fuzzy_rl_safe"]
        raw_delta_steps = int(raw["steps"]) - int(base["steps"])
        safe_delta_steps = int(safe["steps"]) - int(base["steps"])
        safe_delta_torque = float(safe["mean_torque_norm"]) - float(base["mean_torque_norm"])
        switch = safe["residual_switch_step"] or "-"
        lines.append(
            f"{target_id}: succes flou={base['done']}, "
            f"succes brut={raw['done']}, succes securise={safe['done']}, "
            f"delta pas brut={raw_delta_steps:+d}, "
            f"delta pas securise={safe_delta_steps:+d}, "
            f"delta couple securise={safe_delta_torque:+.4e}, "
            f"coupure={switch}"
        )
    return lines


def build_report(output_path: Path = REPORT_PATH) -> Path:
    """Create the preliminary PDF report and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _add_text_page(
            pdf,
            "Rapport preliminaire - Simulation FRL pour bras 2 DDL",
            [
                "Objectif general : construire progressivement une simulation qui relie commande classique, logique floue et apprentissage par renforcement pour un bras robotique planaire a deux degres de liberte.",
                "Ce document est genere par Python. Il sert de structure de depart pour le rapport final : les sections, figures et tableaux seront enrichis au fur et a mesure des implementations.",
                "La contribution actuelle se concentre sur une architecture hybride ou la logique floue fournit une commande stabilisante et ou le Q-learning apprend un residu d'acceleration.",
            ],
            footer="Generation automatique depuis reports/build_report.py",
        )
        _add_text_page(
            pdf,
            "Formulation hybride flou/RL",
            [
                "La commande finale utilise la forme q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel.",
                "Le controleur flou calcule une acceleration articulaire de base a partir de l'erreur et de sa variation. Le modele dynamique inverse transforme ensuite cette acceleration en couple moteur.",
                "Le RL n'apprend pas tout le couple moteur. Il apprend une correction discrete dans un espace de 81 regles floues construites depuis erreur_q1, erreur_q2, q1_dot et q2_dot.",
                "Cette architecture donne un role distinct a chaque composant : stabilite et interpretabilite pour le flou, adaptation par experience pour le RL.",
            ],
        )
        _add_image_page(
            pdf,
            "Experience step_10 - cible de reference",
            FIGURES / "step_10_fuzzy_residual_q_learning_2dof.png",
            [
                "Le residu RL accelere la convergence sur la cible d'entrainement.",
                "La precision finale reste comparable au flou seul.",
                "Le couple moyen augmente legerement : le gain est donc un compromis vitesse/effort.",
            ],
        )
        _add_image_page(
            pdf,
            "Experience step_11 - generalisation",
            FIGURES / "step_11_fuzzy_residual_generalization_2dof.png",
            [
                "La meme table Q floue est testee sur plusieurs cibles.",
                "La politique apprise accelere quatre cibles sur cinq.",
                "Une cible degradee montre que la robustesse globale reste a ameliorer.",
            ],
        )
        _add_text_page(
            pdf,
            "Synthese des resultats step_11",
            _generalization_summary(),
        )
        _add_image_page(
            pdf,
            "Experience step_12 - generalisation securisee",
            FIGURES / "step_12_fuzzy_residual_safe_generalization_2dof.png",
            [
                "Le superviseur coupe le residu lorsque la distance ne progresse plus.",
                "Le taux de succes revient a cinq cibles sur cinq.",
                "La robustesse augmente, mais certains gains de vitesse peuvent etre attenues.",
            ],
        )
        _add_text_page(
            pdf,
            "Synthese des resultats step_12",
            _safe_generalization_summary(),
        )
        _add_text_page(
            pdf,
            "Extension spatiale 3 DDL",
            [
                "Le bras 3 DDL ajoute une rotation de base autour de l'axe vertical au bras 2 DDL initial. La cible devient spatiale et la cinematique inverse se decompose en un azimut de base puis une IK planaire dans le plan radial-z.",
                "La meme architecture flou/RL est conservee. L'etat flou passe de quatre variables a six variables : erreur_q0, erreur_q1, erreur_q2, q0_dot, q1_dot et q2_dot.",
                "La base de regles passe ainsi de 81 a 729 regles, et l'espace d'actions residuelles de 9 a 27 actions. Cette etape mesure donc l'effet de la montee en dimension avant le passage vers des bras plus complexes.",
            ],
        )
        _add_image_page(
            pdf,
            "Experience step_21 - flou/RL 3 DDL",
            FIGURES / "step_21_fuzzy_residual_q_learning_3dof.png",
            [
                "Le controleur flou reste la politique stabilisante.",
                "Le Q-learning apprend un residu d'acceleration dans un espace de 729 regles.",
                "La comparaison avec le flou seul met en evidence l'effet de la dimension supplementaire sur le compromis vitesse/effort.",
            ],
        )
        _add_image_page(
            pdf,
            "Experience step_22 - generalisation 3 DDL",
            FIGURES / "step_22_fuzzy_residual_generalization_3dof.png",
            [
                "La table Q est entrainee sur une cible spatiale puis testee sur plusieurs cibles 3D.",
                "La version securisee conserve le controleur flou comme politique de secours.",
                "Les resultats servent de transition vers l'apprentissage multi-cibles et les methodes continues.",
            ],
        )
        _add_text_page(
            pdf,
            "Synthese des resultats step_22",
            _generalization_3dof_summary(),
        )
        _add_text_page(
            pdf,
            "Prochaines etapes",
            [
                "Entrainer l'agent hybride sur une distribution de cibles afin de reduire la dependance a une cible unique.",
                "Remplacer le superviseur heuristique par une estimation explicite de confiance ou par une decision apprise.",
                "Completer la comparaison globale : PID dynamique, flou dynamique, Q-learning residuel PID, Q-learning residuel flou, puis variantes multi-cibles.",
                "Transformer ce rapport preliminaire en rapport final lorsque les experiences seront stabilisees.",
            ],
        )

    return output_path


def main() -> int:
    output_path = build_report()
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
