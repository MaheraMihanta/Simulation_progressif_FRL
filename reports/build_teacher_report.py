"""Build the short teacher-facing FRL report as a PDF."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
REPORT_DIR = ROOT / "results" / "report"
REPORT_PATH = REPORT_DIR / "rapport_frl_flou_rl_enseignant.pdf"


def _wrap_lines(text: str, width: int = 88) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, replace_whitespace=False)


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
    ax.text(0.0, y, title, fontsize=17, weight="bold", va="top")
    y -= 0.065

    for paragraph in paragraphs:
        if paragraph.startswith("```"):
            continue
        is_formula = paragraph.startswith("FORMULA: ")
        is_bullet = paragraph.startswith("- ")
        content = paragraph.removeprefix("FORMULA: ")
        width = 82 if is_formula else 88
        font = "monospace" if is_formula else "sans-serif"
        size = 9.4 if is_formula else 10.2
        x = 0.03 if is_bullet else 0.0

        for line in _wrap_lines(content, width=width):
            ax.text(x, y, line, fontsize=size, family=font, va="top")
            y -= 0.024
        y -= 0.012 if is_formula else 0.018
        if y < 0.06:
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
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.95, title, fontsize=15.5, weight="bold", va="top")

    if image_path.exists():
        image = mpimg.imread(image_path)
        ax_image = fig.add_axes([0.045, 0.13, 0.65, 0.74])
        ax_image.imshow(image)
        ax_image.axis("off")
    else:
        ax_image = fig.add_axes([0.045, 0.13, 0.65, 0.74])
        ax_image.text(0.5, 0.5, f"Figure absente:\n{image_path}", ha="center")
        ax_image.axis("off")

    ax_notes = fig.add_axes([0.72, 0.13, 0.23, 0.74])
    ax_notes.axis("off")
    y = 1.0
    for note in notes:
        for line in _wrap_lines(note, width=38):
            ax_notes.text(0.0, y, line, fontsize=10, va="top")
            y -= 0.045
        y -= 0.04

    fig.text(0.05, 0.055, str(image_path.relative_to(ROOT)), fontsize=8, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def _add_two_image_page(
    pdf: PdfPages,
    title: str,
    top_image: Path,
    bottom_image: Path,
    notes: list[str],
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.95, title, fontsize=15.5, weight="bold", va="top")

    for image_path, y0 in ((top_image, 0.52), (bottom_image, 0.12)):
        ax_image = fig.add_axes([0.05, y0, 0.60, 0.34])
        if image_path.exists():
            ax_image.imshow(mpimg.imread(image_path))
        else:
            ax_image.text(0.5, 0.5, f"Figure absente:\n{image_path}", ha="center")
        ax_image.axis("off")
        fig.text(0.05, y0 - 0.02, str(image_path.relative_to(ROOT)), fontsize=7.5, color="0.35")

    ax_notes = fig.add_axes([0.70, 0.14, 0.25, 0.72])
    ax_notes.axis("off")
    y = 1.0
    for note in notes:
        for line in _wrap_lines(note, width=40):
            ax_notes.text(0.0, y, line, fontsize=10, va="top")
            y -= 0.045
        y -= 0.04

    pdf.savefig(fig)
    plt.close(fig)


def build_report(output_path: Path = REPORT_PATH) -> Path:
    """Create the short PDF report and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _add_text_page(
            pdf,
            "Commande floue et RL pour bras robotique 2 DDL",
            [
                "Rapport court destine a un enseignant-chercheur. L'objectif est de justifier la methode a adopter pour la suite du projet : combiner logique floue et apprentissage par renforcement, d'abord en Python, puis dans CoppeliaSim.",
                "Le cahier des charges demandait une progression 2 DDL -> 3 DDL -> 6 DDL, avec formulation mathematique, comparaison PID/flou/RL et mise en evidence des notions RL : etat, action, reward, return, politique, valeur et Bellman.",
                "L'hypothese defendue est que le flou/RL est plus prudent qu'un RL pur : le flou stabilise et rend la commande interpretable, tandis que le RL apprend une correction residuelle par interaction.",
            ],
            footer="Source Markdown: reports/rapport_frl_flou_rl_enseignant.md",
        )
        _add_text_page(
            pdf,
            "Rappel conceptuel : logique floue",
            [
                "La logique floue commande un systeme avec des concepts linguistiques : erreur negative, nulle, positive ; vitesse lente ou rapide ; commande faible ou forte.",
                "Une variable n'appartient pas a un seul ensemble : elle active plusieurs fonctions d'appartenance avec des degres compris entre 0 et 1.",
                "Dans ce projet, le controleur flou calcule une acceleration articulaire a partir de l'erreur et de sa variation.",
                "FORMULA: e = q_desire - q",
                "FORMULA: de/dt ~= (e(t) - e(t-dt)) / dt",
                "FORMULA: q_ddot_flou = defuzzification(regles(e, de/dt))",
                "Interet pour le projet : une politique lisible, stable, peu couteuse, qui peut servir de base a l'apprentissage.",
            ],
        )
        _add_text_page(
            pdf,
            "Rappel conceptuel : apprentissage par renforcement",
            [
                "Le RL modelise le probleme comme un processus de decision de Markov.",
                "FORMULA: MDP = (S, A, P, R, gamma)",
                "Un etat decrit la situation, une action modifie le systeme, un reward mesure la qualite de la transition, et gamma actualise les gains futurs.",
                "FORMULA: G = r0 + gamma r1 + gamma^2 r2 + ... + gamma^T rT",
                "FORMULA: V_pi(s) = E_pi[G | S0 = s]",
                "FORMULA: Q_pi(s,a) = E_pi[G | S0 = s, A0 = a]",
                "FORMULA: V*(s) = max_a sum_s' P(s'|s,a)[R(s,a,s') + gamma V*(s')]",
                "FORMULA: Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]",
                "Dans notre travail, le RL n'apprend pas directement tout le couple moteur : il apprend un residu autour d'un controleur stabilisant.",
            ],
        )
        _add_text_page(
            pdf,
            "Modele du bras 2 DDL",
            [
                "Le bras est planaire, compose de deux segments de longueurs l1 et l2 et de deux articulations rotatives q1, q2.",
                "FORMULA: x = l1 cos(q1) + l2 cos(q1 + q2)",
                "FORMULA: y = l1 sin(q1) + l2 sin(q1 + q2)",
                "La dynamique utilisee dans la simulation a couples est :",
                "FORMULA: M(q) q_ddot + C(q,q_dot) q_dot + G(q) + F q_dot = tau",
                "La commande a couple calcule convertit une acceleration desiree en couples moteurs :",
                "FORMULA: tau = M(q) q_ddot_cmd + C(q,q_dot)q_dot + G(q) + F q_dot",
                "Cette formulation permet de comparer PID, flou et RL dans le meme environnement physique.",
            ],
        )
        _add_text_page(
            pdf,
            "Architecture proposee : flou/RL residuel",
            [
                "La commande finale combine la politique floue et une correction apprise par Q-learning.",
                "FORMULA: q_ddot_cmd = q_ddot_flou + q_ddot_RL_residuel",
                "FORMULA: tau = M(q) q_ddot_cmd + C(q,q_dot)q_dot + G(q) + F q_dot",
                "L'etat continu du RL est :",
                "FORMULA: x = (erreur_q1, erreur_q2, q1_dot, q2_dot)",
                "Chaque variable est projetee sur negative, zero, positive, donc 3^4 = 81 regles floues.",
                "FORMULA: Q_flou(x,a) = somme_i w_i(x) Q(regle_i, a)",
                "FORMULA: delta = r + gamma max_a' Q_flou(x',a') - Q_flou(x,a)",
                "FORMULA: Q(regle_i,a) <- Q(regle_i,a) + alpha w_i(x) delta",
                "Le superviseur de securite coupe le residu si la distance ne progresse plus, et revient au controleur flou seul.",
            ],
        )
        _add_two_image_page(
            pdf,
            "Resultats : references PID et flou dynamiques",
            FIGURES / "step_06_pid_dynamic_2dof.png",
            FIGURES / "step_07_fuzzy_dynamic_2dof.png",
            [
                "PID dynamique : 131 pas, distance finale 1.38e-03, couple moyen 13.66 N.m.",
                "Flou dynamique : 360 pas, distance finale 5.23e-03, couple moyen 13.85 N.m.",
                "Le flou est plus lent ici, mais il converge et fournit une politique interpretable pour l'hybridation.",
            ],
        )
        _add_image_page(
            pdf,
            "Resultat : Q-learning tabulaire",
            FIGURES / "step_08_q_learning_2dof.png",
            [
                "Le Q-learning retrouve une politique efficace sur le MDP discret.",
                "La trajectoire apprise atteint la cible en 8 actions, comme la solution optimale calculee par value iteration.",
                "Cette etape valide les notions RL fondamentales avant de passer a la dynamique.",
            ],
        )
        _add_image_page(
            pdf,
            "Resultat : flou/RL sur cible de reference",
            FIGURES / "step_10_fuzzy_residual_q_learning_2dof.png",
            [
                "Flou seul : 360 pas, distance finale 0.005227, couple moyen 13.853 N.m.",
                "Flou + Q residuel : 269 pas, distance finale 0.005537, couple moyen 14.420 N.m.",
                "Gain : 91 pas plus rapide, avec un effort moteur legerement plus eleve.",
            ],
        )
        _add_two_image_page(
            pdf,
            "Resultats : generalisation et securite",
            FIGURES / "step_11_fuzzy_residual_generalization_2dof.png",
            FIGURES / "step_12_fuzzy_residual_safe_generalization_2dof.png",
            [
                "Generalisation brute : flou seul 5/5, flou + Q 4/5. Le residu accelere quatre cibles mais degrade T4_high.",
                "Version securisee : retour a 5/5 avec un gain moyen de 55 pas par rapport au flou seul.",
                "Conclusion : le RL apporte une acceleration, mais le flou doit rester la politique de secours.",
            ],
        )
        _add_text_page(
            pdf,
            "Conclusion et suite recommandee",
            [
                "Le travail valide une progression coherente : modelisation 2 DDL, PID, flou, RL discret, RL dynamique residuel, puis flou/RL securise.",
                "La methode a privilegier est l'hybridation flou/RL : elle est interpretable, plus stable qu'un RL pur, et assez flexible pour etre transferee vers CoppeliaSim.",
                "Le resultat actuel ne prouve pas une superiorite definitive ; il montre un compromis scientifique exploitable : convergence plus rapide contre effort legerement plus eleve.",
                "Prochaines etapes : entrainement multi-cibles, reward plus fin pour l'effort, estimation de confiance du residu, puis connexion a CoppeliaSim via la boucle interactive deja implementee.",
            ],
        )

    return output_path


def main() -> int:
    output = build_report()
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
