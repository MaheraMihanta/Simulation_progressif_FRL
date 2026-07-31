"""Build a standalone summary PDF for the spatial 3-DOF arm study."""

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
REPORT_PATH = REPORT_DIR / "resume_bras_3ddl.pdf"


def _wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    return textwrap.wrap(
        line,
        width=width,
        break_long_words=False,
        replace_whitespace=False,
    )


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
        if paragraph.startswith("```"):
            y -= 0.01
            continue
        is_formula = paragraph.startswith("    ")
        font_family = "monospace" if is_formula else "sans-serif"
        font_size = 8.6 if is_formula else 10.5
        width = 82 if is_formula else 88
        text = paragraph[4:] if is_formula else paragraph

        for line in _wrap_line(text, width):
            ax.text(0.0, y, line, fontsize=font_size, family=font_family, va="top")
            y -= 0.023 if is_formula else 0.025
        y -= 0.018
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
    ax_image = fig.add_axes([0.05, 0.13, 0.64, 0.75])
    ax_image.imshow(image)
    ax_image.axis("off")

    ax_notes = fig.add_axes([0.73, 0.13, 0.22, 0.75])
    ax_notes.axis("off")
    y = 1.0
    for note in notes:
        for line in textwrap.wrap(note, width=35, break_long_words=False):
            ax_notes.text(0.0, y, line, fontsize=10, va="top")
            y -= 0.045
        y -= 0.04
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


def _resource_scaling_rows() -> list[tuple[int, int, int, int, str]]:
    rows = []
    for dof in range(2, 7):
        rules = 3 ** (2 * dof)
        actions = 3**dof
        entries = rules * actions
        memory_mb = entries * 8.0 / (1024.0 * 1024.0)
        if memory_mb < 1.0:
            memory = f"{memory_mb * 1024.0:.1f} KB"
        elif memory_mb < 1024.0:
            memory = f"{memory_mb:.1f} MB"
        else:
            memory = f"{memory_mb / 1024.0:.2f} GB"
        rows.append((dof, rules, actions, entries, memory))
    return rows


def _resource_scaling_text() -> list[str]:
    lines = [
        "La consommation augmente de maniere combinatoire si l'on conserve la meme structure tabulaire floue/RL.",
        "Pour n degres de liberte, l'etat flou utilise ici deux variables par articulation : erreur articulaire et vitesse articulaire.",
        "Avec trois termes linguistiques par variable, le nombre de regles vaut :",
        "    N_regles(n) = 3^(2n) = 9^n",
        "Si le residu RL garde trois choix par articulation (-, 0, +), le nombre d'actions vaut :",
        "    N_actions(n) = 3^n",
        "La table Q contient donc :",
        "    N_Q(n) = N_regles(n) N_actions(n) = 3^(3n) = 27^n",
        "Evolution indicative en float64 :",
        "    DDL | regles | actions | valeurs Q | memoire table Q",
    ]
    for dof, rules, actions, entries, memory in _resource_scaling_rows():
        lines.append(
            f"    {dof:>3} | {rules:>7} | {actions:>7} | {entries:>10} | {memory:>15}"
        )
    lines.extend(
        [
            "Conclusion : le passage 2 DDL -> 3 DDL reste leger, mais le 6 DDL devient tres couteux avec une table Q dense.",
            "Le probleme n'est pas seulement la memoire. A chaque pas, l'encodeur actuel calcule les activations sur toute la base de regles. En 6 DDL, cela represente 531441 regles a parcourir par pas de simulation.",
            "Pour 6 DDL, il faudra eviter une table dense : encodeur flou creux, apprentissage factorise par articulation, approximation de fonction, acteur-critique continu ou apprentissage hierarchique.",
        ]
    )
    return lines


def _read_step22_rows() -> list[dict[str, str]]:
    csv_path = TABLES / "step_22_fuzzy_residual_generalization_3dof.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _step22_summary_text() -> list[str]:
    rows = _read_step22_rows()
    if not rows:
        return ["Le tableau step_22 n'est pas encore disponible."]

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["target_id"], {})[row["method"]] = row

    target_count = len(grouped)
    base_successes = sum(rowset["fuzzy_base"]["done"] == "True" for rowset in grouped.values())
    raw_successes = sum(rowset["fuzzy_rl"]["done"] == "True" for rowset in grouped.values())
    safe_successes = sum(rowset["fuzzy_rl_safe"]["done"] == "True" for rowset in grouped.values())

    lines = [
        "Synthese multi-cibles 3 DDL :",
        f"    flou seul        : succes {base_successes}/{target_count}",
        f"    flou + Q brut    : succes {raw_successes}/{target_count}",
        f"    flou + Q securise: succes {safe_successes}/{target_count}",
        "",
        "Detail par cible :",
        "    cible        | flou pas/dist | flou+Q pas/dist | securise pas/dist",
    ]
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        raw = methods["fuzzy_rl"]
        safe = methods["fuzzy_rl_safe"]
        lines.append(
            "    "
            f"{target_id:<12} | "
            f"{int(base['steps']):>4}/{float(base['final_distance']):.4f} | "
            f"{int(raw['steps']):>4}/{float(raw['final_distance']):.4f} | "
            f"{int(safe['steps']):>4}/{float(safe['final_distance']):.4f}"
        )
    lines.extend(
        [
            "",
            "Lecture : le residu RL 3 DDL conserve la convergence sur les cinq cibles. Les gains de temps sont modestes, car le residu a ete volontairement regle de facon douce pour ne pas degrader le controleur flou de base.",
        ]
    )
    return lines


def build_report(output_path: Path = REPORT_PATH) -> Path:
    """Create the standalone 3-DOF summary PDF and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _add_text_page(
            pdf,
            "Resume - Bras robotique spatial 3 DDL",
            [
                "Objet : presenter uniquement l'extension du bras 2 DDL planaire vers un bras 3 DDL spatial, avec les formulations mathematiques, les outils utilises et les resultats de simulation.",
                "Le document ne reprend pas les bases generales de la logique floue ni de l'apprentissage par renforcement. Ces bases restent dans le resume FRL principal.",
                "Configuration retenue : une base rotative autour de l'axe vertical z, sur laquelle est superpose le bras 2 DDL. Le bras devient spatial car la base choisit le plan de travail radial-vertical.",
            ],
            footer="Generation automatique depuis reports/build_3dof_summary_report.py",
        )
        _add_text_page(
            pdf,
            "Croissance des ressources de calcul",
            _resource_scaling_text(),
        )
        _add_text_page(
            pdf,
            "Cinematique directe 3 DDL",
            [
                "Le vecteur articulaire est q = [q0, q1, q2]^T. q0 est le lacet de base, q1 l'epaule et q2 le coude.",
                "Dans le plan radial-z du bras 2R :",
                "    rho(q) = l1 cos(q1) + l2 cos(q1 + q2)",
                "    z(q)   = l1 sin(q1) + l2 sin(q1 + q2)",
                "La rotation de base projette ce point dans l'espace :",
                "    x(q) = rho(q) cos(q0)",
                "    y(q) = rho(q) sin(q0)",
                "    z(q) = l1 sin(q1) + l2 sin(q1 + q2)",
                "La position du coude est obtenue avec rho1 = l1 cos(q1) et z1 = l1 sin(q1) :",
                "    p_coude = [rho1 cos(q0), rho1 sin(q0), z1]^T",
                "    p_eff   = [rho cos(q0),  rho sin(q0),  z]^T",
            ],
        )
        _add_text_page(
            pdf,
            "Cinematique inverse et Jacobienne",
            [
                "Pour une cible p* = (x*, y*, z*), le probleme se separe en deux etapes.",
                "La base s'oriente vers la projection horizontale de la cible :",
                "    q0 = atan2(y*, x*)",
                "    rho* = sqrt(x*^2 + y*^2)",
                "Le reste est une cinematique inverse planaire dans le plan (rho, z) :",
                "    r2 = rho*^2 + z*^2",
                "    c2 = (r2 - l1^2 - l2^2) / (2 l1 l2)",
                "    q2 = atan2(s2, c2)",
                "    q1 = atan2(z*, rho*) - atan2(l2 s2, l1 + l2 c2)",
                "avec s2 = +/- sqrt(1 - c2^2). Le signe choisit la configuration coude haut ou coude bas.",
                "Le domaine atteignable est une coque spherique :",
                "    abs(l1 - l2) <= sqrt(x*^2 + y*^2 + z*^2) <= l1 + l2",
                "La Jacobienne utilise rho, drho/dq1, drho/dq2, dz/dq1 et dz/dq2 :",
                "    J(q) = [ -rho sin(q0)   drho1 cos(q0)   drho2 cos(q0) ]",
                "           [  rho cos(q0)   drho1 sin(q0)   drho2 sin(q0) ]",
                "           [       0              dz1              dz2      ]",
                "avec drho1 = -l1 sin(q1) - l2 sin(q1+q2), drho2 = -l2 sin(q1+q2), dz1 = rho et dz2 = l2 cos(q1+q2).",
            ],
        )
        _add_text_page(
            pdf,
            "Modele dynamique 3 DDL",
            [
                "Le modele dynamique conserve la forme generale utilisee dans les simulations a couples :",
                "    M(q) q_ddot + C(q,q_dot)q_dot + G(q) + F q_dot = tau",
                "Le sous-systeme epaule-coude reprend la dynamique planaire 2R dans le plan vertical. La base ajoute une inertie de lacet dependant de la posture.",
                "Avec r1 et r2 les distances des centres de masse :",
                "    rho_c1 = r1 cos(q1)",
                "    rho_c2 = l1 cos(q1) + r2 cos(q1 + q2)",
                "    I_yaw(q) = I0 + m1 rho_c1^2 + m2 rho_c2^2",
                "La matrice d'inertie est :",
                "    M_3(q) = [ I_yaw(q)   0      0   ]",
                "             [    0      M11    M12  ]",
                "             [    0      M12    M22  ]",
                "ou :",
                "    M11 = I1 + I2 + m1 r1^2 + m2(l1^2 + r2^2 + 2 l1 r2 cos(q2))",
                "    M12 = I2 + m2(r2^2 + l1 r2 cos(q2))",
                "    M22 = I2 + m2 r2^2",
            ],
        )
        _add_text_page(
            pdf,
            "Termes passifs et commande",
            [
                "Le couple de gravite ne s'applique pas directement sur l'axe de lacet :",
                "    G(q) = [0, G1, G2]^T",
                "    G1 = (m1 r1 + m2 l1) g cos(q1) + m2 r2 g cos(q1 + q2)",
                "    G2 = m2 r2 g cos(q1 + q2)",
                "Les termes centrifuges et Coriolis du plan 2R utilisent h = m2 l1 r2 sin(q2). Le modele ajoute aussi l'effet de la variation de I_yaw :",
                "    C0 = d(I_yaw)/dt q0_dot",
                "    C1 = -h(2 q1_dot q2_dot + q2_dot^2) - 0.5 d(I_yaw)/dq1 q0_dot^2",
                "    C2 =  h q1_dot^2 - 0.5 d(I_yaw)/dq2 q0_dot^2",
                "Le frottement est visqueux :",
                "    F q_dot = [f0 q0_dot, f1 q1_dot, f2 q2_dot]^T",
                "La commande a couple calcule utilise :",
                "    tau = M(q) q_ddot_cmd + C(q,q_dot)q_dot + G(q) + F q_dot",
                "Dans les experiences flou/RL, q_ddot_cmd est la somme d'une acceleration floue stabilisante et d'un residu RL discret.",
            ],
        )
        _add_text_page(
            pdf,
            "Outils necessaires",
            [
                "Les simulations 3 DDL utilisent uniquement la chaine Python deja presente dans le projet.",
                "Modules principaux :",
                "    src/robot/kinematics_3dof.py",
                "    src/robot/dynamics_3dof.py",
                "    src/envs/arm_3dof_env.py",
                "    src/envs/arm_3dof_dynamic_env.py",
                "    src/rl/fuzzy_residual_q_learning_3dof.py",
                "    src/visualization/plots.py",
                "Bibliotheques : NumPy pour le calcul matriciel, Matplotlib pour les figures et le PDF, unittest pour la validation.",
                "Scripts d'experience : run_kinematics_3dof.py, run_pid_3dof.py, run_fuzzy_3dof.py, run_pid_dynamic_3dof.py, run_fuzzy_dynamic_3dof.py, run_fuzzy_residual_q_learning_3dof.py et run_fuzzy_residual_generalization_3dof.py.",
            ],
        )
        _add_text_page(
            pdf,
            "Resultats numeriques",
            [
                "Cible de reference : (0.95, 0.55, 0.50).",
                "Cinematique inverse : erreur finale 1.57e-16, ce qui valide la formulation analytique.",
                "PID cinematique : 32 pas, distance finale 9.15e-03.",
                "Flou cinematique : 58 pas, distance finale 9.81e-03.",
                "PID dynamique : 133 pas, distance finale 2.68e-03, vitesse finale 7.68e-02, couple moyen 14.48 N.m.",
                "Flou dynamique : 372 pas, distance finale 9.22e-03, vitesse finale 7.90e-02, couple moyen 15.47 N.m.",
                "Flou/RL 3 DDL : 729 regles floues, 27 actions residuelles, succes sur la cible de reference en 376 pas, distance finale 7.93e-03.",
            ],
        )
        _add_text_page(pdf, "Synthese step_22", _step22_summary_text())
        _add_image_page(
            pdf,
            "Step 16 - Cinematique inverse 3 DDL",
            FIGURES / "step_16_kinematics_3dof.png",
            [
                "La base oriente le bras vers la projection horizontale de la cible.",
                "Le sous-probleme restant est resolu dans le plan radial-z.",
                "L'erreur numerique est de l'ordre de 1e-16.",
            ],
        )
        _add_image_page(
            pdf,
            "Steps 17-18 - Commande cinematique",
            FIGURES / "step_18_fuzzy_3dof.png",
            [
                "Le controleur flou cinematique atteint la cible en 58 pas.",
                "Le PID cinematique, plus direct, atteint la meme tolerance en 32 pas.",
                "Les deux valident l'environnement cinematique 3 DDL.",
            ],
        )
        _add_image_page(
            pdf,
            "Step 20 - Controle flou dynamique",
            FIGURES / "step_20_fuzzy_dynamic_3dof.png",
            [
                "Le controleur flou est interprete comme une commande d'acceleration.",
                "Le modele dynamique inverse transforme cette acceleration en couple moteur.",
                "La convergence est plus lente que le PID, mais reste stable.",
            ],
        )
        _add_image_page(
            pdf,
            "Step 21 - Flou/RL residuel 3 DDL",
            FIGURES / "step_21_fuzzy_residual_q_learning_3dof.png",
            [
                "La table Q contient 729 x 27 valeurs.",
                "Le residu est volontairement limite pour rester proche de la politique floue stable.",
                "La precision finale est legerement meilleure que le flou seul sur la cible de reference.",
            ],
        )
        _add_image_page(
            pdf,
            "Step 22 - Generalisation multi-cibles 3 DDL",
            FIGURES / "step_22_fuzzy_residual_generalization_3dof.png",
            [
                "Les cinq cibles spatiales sont atteintes par les trois methodes.",
                "Les gains de temps du residu RL restent modestes.",
                "Cette etape montre surtout la faisabilite et le cout de la montee 2 DDL vers 3 DDL.",
            ],
        )
        _add_text_page(
            pdf,
            "Conclusion",
            [
                "Le passage au bras spatial 3 DDL est valide : la cinematique inverse, la Jacobienne, la dynamique a couples, les environnements et les simulations produisent des resultats coherents.",
                "La montee en dimension est deja visible : 81 regles et 9 actions en 2 DDL deviennent 729 regles et 27 actions en 3 DDL.",
                "En gardant la meme approche tabulaire jusqu'a 6 DDL, on obtiendrait 531441 regles, 729 actions et 387420489 valeurs Q. Cette taille reste possible a stocker seulement avec prudence, mais elle devient lourde a apprendre et a evaluer a chaque pas.",
                "La suite recommandee est donc de conserver le flou comme structure interpretable, mais de remplacer la table dense par une version creuse, factorisee ou par une approximation de fonction avant de passer au 6 DDL.",
            ],
        )

    return output_path


def main() -> int:
    output_path = build_report()
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
