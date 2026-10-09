from __future__ import annotations

import re
import sys
import textwrap
import unicodedata
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, EXPECTED_SHAPE, locate_csv
from b1_target_emse import TARGET_NAME, build_q49_code_to_days, build_target


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "b3_figures"
DATA_DICTIONARY_PATH = BASE_DIR / "b2_data_dictionary.csv"
B2_MISSING_PATH = BASE_DIR / "b2_missing_summary.csv"

DIAGNOSIS_PATH = BASE_DIR / "b3_missing_diagnosis.csv"
TARGET_PATH = BASE_DIR / "b3_q4q5_missing_by_target.csv"
PSU_PATH = BASE_DIR / "b3_q4q5_missing_by_psu.csv"
STRATUM_PATH = BASE_DIR / "b3_q4q5_missing_by_stratum.csv"
SEX_PATH = BASE_DIR / "b3_q4q5_missing_by_sex.csv"
AGE_PATH = BASE_DIR / "b3_q4q5_missing_by_age.csv"
GRADE_PATH = BASE_DIR / "b3_q4q5_missing_by_grade.csv"
EVIDENCE_PATH = BASE_DIR / "b3_q4q5_evidence_summary.csv"
DEMOGRAPHIC_ID_PATH = BASE_DIR / "b3_demographic_identification.csv"
BRANCHING_PATH = BASE_DIR / "b3_branching_candidates.csv"
COOCCURRENCE_PATH = BASE_DIR / "b3_missing_cooccurrence.csv"

MISSING_DIAGNOSIS_POLICY = {
    "q4_q5_initial_hypothesis": "MAR",
    "q4_q5_reserve": "MNAR_POSIBLE_NO_CONFIRMABLE",
    "q4_q5_not_structural": True,
    "q4_q5_not_mcar": True,
    "qn40_mechanism": "NO_IDENTIFICABLE",
    "qn40_feature_status": "EXCLUIR_VACIO_INFORMATIVO",
}

PRIORITY_QN_COLS = [
    "qn31",
    "qn18",
    "qn19",
    "qnc1g",
    "qnbcanyg",
    "qn48",
    "qn47",
    "qn45",
    "qn75",
    "qn28",
    "qn71",
    "qn72",
    "qn37",
    "qn36",
]

# Umbral descriptivo, no inferencial, para llamar "material" a un rango.
MATERIAL_RANGE_PP = 5.0
COOCCURRENCE_MIN_SHARED_PCT = 80.0

OPERATION_FLAGS = {
    "imputation": False,
    "row_drop_for_missing": False,
    "feature_drop": False,
    "split": False,
    "model_transformation": False,
    "scaling": False,
    "feature_encoding": False,
    "discretization": False,
    "feature_engineering": False,
    "smote": False,
    "feature_selection": False,
    "pca": False,
    "model_training": False,
}

BLUE = "#2F6690"
ORANGE = "#F28E2B"
TEAL = "#2A9D8F"
LIGHT = "#DCEAF4"


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in text if not unicodedata.combining(char)).casefold()


def is_explicit_non_applicable_label(label: str) -> bool:
    normalized = normalize_text(label).strip()
    prefixes = (
        "nunca ",
        "no tuve ",
        "no sufri ",
        "no intente ",
        "no tome ",
        "no fume ",
        "no probe ",
        "no maneje ",
        "no viaje ",
        "yo no tomo ",
    )
    return normalized.startswith(prefixes)


def category_missing_table(
    df: pd.DataFrame,
    missing_mask: pd.Series,
    code_column: str,
    text_column: str,
) -> pd.DataFrame:
    table = (
        pd.DataFrame(
            {
                "code": df[code_column],
                "label": df[text_column],
                "missing_q4_q5": missing_mask,
            }
        )
        .groupby(["code", "label"], dropna=False, observed=True)["missing_q4_q5"]
        .agg(n="size", missing_n="sum")
        .reset_index()
        .sort_values("code", na_position="last")
        .reset_index(drop=True)
    )
    table.insert(0, "column", code_column)
    table["missing_n"] = table["missing_n"].astype(int)
    table["observed_n"] = table["n"] - table["missing_n"]
    table["missing_pct"] = 100 * table["missing_n"] / table["n"]
    return table


def grouped_rate_summary(
    table: pd.DataFrame,
    source: str,
    interpretation: str,
    observed_codes_only: bool = False,
) -> dict[str, object]:
    rates = table.loc[table["code"].notna(), "missing_pct"] if observed_codes_only else table["missing_pct"]
    return {
        "evidence_source": source,
        "groups": int(len(rates)),
        "min_missing_pct": float(rates.min()),
        "max_missing_pct": float(rates.max()),
        "absolute_range_pp": float(rates.max() - rates.min()),
        "interpretation": interpretation,
    }


def build_branching_table(
    df: pd.DataFrame,
    high_missing_qn_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dependent_col in high_missing_qn_cols:
        dependent_missing = df[dependent_col].isna()
        dependent_text_col = f"texto_{dependent_col}"
        dependent_labels = (
            sorted(df[dependent_text_col].dropna().astype(str).unique().tolist())
            if dependent_text_col in df
            else []
        )
        base_row = {
            "dependent_col": dependent_col,
            "dependent_text_col": dependent_text_col if dependent_text_col in df else "",
            "dependent_observed_labels": " | ".join(dependent_labels),
            "filter_col": "",
            "filter_non_applicable_label": "",
            "n_filter_non_applicable": 0,
            "n_missing_dependent": int(dependent_missing.sum()),
            "exact_mask_match": False,
            "mismatch_n": int(dependent_missing.sum()),
            "classification": "NO_IDENTIFICABLE",
            "evidence": "No se identificó antecedente qN directo y verificable por nombre.",
        }

        if dependent_col == "qn40":
            base_row["evidence"] = (
                "Criterio metodológico: 100% faltante; no se intenta atribuir causa ni branching."
            )
            rows.append(base_row)
            continue

        match = re.fullmatch(r"qn(\d+)", dependent_col)
        if not match:
            rows.append(base_row)
            continue

        filter_col = f"q{match.group(1)}"
        text_col = f"texto_{filter_col}"
        if filter_col not in df or text_col not in df:
            rows.append(base_row)
            continue

        labels = [
            str(label)
            for label in df[text_col].dropna().unique()
            if is_explicit_non_applicable_label(str(label))
        ]
        if not labels:
            base_row.update(
                {
                    "filter_col": filter_col,
                    "evidence": (
                        f"Se inspeccionó {filter_col}<->{text_col}; no apareció "
                        "etiqueta inequívoca de no-aplicación con el criterio conservador."
                    ),
                }
            )
            rows.append(base_row)
            continue

        non_applicable = df[text_col].isin(labels)
        exact_match = bool(non_applicable.equals(dependent_missing))
        mismatch_n = int((non_applicable != dependent_missing).sum())
        all_non_app_are_missing = bool(dependent_missing.loc[non_applicable].all())
        if exact_match:
            classification = "ESTRUCTURAL_DEMOSTRADO"
        elif all_non_app_are_missing:
            classification = "NO_IDENTIFICABLE_POSIBLE_BRANCHING"
        else:
            classification = "NO_IDENTIFICABLE"

        base_row.update(
            {
                "filter_col": filter_col,
                "filter_non_applicable_label": " | ".join(labels),
                "n_filter_non_applicable": int(non_applicable.sum()),
                "exact_mask_match": exact_match,
                "mismatch_n": mismatch_n,
                "classification": classification,
                "evidence": (
                    f"Máscara no-aplicación de {text_col}: n={int(non_applicable.sum())}; "
                    f"missing {dependent_col}: n={int(dependent_missing.sum())}; "
                    f"mismatches={mismatch_n}; todos los no-aplicables faltan="
                    f"{all_non_app_are_missing}."
                ),
            }
        )
        rows.append(base_row)

    return pd.DataFrame(rows)


def build_cooccurrence(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column_a, column_b in combinations(columns, 2):
        missing_a = df[column_a].isna()
        missing_b = df[column_b].isna()
        n_a = int(missing_a.sum())
        n_b = int(missing_b.sum())
        both = int((missing_a & missing_b).sum())
        shared_a = 100 * both / n_a if n_a else 0.0
        shared_b = 100 * both / n_b if n_b else 0.0
        union = int((missing_a | missing_b).sum())
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "missing_a": n_a,
                "missing_b": n_b,
                "both_missing": both,
                "missing_a_shared_pct": shared_a,
                "missing_b_shared_pct": shared_b,
                "min_shared_pct": min(shared_a, shared_b),
                "jaccard_missing": both / union if union else 0.0,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["min_shared_pct", "both_missing"], ascending=[False, False]
    )
    q4_q5_pair = table["column_a"].eq("q4") & table["column_b"].eq("q5")
    selected = table.loc[
        table["min_shared_pct"].ge(COOCCURRENCE_MIN_SHARED_PCT) | q4_q5_pair
    ]
    return selected.head(50).reset_index(drop=True)


def build_diagnosis(
    missing_summary: pd.DataFrame,
    q4_q5_result: str,
    q4_q5_review: bool,
    branching: pd.DataFrame,
) -> pd.DataFrame:
    branching_by_column = branching.set_index("dependent_col").to_dict("index")
    rows: list[dict[str, object]] = []
    for item in missing_summary.itertuples(index=False):
        column = str(item.column)
        missing_n = int(item.missing_n)
        missing_pct = float(item.missing_pct)
        feature_status = ""

        if column in {"q4", "q5"}:
            mechanism = q4_q5_result
            confidence = "MEDIA"
            evidence = (
                "Co-ausencia q4/q5 exacta; tasas comparadas por target, PSU, stratum, "
                "sexo, edad y grado en b3_q4q5_evidence_summary.csv."
            )
            review = q4_q5_review
            notes = "Reserva: MNAR_POSIBLE_NO_CONFIRMABLE; no estructural; no MCAR afirmado."
        elif column == "qn40":
            mechanism = "NO_IDENTIFICABLE"
            confidence = "ALTA_EN_LIMITACION"
            evidence = "missing_n=55551, missing_pct=100%; ausencia total no identifica causa."
            review = False
            notes = (
                "100% faltante permite decidir inutilidad como feature, no el mecanismo."
            )
            feature_status = "EXCLUIR_VACIO_INFORMATIVO"
        elif missing_n == 0:
            mechanism = "SIN_FALTANTES"
            confidence = "ALTA"
            evidence = "missing_n=0 en el universo B3 de 55551 filas."
            review = False
            notes = "No corresponde diagnosticar un mecanismo de ausencia."
        elif column in branching_by_column:
            candidate = branching_by_column[column]
            mechanism = str(candidate["classification"])
            confidence = "ALTA" if mechanism == "ESTRUCTURAL_DEMOSTRADO" else "MEDIA"
            evidence = str(candidate["evidence"])
            review = mechanism != "ESTRUCTURAL_DEMOSTRADO"
            notes = (
                "ESTRUCTURAL_DEMOSTRADO exige igualdad exacta de máscaras; "
                "la co-ausencia por sí sola no prueba mecanismo."
            )
        else:
            mechanism = "NO_IDENTIFICABLE"
            confidence = "BAJA"
            evidence = (
                f"missing_n={missing_n}, missing_pct={missing_pct:.4f}%; "
                "sin condición previa exacta y semánticamente inequívoca demostrada."
            )
            review = True
            notes = "Diagnóstico conservador: el porcentaje aislado no identifica mecanismo."

        rows.append(
            {
                "column": column,
                "missing_n": missing_n,
                "missing_pct": missing_pct,
                "proposed_mechanism": mechanism,
                "confidence": confidence,
                "evidence": evidence,
                "requires_review": review,
                "feature_status": feature_status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_figures(
    missing_summary: pd.DataFrame,
    df_target: pd.DataFrame,
    target_table: pd.DataFrame,
    psu_table: pd.DataFrame,
    stratum_table: pd.DataFrame,
    demographic_tables: dict[str, pd.DataFrame],
    priority_heatmap_cols: list[str],
) -> dict[str, str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    descriptions: dict[str, str] = {}

    top = missing_summary.head(20).sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["column"], top["missing_pct"], color=BLUE)
    ax.set(title="Variables con mayor porcentaje de faltantes", xlabel="Faltantes (%)")
    ax.set_xlim(0, 105)
    descriptions["01_missing_general_top.png"] = "Top 20 de faltantes sustantivos."
    save_figure(fig, "01_missing_general_top.png")

    masks = df_target[priority_heatmap_cols].isna().astype("int64")
    intersection = masks.T.dot(masks).to_numpy(dtype=float)
    counts = np.diag(intersection)
    union = counts[:, None] + counts[None, :] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union != 0,
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(jaccard, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(priority_heatmap_cols)), priority_heatmap_cols, rotation=60, ha="right")
    ax.set_yticks(range(len(priority_heatmap_cols)), priority_heatmap_cols)
    ax.set_title("Similitud Jaccard entre máscaras de faltantes prioritarias")
    fig.colorbar(image, ax=ax, label="Jaccard")
    descriptions["02_missing_priority_heatmap.png"] = (
        "Heatmap acotado de similitud entre máscaras de faltantes."
    )
    save_figure(fig, "02_missing_priority_heatmap.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(target_table["target"].astype(str), target_table["missing_pct"], color=[BLUE, ORANGE])
    for bar, value in zip(bars, target_table["missing_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}%", ha="center")
    ax.set(title="Ausencia conjunta q4/q5 por target", xlabel=TARGET_NAME, ylabel="Faltantes (%)")
    ax.set_ylim(0, max(target_table["missing_pct"]) * 1.18)
    descriptions["03_q4q5_missing_by_target.png"] = "Tasa de co-ausencia q4/q5 por clase."
    save_figure(fig, "03_q4q5_missing_by_target.png")

    psu_sorted = psu_table.sort_values("missing_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(psu_sorted)), psu_sorted["missing_pct"], color=TEAL, width=0.9)
    ax.set(title="Ausencia conjunta q4/q5 por PSU (ordenada)", xlabel="PSU ordenada por tasa", ylabel="Faltantes (%)")
    ax.set_ylim(0, 105)
    descriptions["04_q4q5_missing_by_psu.png"] = "Tasas por PSU, ordenadas."
    save_figure(fig, "04_q4q5_missing_by_psu.png")

    stratum_sorted = stratum_table.sort_values("missing_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(stratum_sorted)), stratum_sorted["missing_pct"], color=BLUE, linewidth=1.5)
    ax.fill_between(np.arange(len(stratum_sorted)), stratum_sorted["missing_pct"], color=LIGHT)
    ax.set(title="Ausencia conjunta q4/q5 por stratum (ordenada)", xlabel="Estrato ordenado por tasa", ylabel="Faltantes (%)")
    ax.set_ylim(0, 105)
    descriptions["05_q4q5_missing_by_stratum.png"] = "Tasas por estrato, ordenadas."
    save_figure(fig, "05_q4q5_missing_by_stratum.png")

    for number, (name, table) in enumerate(demographic_tables.items(), start=6):
        labels = []
        for row in table.itertuples(index=False):
            code = "NA" if pd.isna(row.code) else f"{float(row.code):g}"
            labels.append(textwrap.fill(f"{code}: {row.label}", width=24))
        fig_width = max(7, len(table) * 1.25)
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))
        bars = ax.bar(np.arange(len(table)), table["missing_pct"], color=ORANGE)
        ax.set_xticks(np.arange(len(table)), labels, rotation=25, ha="right")
        ax.set(title=f"Ausencia conjunta q4/q5 por {name}", ylabel="Faltantes (%)")
        ax.set_ylim(0, max(table["missing_pct"].max() * 1.18, 10))
        for bar, value in zip(bars, table["missing_pct"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", fontsize=8)
        filename = f"{number:02d}_q4q5_missing_by_{name}.png"
        descriptions[filename] = f"Tasas de co-ausencia q4/q5 por {name}."
        save_figure(fig, filename)

    return descriptions


def print_group_summary(label: str, table: pd.DataFrame, top_n: int = 5) -> None:
    print(f"\n{label}: grupos={len(table)}; min={table['missing_pct'].min():.4f}%; "
          f"mediana={table['missing_pct'].median():.4f}%; max={table['missing_pct'].max():.4f}%")
    print("Mayor tasa:")
    print(table.nlargest(top_n, "missing_pct").to_string(index=False))
    print("Menor tasa:")
    print(table.nsmallest(top_n, "missing_pct").to_string(index=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    csv_path = locate_csv()
    with warnings.catch_warnings(record=True) as read_warnings:
        warnings.simplefilter("always")
        df = pd.read_csv(
            csv_path,
            sep=CSV_SEPARATOR,
            encoding=CSV_ENCODING,
            low_memory=False,
        )

    code_to_days, mapping_consistent = build_q49_code_to_days(df)
    if not mapping_consistent:
        raise AssertionError("Mapping q49/texto_q49 inconsistente; B3 detenido.")
    target = build_target(df, code_to_days)
    target_mask = target.notna()
    df_target = df.loc[target_mask].copy()
    target_observed = target.loc[target_mask].astype("int8")

    data_dictionary = pd.read_csv(DATA_DICTIONARY_PATH, encoding="utf-8-sig")
    missing_summary = pd.read_csv(B2_MISSING_PATH, encoding="utf-8-sig")
    q4_before = df_target["q4"].copy(deep=True)
    q5_before = df_target["q5"].copy(deep=True)

    q4_missing = df_target["q4"].isna()
    q5_missing = df_target["q5"].isna()
    missing_q4_q5 = q4_missing & q5_missing

    target_table = (
        pd.DataFrame({"target": target_observed, "missing_q4_q5": missing_q4_q5})
        .groupby("target", observed=True)["missing_q4_q5"]
        .agg(n="size", missing_n="sum")
        .reset_index()
    )
    target_table["missing_n"] = target_table["missing_n"].astype(int)
    target_table["observed_n"] = target_table["n"] - target_table["missing_n"]
    target_table["missing_pct"] = 100 * target_table["missing_n"] / target_table["n"]
    target_table.to_csv(TARGET_PATH, index=False, encoding="utf-8-sig")

    target_rates = target_table.set_index("target")["missing_pct"]
    target_difference_pp = abs(float(target_rates.loc[0] - target_rates.loc[1]))
    target_rate_ratio = float(target_rates.loc[0] / target_rates.loc[1])
    target_contingency = pd.crosstab(
        target_observed,
        missing_q4_q5.map({False: "observado_q4q5", True: "missing_q4q5"}),
    )

    psu_table = (
        pd.DataFrame(
            {
                "psu": df_target["psu"],
                "missing_q4_q5": missing_q4_q5,
                "target": target_observed,
            }
        )
        .groupby("psu", dropna=False, observed=True)
        .agg(n=("missing_q4_q5", "size"), missing_n=("missing_q4_q5", "sum"), target_positive_rate=("target", "mean"))
        .reset_index()
    )
    psu_table["missing_n"] = psu_table["missing_n"].astype(int)
    psu_table["missing_pct"] = 100 * psu_table["missing_n"] / psu_table["n"]
    psu_table.to_csv(PSU_PATH, index=False, encoding="utf-8-sig")

    stratum_table = (
        pd.DataFrame({"stratum": df_target["stratum"], "missing_q4_q5": missing_q4_q5})
        .groupby("stratum", dropna=False, observed=True)["missing_q4_q5"]
        .agg(n="size", missing_n="sum")
        .reset_index()
    )
    stratum_table["missing_n"] = stratum_table["missing_n"].astype(int)
    stratum_table["observed_n"] = stratum_table["n"] - stratum_table["missing_n"]
    stratum_table["missing_pct"] = 100 * stratum_table["missing_n"] / stratum_table["n"]
    stratum_table.to_csv(STRATUM_PATH, index=False, encoding="utf-8-sig")

    demographic_identification = pd.DataFrame(
        [
            {
                "concept": "edad",
                "column": "q1",
                "paired_text_column": "texto_q1",
                "reason": "Las etiquetas observadas expresan edades en años y rangos de edad.",
                "confidence": "ALTA",
            },
            {
                "concept": "sexo",
                "column": "q2",
                "paired_text_column": "texto_q2",
                "reason": "Las etiquetas observadas son Masculino y Femenino.",
                "confidence": "ALTA",
            },
            {
                "concept": "grado",
                "column": "q3",
                "paired_text_column": "texto_q3",
                "reason": "Las etiquetas observadas expresan grados/años escolares ordenados.",
                "confidence": "ALTA",
            },
        ]
    )
    demographic_identification.to_csv(DEMOGRAPHIC_ID_PATH, index=False, encoding="utf-8-sig")

    demographic_specs = {
        "sexo": ("q2", "texto_q2", SEX_PATH),
        "edad": ("q1", "texto_q1", AGE_PATH),
        "grado": ("q3", "texto_q3", GRADE_PATH),
    }
    demographic_tables: dict[str, pd.DataFrame] = {}
    for concept, (column, text_column, path) in demographic_specs.items():
        table = category_missing_table(df_target, missing_q4_q5, column, text_column)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        demographic_tables[concept] = table

    evidence_rows = [
        {
            "evidence_source": "target",
            "groups": int(len(target_table)),
            "min_missing_pct": float(target_table["missing_pct"].min()),
            "max_missing_pct": float(target_table["missing_pct"].max()),
            "absolute_range_pp": target_difference_pp,
            "interpretation": "La tasa cambia entre clases del target observado; asociación descriptiva, no causal.",
        },
        {
            "evidence_source": "PSU",
            "groups": int(len(psu_table)),
            "min_missing_pct": float(psu_table["missing_pct"].min()),
            "max_missing_pct": float(psu_table["missing_pct"].max()),
            "absolute_range_pp": float(psu_table["missing_pct"].max() - psu_table["missing_pct"].min()),
            "interpretation": "Heterogeneidad por agrupamiento observado; no atribuye causalidad a la escuela.",
        },
        {
            "evidence_source": "stratum",
            "groups": int(len(stratum_table)),
            "min_missing_pct": float(stratum_table["missing_pct"].min()),
            "max_missing_pct": float(stratum_table["missing_pct"].max()),
            "absolute_range_pp": float(stratum_table["missing_pct"].max() - stratum_table["missing_pct"].min()),
            "interpretation": "Heterogeneidad descriptiva entre estratos observados.",
        },
    ]
    for concept, table in demographic_tables.items():
        evidence_rows.append(
            grouped_rate_summary(
                table,
                concept,
                f"Variación descriptiva por categorías observadas de {concept}.",
                observed_codes_only=True,
            )
        )
    evidence_summary = pd.DataFrame(evidence_rows)
    evidence_summary.to_csv(EVIDENCE_PATH, index=False, encoding="utf-8-sig")

    material_sources = evidence_summary.loc[
        evidence_summary["absolute_range_pp"].ge(MATERIAL_RANGE_PP), "evidence_source"
    ].tolist()
    if len(material_sources) >= 2:
        q4_q5_result = "MAR_HIPOTESIS_RESPALDADA"
    elif material_sources:
        q4_q5_result = "MAR_HIPOTESIS_DEBIL"
    else:
        q4_q5_result = "NO_IDENTIFICABLE"
    q4_q5_requires_review = q4_q5_result == "NO_IDENTIFICABLE"

    high_missing_qn_cols = sorted(
        set(PRIORITY_QN_COLS)
        | set(
            missing_summary.loc[
                missing_summary["column"].astype(str).str.startswith("qn")
                & missing_summary["missing_pct"].ge(40),
                "column",
            ].astype(str)
        )
    )
    branching_table = build_branching_table(df_target, high_missing_qn_cols)
    branching_table.to_csv(BRANCHING_PATH, index=False, encoding="utf-8-sig")

    diagnosis = build_diagnosis(
        missing_summary,
        q4_q5_result,
        q4_q5_requires_review,
        branching_table,
    )
    diagnosis.to_csv(DIAGNOSIS_PATH, index=False, encoding="utf-8-sig")

    cooccurrence_cols = ["q4", "q5"] + missing_summary.loc[
        missing_summary["missing_pct"].ge(40), "column"
    ].astype(str).tolist()
    cooccurrence_cols = list(dict.fromkeys(cooccurrence_cols))
    cooccurrence = build_cooccurrence(df_target, cooccurrence_cols)
    cooccurrence.to_csv(COOCCURRENCE_PATH, index=False, encoding="utf-8-sig")

    priority_heatmap_cols = list(dict.fromkeys(["q4", "q5"] + high_missing_qn_cols))
    figure_descriptions = create_figures(
        missing_summary,
        df_target,
        target_table,
        psu_table,
        stratum_table,
        demographic_tables,
        priority_heatmap_cols,
    )

    structural_rows = branching_table.loc[
        branching_table["classification"].eq("ESTRUCTURAL_DEMOSTRADO")
    ]
    forbidden_mechanisms = set(diagnosis["proposed_mechanism"])
    expected_artifacts = [
        DIAGNOSIS_PATH,
        TARGET_PATH,
        PSU_PATH,
        STRATUM_PATH,
        SEX_PATH,
        AGE_PATH,
        GRADE_PATH,
        EVIDENCE_PATH,
        DEMOGRAPHIC_ID_PATH,
        BRANCHING_PATH,
        COOCCURRENCE_PATH,
    ] + [FIGURE_DIR / filename for filename in figure_descriptions]

    checks = {
        "csv_crudo_56981x309": df.shape == EXPECTED_SHAPE,
        "universo_55551": df_target.shape == (55_551, 309),
        "q4_sin_modificacion": df_target["q4"].equals(q4_before),
        "q5_sin_modificacion": df_target["q5"].equals(q5_before),
        "mascara_missing_q4_igual_q5": q4_missing.equals(q5_missing),
        "missing_q4_q5_20209": int(missing_q4_q5.sum()) == 20_209,
        "qn40_100pct_missing": bool(df_target["qn40"].isna().all()),
        "sin_imputacion": not OPERATION_FLAGS["imputation"] and df_target.isna().equals(df.loc[target_mask].isna()),
        "sin_drop_filas_features": (
            not OPERATION_FLAGS["row_drop_for_missing"]
            and not OPERATION_FLAGS["feature_drop"]
            and df_target.shape == (55_551, 309)
            and list(df_target.columns) == list(df.columns)
        ),
        "sin_split": not OPERATION_FLAGS["split"],
        "sin_transformacion_modelado": not any(
            OPERATION_FLAGS[name]
            for name in (
                "model_transformation",
                "scaling",
                "feature_encoding",
                "discretization",
                "feature_engineering",
                "smote",
                "feature_selection",
                "pca",
                "model_training",
            )
        ),
        "estructural_solo_con_match_exacto": bool(
            structural_rows.empty
            or (
                structural_rows["exact_mask_match"].all()
                and structural_rows["filter_col"].ne("").all()
                and structural_rows["filter_non_applicable_label"].ne("").all()
            )
        ),
        "sin_MCAR_CONFIRMADO": "MCAR_CONFIRMADO" not in forbidden_mechanisms,
        "sin_MNAR_CONFIRMADO": "MNAR_CONFIRMADO" not in forbidden_mechanisms,
        "toda_clasificacion_con_evidencia": diagnosis["evidence"].fillna("").str.strip().ne("").all(),
        "contradiccion_MAR_marcada_revision": (
            q4_q5_result != "NO_IDENTIFICABLE" or q4_q5_requires_review
        ),
        "qn40_estado_feature_correcto": diagnosis.loc[
            diagnosis["column"].eq("qn40"), "feature_status"
        ].eq("EXCLUIR_VACIO_INFORMATIVO").all(),
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "b2_data_dictionary_309_sin_modificar": (
            len(data_dictionary) == 309
            and data_dictionary["column"].nunique() == 309
            and set(data_dictionary["column"]) == set(df.columns)
        ),
    }

    print("=== RECONSTRUCCION B3 ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Shape crudo: {df.shape}")
    print(f"Shape universo target observado: {df_target.shape}")
    print(
        "Warnings de lectura: "
        + (
            "; ".join(f"{item.category.__name__}: {item.message}" for item in read_warnings)
            if read_warnings
            else "ninguno"
        )
    )
    print(f"Missing q4: {int(q4_missing.sum())} ({100 * q4_missing.mean():.4f}%)")
    print(f"Missing q5: {int(q5_missing.sum())} ({100 * q5_missing.mean():.4f}%)")
    print(f"Ambas: {int(missing_q4_q5.sum())}; solo q4: {int((q4_missing & ~q5_missing).sum())}; solo q5: {int((~q4_missing & q5_missing).sum())}")

    print("\n=== q4/q5 POR TARGET ===")
    print(target_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Tabla target x missing_q4_q5:")
    print(target_contingency.to_string())
    print(f"Diferencia absoluta: {target_difference_pp:.4f} pp")
    print(f"Razón de tasas target 0 / target 1: {target_rate_ratio:.4f}")

    print("\n=== q4/q5 POR PSU ===")
    psu_zero = int(np.isclose(psu_table["missing_pct"], 0).sum())
    psu_full = int(np.isclose(psu_table["missing_pct"], 100).sum())
    psu_partial = int(len(psu_table) - psu_zero - psu_full)
    print(f"PSU={len(psu_table)}; min={psu_table['missing_pct'].min():.4f}%; mediana={psu_table['missing_pct'].median():.4f}%; max={psu_table['missing_pct'].max():.4f}%")
    print(f"PSU 0%={psu_zero}; PSU 100%={psu_full}; PSU entre 0% y 100%={psu_partial}")
    print_group_summary("PSU", psu_table)

    print("\n=== q4/q5 POR STRATUM ===")
    stratum_zero = int(np.isclose(stratum_table["missing_pct"], 0).sum())
    stratum_full = int(np.isclose(stratum_table["missing_pct"], 100).sum())
    stratum_partial = int(len(stratum_table) - stratum_zero - stratum_full)
    print(f"Strata={len(stratum_table)}; 0%={stratum_zero}; 100%={stratum_full}; parciales={stratum_partial}")
    print_group_summary("stratum", stratum_table)

    print("\n=== IDENTIFICACION SEXO / EDAD / GRADO ===")
    for row in demographic_identification.itertuples(index=False):
        print(f"{row.concept}: columna={row.column}; par={row.paired_text_column}; confianza={row.confidence}")
        print(f"Razón: {row.reason}")
        mapping = df_target[[row.column, row.paired_text_column]].drop_duplicates().sort_values(row.column, na_position="last")
        print(mapping.to_string(index=False))
        print("Tasas missing q4/q5:")
        print(demographic_tables[row.concept].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n=== EVIDENCIA q4/q5 Y CRITERIO ADOPTADO ===")
    print(evidence_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Umbral descriptivo de materialidad: {MATERIAL_RANGE_PP:.1f} pp")
    print(f"Fuentes con rango material: {material_sources}")
    print(f"DECISION_INICIAL = {MISSING_DIAGNOSIS_POLICY['q4_q5_initial_hypothesis']}")
    print(f"RESULTADO_EMPIRICO = {q4_q5_result}")
    print(f"REQUIERE_REVISION = {'SI' if q4_q5_requires_review else 'NO'}")
    print(f"RESERVA = {MISSING_DIAGNOSIS_POLICY['q4_q5_reserve']}")

    print("\n=== BRANCHING: CANDIDATOS PRIORITARIOS / >=40% ===")
    print(branching_table.to_string(index=False))
    print(f"ESTRUCTURAL_DEMOSTRADO: {structural_rows['dependent_col'].tolist()}")
    possible_branching = branching_table.loc[
        branching_table["classification"].eq("NO_IDENTIFICABLE_POSIBLE_BRANCHING"),
        "dependent_col",
    ].tolist()
    print(f"NO_IDENTIFICABLE_POSIBLE_BRANCHING: {possible_branching}")

    print("\n=== DIAGNOSTICO GENERAL ===")
    print(f"Filas de diagnóstico: {len(diagnosis)}")
    print(diagnosis["proposed_mechanism"].value_counts().to_string())
    qn40_row = diagnosis.loc[diagnosis["column"].eq("qn40")].iloc[0]
    print("\nqn40:")
    print(f"missing_pct={qn40_row['missing_pct']:.4f}%")
    print(f"mechanism={qn40_row['proposed_mechanism']}")
    print(f"feature_status={qn40_row['feature_status']}")
    print(f"explicación={qn40_row['notes']}")

    print("\n=== CO-AUSENCIA DESTACADA ===")
    print(f"Umbral de presentación: ambos porcentajes compartidos >= {COOCCURRENCE_MIN_SHARED_PCT:.1f}%")
    print(f"Pares guardados: {len(cooccurrence)}")
    print(cooccurrence.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Nota: co-ausencia no implica mecanismo común.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)
    print("\n=== GRAFICOS ===")
    for filename, description in figure_descriptions.items():
        print(f"{filename}: {description}")

    print("\n=== CHECKS B3 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B3 detenido; checks fallidos: {failed}")
    print("B3 finalizado.")


if __name__ == "__main__":
    main()
