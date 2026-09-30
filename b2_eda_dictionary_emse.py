from __future__ import annotations

import json
import re
import sys
import textwrap
import unicodedata
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, EXPECTED_SHAPE, locate_csv
from b1_target_emse import TARGET_NAME, build_q49_code_to_days, build_target


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "b2_figures"
DATA_DICTIONARY_PATH = BASE_DIR / "b2_data_dictionary.csv"
NUMERIC_SUMMARY_PATH = BASE_DIR / "b2_numeric_summary.csv"
CATEGORICAL_SUMMARY_PATH = BASE_DIR / "b2_categorical_summary.csv"
MISSING_SUMMARY_PATH = BASE_DIR / "b2_missing_summary.csv"
PSU_PROFILE_PATH = BASE_DIR / "b2_psu_profile.csv"
NUMERIC_BY_TARGET_PATH = BASE_DIR / "b2_numeric_by_target.csv"
CATEGORICAL_BY_TARGET_PATH = BASE_DIR / "b2_categorical_by_target.csv"

HARD_LEAKAGE_COLS = [
    "q49",
    "qn49",
    "qnpa5g",
    "qnpa7g",
    "texto_q49",
    "texto_qn49",
    "texto_qnpa5g",
    "texto_qnpa7g",
]
SAME_DOMAIN_REVIEW_COLS = ["q50", "q51"]
METADATA_COLS = ["weight", "stratum", "psu", "sitio", "record"]
CONTINUOUS_COLS = ["q4", "q5"]
REPRESENTATIVE_CATEGORICALS = {
    "q2": "binaria, etiquetas inequívocas y alta cobertura",
    "q6": "escala ordinal de frecuencia y cardinalidad manejable",
    "q10": "conteo ordinal de días y cardinalidad manejable",
    "q50": "revisión obligatoria: mismo dominio, no leakage duro",
    "q51": "revisión obligatoria: mismo dominio, no leakage duro",
}
OPERATION_FLAGS = {
    "split": False,
    "imputation": False,
    "scaling": False,
    "feature_encoding": False,
    "smote": False,
    "pca": False,
    "feature_selection": False,
    "model_training": False,
}

BLUE = "#2F5D7C"
ORANGE = "#C97B36"
INK = "#263238"
GRID = "#D9E0E4"


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.strip().casefold())


def get_layer(column: str) -> str:
    if column in METADATA_COLS:
        return "metadata"
    if re.fullmatch(r"q\d+", column):
        return "q_response"
    if column.startswith("qn"):
        return "qn_derived"
    if re.fullmatch(r"texto_q\d+", column):
        return "response_text"
    if column.startswith("texto_qn"):
        return "derived_text"
    return "other"


def paired_text_column(column: str, columns: pd.Index) -> str:
    if re.fullmatch(r"q\d+", column) or column.startswith("qn"):
        paired = f"texto_{column}"
        return paired if paired in columns else ""
    return ""


def ordered_labels(df: pd.DataFrame, column: str, text_column: str) -> list[str]:
    if not text_column:
        return []
    pairs = (
        df.loc[df[column].notna(), [column, text_column]]
        .dropna(subset=[text_column])
        .drop_duplicates()
        .sort_values(column)
    )
    return pairs[text_column].astype(str).tolist()


def classify_categorical(labels: list[str], n_unique: int) -> tuple[str, str, str]:
    if n_unique == 2 and len(labels) == 2:
        return (
            "binary_categorical",
            "ALTA",
            "dos categorías observadas con etiquetas inequívocas",
        )
    if not labels:
        return (
            "unknown_categorical",
            "A_CONFIRMAR",
            "sin etiquetas textuales suficientes",
        )

    normalized = [normalize_label(label) for label in labels]
    frequency_scale = {
        "nunca",
        "rara vez",
        "algunas veces",
        "casi siempre",
        "siempre",
        "la mayoria del tiempo",
        "casi todos los dias",
        "todos los dias",
    }
    likelihood_scale = {
        "definitivamente no",
        "probablemente no",
        "probablemente si",
        "definitivamente si",
    }
    frequency_hits = [label in frequency_scale for label in normalized]
    if any(frequency_hits):
        if all(frequency_hits):
            return (
                "ordinal_categorical",
                "ALTA",
                "todas las etiquetas pertenecen a una escala de frecuencia ordenada",
            )
        return (
            "unknown_categorical",
            "A_CONFIRMAR",
            "mezcla escala de frecuencia con categorías no ordenables",
        )
    if set(normalized) == likelihood_scale:
        return (
            "ordinal_categorical",
            "ALTA",
            "escala completa de intensidad/probabilidad ordenada",
        )

    numeric_pattern = re.compile(
        r"^(?:\d+|los \d+|menos de (?:un|una|\d+)|mas de (?:un|una|\d+))"
    )
    numeric_hits = [bool(numeric_pattern.match(label)) for label in normalized]
    has_person_unit = any("persona" in label for label in normalized)

    def is_zero_quantity_label(label: str) -> bool:
        if label.startswith("nunca tuve relaciones sexuales"):
            return has_person_unit
        return label.startswith(("no tome alcohol", "no comi ", "no tome gaseosas"))

    if any(numeric_hits):
        compatible = [
            numeric_hit
            or label in {"ninguna", "ninguno"}
            or is_zero_quantity_label(label)
            for label, numeric_hit in zip(normalized, numeric_hits)
        ]
        if all(compatible):
            return (
                "ordinal_categorical",
                "ALTA",
                "todas las etiquetas expresan cantidades, rangos o frecuencia cero",
            )
        return (
            "unknown_categorical",
            "A_CONFIRMAR",
            "mezcla rangos ordenables con categorías fuera de la escala",
        )

    education_hits = [
        any(
            token in label
            for token in ("primaria", "secundario", "terciario", "universitario")
        )
        for label in normalized
    ]
    if any(education_hits):
        if all(education_hits):
            return (
                "ordinal_categorical",
                "ALTA",
                "niveles educativos con progresión explícita",
            )
        return (
            "unknown_categorical",
            "A_CONFIRMAR",
            "niveles educativos mezclados con respuesta no ordenable",
        )

    return (
        "nominal_categorical",
        "MEDIA",
        "etiquetas disponibles sin orden intrínseco defendible",
    )


def compact_domain(
    df: pd.DataFrame, column: str, text_column: str, n_unique: int
) -> str:
    labels = ordered_labels(df, column, text_column)
    if labels:
        return json.dumps(labels, ensure_ascii=False)
    series = df[column]
    values = series.dropna().drop_duplicates()
    if n_unique <= 20:
        return json.dumps(values.astype(str).tolist(), ensure_ascii=False)
    if pd.api.types.is_numeric_dtype(series):
        return f"min={series.min():.6g}; max={series.max():.6g}; n_unique={n_unique}"
    return f"HIGH_CARDINALITY; n_unique={n_unique}"


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        layer = get_layer(column)
        paired = paired_text_column(column, df.columns)
        n_unique = int(df[column].nunique(dropna=True))
        missing_n = int(df[column].isna().sum())
        labels = ordered_labels(df, column, paired)
        notes: list[str] = []

        if column in CONTINUOUS_COLS:
            role = "continuous_feature"
            statistical_type = "continuous"
            confidence = "ALTA"
            basis = "semántica confirmada en contrato B0 y dominio numérico continuo"
            notes.append("altura en metros" if column == "q4" else "peso en kg")
        elif column == "record":
            role = "identifier"
            statistical_type = "identifier"
            confidence = "ALTA"
            basis = "cardinalidad igual al número de filas del universo B2"
        elif column == "sitio" and n_unique == 1:
            role = "constant"
            statistical_type = "constant"
            confidence = "ALTA"
            basis = "un único valor observado"
        elif column in {"weight", "stratum", "psu"}:
            role = "metadata"
            statistical_type = "metadata"
            confidence = "ALTA"
            basis = "metadata muestral confirmada en B0"
        elif layer in {"response_text", "derived_text"}:
            role = "hard_leakage" if column in HARD_LEAKAGE_COLS else "interpretation_only"
            statistical_type = "interpretation_text"
            confidence = "ALTA"
            basis = "etiqueta textual redundante del código emparejado"
            notes.append("no usar simultáneamente como feature junto con su código")
        elif column == "q49":
            role = "target_source"
            statistical_type, confidence, basis = classify_categorical(
                labels, n_unique
            )
            notes.append("fuente directa de target_pa_oms5; excluir de X")
        elif column in HARD_LEAKAGE_COLS:
            role = "hard_leakage"
            statistical_type, confidence, basis = classify_categorical(
                labels, n_unique
            )
            notes.append("proxy determinístico directo de q49; excluir de X")
        elif layer in {"q_response", "qn_derived"}:
            statistical_type, confidence, basis = classify_categorical(
                labels, n_unique
            )
            if n_unique <= 1:
                role = "constant"
                statistical_type = "constant"
                confidence = "ALTA"
                basis = "cero o un valor observado"
            else:
                role = "categorical_feature_candidate"
        else:
            role = "metadata"
            statistical_type = "metadata"
            confidence = "A_CONFIRMAR"
            basis = "columna residual sin regla estructural específica"

        if column in SAME_DOMAIN_REVIEW_COLS:
            notes.append(
                "MISMO_DOMINIO_NO_LEAKAGE_DURO — pendiente de decisión final de features"
            )
        if statistical_type == "unknown_categorical":
            notes.append("A_CONFIRMAR antes del encoding")
        if column in CONTINUOUS_COLS:
            notes.append("PATRON_A_REVISAR_EN_B3: faltantes conjuntos q4/q5")

        model_candidate = role in {
            "continuous_feature",
            "categorical_feature_candidate",
        }
        rows.append(
            {
                "column": column,
                "layer": layer,
                "role": role,
                "paired_text_column": paired,
                "pandas_dtype": str(df[column].dtype),
                "statistical_type": statistical_type,
                "n_unique": n_unique,
                "missing_n": missing_n,
                "missing_pct": 100 * missing_n / len(df),
                "observed_labels": compact_domain(df, column, paired, n_unique),
                "type_confidence": confidence,
                "type_basis": basis,
                "model_candidate": model_candidate,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CONTINUOUS_COLS:
        series = df[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        missing_n = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "missing_n": missing_n,
                "missing_pct": 100 * missing_n / len(df),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "max": float(series.max()),
                "n_unique": int(series.nunique()),
                "skewness": float(series.skew()),
            }
        )
    return pd.DataFrame(rows)


def build_numeric_by_target(
    df: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    rows = []
    for column in CONTINUOUS_COLS:
        for target_value in (0, 1):
            group = df.loc[target.eq(target_value), column]
            rows.append(
                {
                    "column": column,
                    "target": target_value,
                    "group_n": len(group),
                    "observed_n": int(group.count()),
                    "missing_n": int(group.isna().sum()),
                    "mean": float(group.mean()),
                    "median": float(group.median()),
                    "q1": float(group.quantile(0.25)),
                    "q3": float(group.quantile(0.75)),
                    "std": float(group.std()),
                }
            )
    return pd.DataFrame(rows)


def interpretation_series(df: pd.DataFrame, column: str) -> pd.Series:
    paired = paired_text_column(column, df.columns)
    if paired:
        return df[paired].where(df[column].notna())
    return df[column].astype("string")


def build_categorical_summary(
    df: pd.DataFrame, categorical_cols: list[str]
) -> pd.DataFrame:
    rows = []
    for column in categorical_cols:
        labels = interpretation_series(df, column)
        counts = labels.value_counts(dropna=True)
        observed_n = int(counts.sum())
        mode_label = str(counts.index[0]) if observed_n else ""
        mode_frequency = int(counts.iloc[0]) if observed_n else 0
        missing_n = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "cardinality": int(df[column].nunique(dropna=True)),
                "missing_n": missing_n,
                "missing_pct": 100 * missing_n / len(df),
                "mode_category": mode_label,
                "mode_frequency": mode_frequency,
                "mode_pct_observed": (
                    100 * mode_frequency / observed_n if observed_n else float("nan")
                ),
                "observed_labels": json.dumps(
                    counts.index.astype(str).tolist(), ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def category_target_table(
    df: pd.DataFrame, target: pd.Series, column: str
) -> pd.DataFrame:
    paired = paired_text_column(column, df.columns)
    labels = df[paired] if paired else df[column].astype("string")
    table = pd.DataFrame(
        {
            "column": column,
            "category_code": df[column],
            "category_label": labels,
            "target": target,
        }
    ).dropna(subset=["category_code"])
    grouped = (
        table.groupby(
            ["column", "category_code", "category_label", "target"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    grouped["category_total"] = grouped.groupby(
        ["column", "category_code", "category_label"], observed=True
    )["n"].transform("sum")
    grouped["target_pct_within_category"] = (
        100 * grouped["n"] / grouped["category_total"]
    )
    return grouped.sort_values(["category_code", "target"])


def save_figure(
    fig: plt.Figure,
    filename: str,
    description: str,
    figure_descriptions: dict[str, str],
) -> None:
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figure_descriptions[filename] = description


def create_figures(
    df: pd.DataFrame,
    target: pd.Series,
    categorical_tables: dict[str, pd.DataFrame],
) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        }
    )
    descriptions: dict[str, str] = {}
    subtitle = f"Universo con target observado; no ponderado; n={len(df):,}"

    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["No cumple (0)", "Cumple (1)"],
        counts.values,
        color=[BLUE, ORANGE],
        edgecolor=INK,
    )
    bars[1].set_hatch("///")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{100 * value / len(target):.1f}%",
            ha="center",
            va="bottom",
        )
    ax.set(title=f"Distribución de {TARGET_NAME}\n{subtitle}", ylabel="Casos")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "01_target_barras.png", "Conteos y porcentajes del target.", descriptions)

    units = {"q4": "Altura (m)", "q5": "Peso (kg)"}
    for number, column in enumerate(CONTINUOUS_COLS, start=2):
        observed = df[column].dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(observed, bins=30, color=BLUE, edgecolor="white")
        ax.set(
            title=f"Distribución de {column}\n{subtitle}",
            xlabel=units[column],
            ylabel="Casos",
        )
        save_figure(
            fig,
            f"{number:02d}_{column}_histograma.png",
            f"Histograma descriptivo de {column}.",
            descriptions,
        )

        fig, ax = plt.subplots(figsize=(8, 3.8))
        box = ax.boxplot(observed, vert=False, patch_artist=True)
        box["boxes"][0].set(facecolor=BLUE, edgecolor=INK)
        box["medians"][0].set(color="white", linewidth=2)
        ax.set(
            title=f"Boxplot de {column}\n{subtitle}",
            xlabel=units[column],
            yticks=[],
        )
        save_figure(
            fig,
            f"{number + 2:02d}_{column}_boxplot.png",
            f"Boxplot descriptivo de {column}; sin tratamiento de valores extremos.",
            descriptions,
        )

    next_number = 6
    for column in CONTINUOUS_COLS:
        groups = [df.loc[target.eq(value), column].dropna() for value in (0, 1)]
        fig, ax = plt.subplots(figsize=(7, 4.8))
        boxes = ax.boxplot(groups, labels=["0", "1"], patch_artist=True)
        for patch, color, hatch in zip(boxes["boxes"], [BLUE, ORANGE], ["", "///"]):
            patch.set(facecolor=color, edgecolor=INK, hatch=hatch)
        for median in boxes["medians"]:
            median.set(color=INK, linewidth=2)
        ax.set(
            title=f"{column} por clase del target\n{subtitle}",
            xlabel=TARGET_NAME,
            ylabel=units[column],
        )
        save_figure(
            fig,
            f"{next_number:02d}_{column}_vs_target_boxplot.png",
            f"Distribución de {column} por target mediante boxplots.",
            descriptions,
        )
        next_number += 1

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(
            groups[0],
            bins=30,
            density=True,
            alpha=0.55,
            color=BLUE,
            edgecolor="white",
            label="Target 0",
        )
        ax.hist(
            groups[1],
            bins=30,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=ORANGE,
            label="Target 1",
        )
        ax.set(
            title=f"{column}: distribución por target\n{subtitle}",
            xlabel=units[column],
            ylabel="Densidad",
        )
        ax.legend(frameon=False)
        save_figure(
            fig,
            f"{next_number:02d}_{column}_vs_target_histograma.png",
            f"Histogramas normalizados de {column} separados por target.",
            descriptions,
        )
        next_number += 1

    for column, table in categorical_tables.items():
        pivot = table.pivot_table(
            index=["category_code", "category_label"],
            columns="target",
            values="n",
            fill_value=0,
            observed=True,
        ).sort_index()
        for target_value in (0, 1):
            if target_value not in pivot.columns:
                pivot[target_value] = 0
        proportions = pivot[[0, 1]].div(pivot[[0, 1]].sum(axis=1), axis=0)
        labels = [textwrap.fill(str(label), 42) for _, label in proportions.index]
        height = max(4.5, 0.62 * len(labels) + 2.1)
        fig, ax = plt.subplots(figsize=(10, height))
        positions = range(len(labels))
        ax.barh(
            positions,
            proportions[0],
            color=BLUE,
            edgecolor=INK,
            label="Target 0",
        )
        ax.barh(
            positions,
            proportions[1],
            left=proportions[0],
            color=ORANGE,
            edgecolor=INK,
            hatch="///",
            label="Target 1",
        )
        ax.set_yticks(list(positions), labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set(
            title=(
                f"Distribución del target dentro de cada categoría de {column}\n"
                f"Categorías observadas; no ponderado; n total={len(df):,}"
            ),
            xlabel="Proporción dentro de la categoría",
            ylabel="Etiqueta de respuesta",
        )
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        save_figure(
            fig,
            f"{next_number:02d}_{column}_vs_target.png",
            f"Proporciones de target 0/1 condicionadas por categoría de {column}.",
            descriptions,
        )
        next_number += 1

    return descriptions


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

    code_to_days, q49_mapping_consistent = build_q49_code_to_days(df)
    if not q49_mapping_consistent:
        raise AssertionError("Mapping q49/texto_q49 inconsistente")
    target = build_target(df, code_to_days)
    target_mask = target.notna()
    df_target = df.loc[target_mask].copy()
    target_observed = target.loc[target_mask].astype("int8")

    data_dictionary = build_data_dictionary(df_target)
    feature_candidate_cols = data_dictionary.loc[
        data_dictionary["model_candidate"], "column"
    ].tolist()
    categorical_candidate_cols = data_dictionary.loc[
        data_dictionary["role"].eq("categorical_feature_candidate"), "column"
    ].tolist()
    interpretation_only_text_cols = [
        column for column in df.columns if column.startswith("texto_")
    ]
    unknown_categorical_cols = data_dictionary.loc[
        data_dictionary["statistical_type"].eq("unknown_categorical"), "column"
    ].tolist()
    constant_cols = data_dictionary.loc[
        data_dictionary["statistical_type"].eq("constant"), "column"
    ].tolist()
    identifier_cols = data_dictionary.loc[
        data_dictionary["statistical_type"].eq("identifier"), "column"
    ].tolist()

    numeric_summary = build_numeric_summary(df_target)
    numeric_by_target = build_numeric_by_target(df_target, target_observed)
    categorical_summary = build_categorical_summary(
        df_target, categorical_candidate_cols
    )

    substantive_cols = [
        column
        for column in df.columns
        if get_layer(column) in {"q_response", "qn_derived"}
    ]
    missing_summary = pd.DataFrame(
        {
            "column": substantive_cols,
            "missing_n": [int(df_target[column].isna().sum()) for column in substantive_cols],
        }
    )
    missing_summary["missing_pct"] = 100 * missing_summary["missing_n"] / len(df_target)
    missing_summary = missing_summary.sort_values(
        ["missing_pct", "column"], ascending=[False, True]
    ).reset_index(drop=True)

    categorical_tables = {
        column: category_target_table(df_target, target_observed, column)
        for column in REPRESENTATIVE_CATEGORICALS
    }
    categorical_by_target = pd.concat(
        categorical_tables.values(), ignore_index=True
    )

    psu_frame = pd.DataFrame(
        {"psu": df_target["psu"], "target": target_observed}
    )
    psu_profile = (
        psu_frame.groupby("psu", observed=True)["target"]
        .agg(
            n_rows="size",
            target_0_n=lambda series: int(series.eq(0).sum()),
            target_1_n=lambda series: int(series.eq(1).sum()),
        )
        .reset_index()
    )
    psu_profile["positive_rate"] = (
        psu_profile["target_1_n"] / psu_profile["n_rows"]
    )

    DATA_DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary.to_csv(DATA_DICTIONARY_PATH, index=False, encoding="utf-8-sig")
    numeric_summary.to_csv(NUMERIC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    categorical_summary.to_csv(
        CATEGORICAL_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )
    missing_summary.to_csv(MISSING_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    psu_profile.to_csv(PSU_PROFILE_PATH, index=False, encoding="utf-8-sig")
    numeric_by_target.to_csv(
        NUMERIC_BY_TARGET_PATH, index=False, encoding="utf-8-sig"
    )
    categorical_by_target.to_csv(
        CATEGORICAL_BY_TARGET_PATH, index=False, encoding="utf-8-sig"
    )

    figure_descriptions = create_figures(
        df_target, target_observed, categorical_tables
    )

    q4_missing = df_target["q4"].isna()
    q5_missing = df_target["q5"].isna()
    q4_q5_pattern = {
        "q4_missing": int(q4_missing.sum()),
        "q5_missing": int(q5_missing.sum()),
        "both_missing": int((q4_missing & q5_missing).sum()),
        "only_q4_missing": int((q4_missing & ~q5_missing).sum()),
        "only_q5_missing": int((~q4_missing & q5_missing).sum()),
    }

    record_unique = df_target["record"].is_unique
    sitio_unique_values = df_target["sitio"].dropna().unique().tolist()
    exact_duplicate_rows = int(df_target.duplicated().sum())
    class_counts = target_observed.value_counts().sort_index()
    psu_both_classes = int(
        ((psu_profile["target_0_n"] > 0) & (psu_profile["target_1_n"] > 0)).sum()
    )
    psu_one_class = int(len(psu_profile) - psu_both_classes)

    required_artifacts = [
        DATA_DICTIONARY_PATH,
        NUMERIC_SUMMARY_PATH,
        CATEGORICAL_SUMMARY_PATH,
        MISSING_SUMMARY_PATH,
        PSU_PROFILE_PATH,
        NUMERIC_BY_TARGET_PATH,
        CATEGORICAL_BY_TARGET_PATH,
    ] + [FIGURE_DIR / filename for filename in figure_descriptions]

    checks = {
        "shape_csv_crudo_56981x309": df.shape == EXPECTED_SHAPE,
        "universo_target_55551": len(df_target) == 55_551,
        "target_solo_0_1": set(target_observed.unique()) == {0, 1},
        "hard_leakage_fuera_features": not (
            set(HARD_LEAKAGE_COLS) & set(feature_candidate_cols)
        ),
        "texto_fuera_features": not (
            set(interpretation_only_text_cols) & set(feature_candidate_cols)
        ),
        "q4_q5_continuas": set(
            data_dictionary.loc[
                data_dictionary["statistical_type"].eq("continuous"), "column"
            ]
        )
        == set(CONTINUOUS_COLS),
        "metadata_separada": set(
            data_dictionary.loc[data_dictionary["layer"].eq("metadata"), "column"]
        )
        == set(METADATA_COLS),
        "record_identificador_unico": bool(record_unique),
        "sitio_constante": len(sitio_unique_values) == 1,
        "data_dictionary_309_filas": len(data_dictionary) == 309,
        "cada_columna_exactamente_una_vez": (
            data_dictionary["column"].is_unique
            and set(data_dictionary["column"]) == set(df.columns)
        ),
        "unknown_categorical_identificadas": data_dictionary.loc[
            data_dictionary["statistical_type"].eq("unknown_categorical"), "notes"
        ].str.contains("A_CONFIRMAR", regex=False).all(),
        "sin_feature_engineering": set(df_target.columns) == set(df.columns),
        "sin_split": not OPERATION_FLAGS["split"],
        "sin_imputacion": (
            not OPERATION_FLAGS["imputation"]
            and df_target.equals(df.loc[target_mask])
        ),
        "sin_scaling": not OPERATION_FLAGS["scaling"],
        "sin_encoding_features": not OPERATION_FLAGS["feature_encoding"],
        "sin_smote_pca_feature_selection": not any(
            OPERATION_FLAGS[name] for name in ("smote", "pca", "feature_selection")
        ),
        "artefactos_generados": all(
            path.exists() and path.stat().st_size > 0 for path in required_artifacts
        ),
    }

    print("=== RECONSTRUCCION B2 ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Shape crudo: {df.shape}")
    print(f"Shape universo target observado: {df_target.shape}")
    print(
        "Warnings de lectura: "
        + (
            "; ".join(
                f"{item.category.__name__}: {item.message}"
                for item in read_warnings
            )
            if read_warnings
            else "ninguno"
        )
    )

    print("\n=== ESTRUCTURA, ROLES Y TIPOS ===")
    print("Columnas por layer:")
    print(data_dictionary["layer"].value_counts().sort_index().to_string())
    print("Columnas por role:")
    print(data_dictionary["role"].value_counts().sort_index().to_string())
    print("Variables por statistical_type:")
    print(data_dictionary["statistical_type"].value_counts().sort_index().to_string())
    print(f"Features candidatas totales: {len(feature_candidate_cols)}")
    print(f"- continuas: {len(CONTINUOUS_COLS)}")
    print(f"- categóricas: {len(categorical_candidate_cols)}")
    print(f"Hard leakage: {len(HARD_LEAKAGE_COLS)} {HARD_LEAKAGE_COLS}")
    print(f"Metadata: {len(METADATA_COLS)} {METADATA_COLS}")
    print(f"Texto interpretativo: {len(interpretation_only_text_cols)}")
    print(f"Columnas constantes: {constant_cols}")
    print(f"Identificadores evidentes: {identifier_cols}")
    print(f"Duplicados exactos de fila: {exact_duplicate_rows}")
    print(f"Unknown categorical ({len(unknown_categorical_cols)}): {unknown_categorical_cols}")

    print("\n=== DATA DICTIONARY ===")
    print(f"Ruta: {DATA_DICTIONARY_PATH}")
    print(f"Filas: {len(data_dictionary)}")
    print("Distribución statistical_type:")
    print(data_dictionary["statistical_type"].value_counts().sort_index().to_string())
    print(f"Lista completa unknown_categorical: {unknown_categorical_cols}")

    print("\n=== TARGET NO PONDERADO ===")
    for target_value, count in class_counts.items():
        print(
            f"Target {target_value}: {count} "
            f"({100 * count / len(target_observed):.4f}%)"
        )

    print("\n=== CONTINUAS: RESUMEN UNIVARIADO ===")
    print(numeric_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nContinuas por target:")
    print(numeric_by_target.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for column in CONTINUOUS_COLS:
        subset = numeric_by_target.loc[numeric_by_target["column"].eq(column)].set_index("target")
        print(
            f"{column}, diferencia descriptiva target 1 - target 0: "
            f"media={subset.loc[1, 'mean'] - subset.loc[0, 'mean']:.4f}; "
            f"mediana={subset.loc[1, 'median'] - subset.loc[0, 'median']:.4f}"
        )

    print("\n=== CATEGORICAS VISUALIZADAS ===")
    for column, reason in REPRESENTATIVE_CATEGORICALS.items():
        print(f"{column}: {reason}")

    print("\n=== q50/q51: REVISION DESCRIPTIVA ===")
    for column in SAME_DOMAIN_REVIEW_COLS:
        paired = f"texto_{column}"
        mapping = (
            df_target.loc[df_target[column].notna(), [column, paired]]
            .drop_duplicates()
            .sort_values(column)
        )
        distribution = interpretation_series(df_target, column).value_counts(dropna=True)
        missing_n = int(df_target[column].isna().sum())
        print(f"\n{column} mapping código <-> texto:")
        print(mapping.to_string(index=False))
        print(
            f"Cardinalidad={df_target[column].nunique()}; faltantes={missing_n} "
            f"({100 * missing_n / len(df_target):.4f}%)"
        )
        distribution_table = pd.DataFrame(
            {
                "n": distribution,
                "pct_observado": 100 * distribution / distribution.sum(),
            }
        )
        print("Distribución:")
        print(distribution_table.to_string(float_format=lambda value: f"{value:.4f}"))
        print("Target dentro de categoría:")
        print(
            categorical_tables[column][
                [
                    "category_code",
                    "category_label",
                    "target",
                    "n",
                    "target_pct_within_category",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print(
            "Estado: MISMO_DOMINIO_NO_LEAKAGE_DURO — pendiente de decisión final de features"
        )

    print("\n=== FALTANTES ===")
    print("Top 15 variables sustantivas por missing_pct:")
    print(missing_summary.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "q4/q5 sobre universo target observado: "
        + "; ".join(f"{key}={value}" for key, value in q4_q5_pattern.items())
    )
    print("Estado mecanismo: PATRON_A_REVISAR_EN_B3")

    print("\n=== METADATA MUESTRAL / ADMINISTRATIVA ===")
    print(
        f"record: cardinalidad={df_target['record'].nunique()}; "
        f"único={record_unique}"
    )
    print(
        f"sitio: cardinalidad={df_target['sitio'].nunique()}; "
        f"valores={sitio_unique_values}"
    )
    psu_sizes = df_target.groupby("psu", observed=True).size()
    print(
        f"psu: n={psu_sizes.size}; tamaño min={psu_sizes.min()}; "
        f"mediana={psu_sizes.median():.1f}; max={psu_sizes.max()}"
    )
    stratum_sizes = df_target.groupby("stratum", observed=True).size()
    print(
        f"stratum: n={stratum_sizes.size}; tamaño min={stratum_sizes.min()}; "
        f"mediana={stratum_sizes.median():.1f}; max={stratum_sizes.max()}"
    )
    weight = df_target["weight"]
    print(
        f"weight: count={weight.count()}; min={weight.min():.6f}; "
        f"mediana={weight.median():.6f}; media={weight.mean():.6f}; "
        f"max={weight.max():.6f}"
    )

    print("\n=== PERFIL PSU ===")
    print(f"Número de PSU: {len(psu_profile)}")
    print(
        f"Filas por PSU: min={psu_profile['n_rows'].min()}; "
        f"mediana={psu_profile['n_rows'].median():.1f}; "
        f"max={psu_profile['n_rows'].max()}"
    )
    print(
        f"Tasa positiva por PSU: min={psu_profile['positive_rate'].min():.6f}; "
        f"mediana={psu_profile['positive_rate'].median():.6f}; "
        f"max={psu_profile['positive_rate'].max():.6f}"
    )
    print(f"PSU con ambas clases: {psu_both_classes}")
    print(f"PSU con una sola clase: {psu_one_class}")
    print(f"Ruta perfil PSU: {PSU_PROFILE_PATH}")

    print("\n=== ARTEFACTOS CSV ===")
    for path in (
        DATA_DICTIONARY_PATH,
        NUMERIC_SUMMARY_PATH,
        CATEGORICAL_SUMMARY_PATH,
        MISSING_SUMMARY_PATH,
        PSU_PROFILE_PATH,
        NUMERIC_BY_TARGET_PATH,
        CATEGORICAL_BY_TARGET_PATH,
    ):
        print(path)

    print("\n=== GRAFICOS ===")
    print(f"Carpeta: {FIGURE_DIR}")
    for filename, description in figure_descriptions.items():
        print(f"{filename}: {description}")

    print("\n=== ASSERTS / CHECKS ===")
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FALLO'}")
    passed_count = sum(checks.values())
    print(f"Resultado: {passed_count}/{len(checks)} checks OK")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"B2 detenido; checks fallidos: {failed}")


if __name__ == "__main__":
    main()
