from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, EXPECTED_SHAPE, locate_csv
from b1_target_emse import TARGET_NAME, build_q49_code_to_days, build_target


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "b4_figures"
IQR_PATH = BASE_DIR / "b4_outliers_iqr.csv"
Z3_PATH = BASE_DIR / "b4_outliers_z3.csv"
DOMAIN_PATH = BASE_DIR / "b4_domain_review_flags.csv"
OVERLAP_PATH = BASE_DIR / "b4_outlier_method_overlap.csv"
CASE_REVIEW_PATH = BASE_DIR / "b4_outlier_case_review.csv"
SUMMARY_PATH = BASE_DIR / "b4_outlier_summary.csv"
TARGET_FLAGS_PATH = BASE_DIR / "b4_outlier_flags_by_target.csv"
COHERENCE_PATH = BASE_DIR / "b4_coherence_checks.csv"

OUTLIERS_POLICY_B4 = "RETENER_Y_MARCAR_SIN_MODIFICAR"

OPERATION_FLAGS = {
    "imputation": False,
    "row_deletion": False,
    "feature_modification": False,
    "split": False,
    "transformer_fit": False,
    "scaling": False,
    "normalization": False,
    "encoding": False,
    "model_discretization": False,
    "persistent_feature_engineering": False,
    "smote": False,
    "pca": False,
    "feature_selection": False,
    "model_training": False,
}

BLUE = "#2F6690"
LIGHT_BLUE = "#BCD7EA"
ORANGE = "#F28E2B"
INK = "#263238"
GRID = "#D9DEE3"


def numeric_profile(series: pd.Series, variable: str) -> dict[str, object]:
    observed = series.dropna()
    finite = observed[np.isfinite(observed)]
    if finite.empty:
        raise AssertionError(f"{variable} no tiene valores finitos para diagnosticar.")
    return {
        "variable": variable,
        "total_n": int(len(series)),
        "observed_n": int(series.notna().sum()),
        "missing_n": int(series.isna().sum()),
        "missing_pct": float(100 * series.isna().mean()),
        "finite_n": int(len(finite)),
        "min": float(finite.min()),
        "q1": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "q3": float(finite.quantile(0.75)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)),
        "skewness": float(finite.skew()),
    }


def iqr_diagnosis(
    series: pd.Series,
    variable: str,
) -> tuple[dict[str, object], pd.Series]:
    finite = series.dropna()
    finite = finite[np.isfinite(finite)]
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    below = series.lt(lower).fillna(False)
    above = series.gt(upper).fillna(False)
    flagged = below | above
    observed_n = int(series.notna().sum())
    return (
        {
            "variable": variable,
            "observed_n": observed_n,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "below_n": int(below.sum()),
            "above_n": int(above.sum()),
            "flagged_n": int(flagged.sum()),
            "flagged_pct_observed": 100 * int(flagged.sum()) / observed_n,
            "flag_label": "CANDIDATO_OUTLIER_IQR",
        },
        flagged.astype(bool),
    )


def z3_diagnosis(
    series: pd.Series,
    variable: str,
) -> tuple[dict[str, object], pd.Series]:
    finite = series.dropna()
    finite = finite[np.isfinite(finite)]
    mean = float(finite.mean())
    std = float(finite.std(ddof=1))
    lower = mean - 3 * std
    upper = mean + 3 * std
    below = series.lt(lower).fillna(False)
    above = series.gt(upper).fillna(False)
    flagged = below | above
    observed_n = int(series.notna().sum())
    return (
        {
            "variable": variable,
            "observed_n": observed_n,
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "below_n": int(below.sum()),
            "above_n": int(above.sum()),
            "flagged_n": int(flagged.sum()),
            "flagged_pct_observed": 100 * int(flagged.sum()) / observed_n,
            "flag_label": "CANDIDATO_OUTLIER_Z3",
        },
        flagged.astype(bool),
    )


def build_domain_flags(
    series_by_variable: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rule_masks = {
        "q4": [("q4 < 1.30 m", "lower", 1.30, series_by_variable["q4"].lt(1.30))],
        "q5": [
            ("q5 < 30 kg", "lower", 30.0, series_by_variable["q5"].lt(30.0)),
            ("q5 > 150 kg", "upper", 150.0, series_by_variable["q5"].gt(150.0)),
        ],
        "imc_diag": [
            ("imc_diag < 10", "lower", 10.0, series_by_variable["imc_diag"].lt(10.0)),
            ("imc_diag > 60", "upper", 60.0, series_by_variable["imc_diag"].gt(60.0)),
        ],
    }
    rows: list[dict[str, object]] = []
    combined: dict[str, pd.Series] = {}
    for variable, rules in rule_masks.items():
        observed_n = int(series_by_variable[variable].notna().sum())
        combined_mask = pd.Series(False, index=series_by_variable[variable].index)
        for rule, direction, threshold, raw_mask in rules:
            mask = raw_mask.fillna(False).astype(bool)
            combined_mask |= mask
            rows.append(
                {
                    "variable": variable,
                    "rule": rule,
                    "direction": direction,
                    "threshold": threshold,
                    "observed_n": observed_n,
                    "flagged_n": int(mask.sum()),
                    "flagged_pct_observed": 100 * int(mask.sum()) / observed_n,
                    "flag_label": "DOMAIN_REVIEW_FLAG",
                }
            )
        combined[variable] = combined_mask
        rows.append(
            {
                "variable": variable,
                "rule": "ANY_DOMAIN_REVIEW_FLAG",
                "direction": "either",
                "threshold": np.nan,
                "observed_n": observed_n,
                "flagged_n": int(combined_mask.sum()),
                "flagged_pct_observed": 100 * int(combined_mask.sum()) / observed_n,
                "flag_label": "DOMAIN_REVIEW_FLAG",
            }
        )
    return pd.DataFrame(rows), combined


def build_overlap(
    series_by_variable: dict[str, pd.Series],
    iqr_masks: dict[str, pd.Series],
    z3_masks: dict[str, pd.Series],
    domain_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for variable, series in series_by_variable.items():
        iqr = iqr_masks[variable]
        z3 = z3_masks[variable]
        domain = domain_masks[variable]
        rows.append(
            {
                "variable": variable,
                "observed_n": int(series.notna().sum()),
                "iqr_n": int(iqr.sum()),
                "z3_n": int(z3.sum()),
                "domain_flag_n": int(domain.sum()),
                "iqr_only_n": int((iqr & ~z3).sum()),
                "z3_only_n": int((z3 & ~iqr).sum()),
                "iqr_and_z3_n": int((iqr & z3).sum()),
                "domain_only_n": int((domain & ~iqr & ~z3).sum()),
                "iqr_and_domain_n": int((iqr & domain).sum()),
                "z3_and_domain_n": int((z3 & domain).sum()),
                "all_three_n": int((iqr & z3 & domain).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_target_flags(
    target: pd.Series,
    series_by_variable: dict[str, pd.Series],
    iqr_masks: dict[str, pd.Series],
    domain_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    any_domain = domain_masks["q4"] | domain_masks["q5"] | domain_masks["imc_diag"]
    rows = []
    for target_value in (0, 1):
        class_mask = target.eq(target_value)
        row: dict[str, object] = {
            "target": target_value,
            "n_total": int(class_mask.sum()),
        }
        for variable in ("q4", "q5", "imc_diag"):
            observed = class_mask & series_by_variable[variable].notna()
            flagged = class_mask & iqr_masks[variable]
            row[f"{variable}_observed_n"] = int(observed.sum())
            row[f"{variable}_iqr_n"] = int(flagged.sum())
            row[f"{variable}_iqr_pct_total"] = 100 * int(flagged.sum()) / int(class_mask.sum())
            row[f"{variable}_iqr_pct_observed"] = 100 * int(flagged.sum()) / int(observed.sum())
        domain_flagged = class_mask & any_domain
        row["any_domain_n"] = int(domain_flagged.sum())
        row["any_domain_pct_total"] = 100 * int(domain_flagged.sum()) / int(class_mask.sum())
        complete_observed = class_mask & series_by_variable["imc_diag"].notna()
        row["any_domain_pct_complete_cases"] = (
            100 * int(domain_flagged.sum()) / int(complete_observed.sum())
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)


def create_figures(
    series_by_variable: dict[str, pd.Series],
    target: pd.Series,
    iqr_table: pd.DataFrame,
    iqr_masks: dict[str, pd.Series],
    z3_masks: dict[str, pd.Series],
    domain_masks: dict[str, pd.Series],
    q4: pd.Series,
    q5: pd.Series,
    overlap: pd.DataFrame,
) -> dict[str, str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    descriptions: dict[str, str] = {}
    units = {"q4": "Altura (m)", "q5": "Peso (kg)", "imc_diag": "IMC diagnóstico"}

    for number, variable in enumerate(("q4", "q5", "imc_diag")):
        observed = series_by_variable[variable].dropna()
        limits = iqr_table.loc[iqr_table["variable"].eq(variable)].iloc[0]
        hist_number = 1 + number * 2
        box_number = hist_number + 1

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(observed, bins=45, color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.7)
        ax.axvline(limits["lower"], color=ORANGE, linestyle="--", linewidth=2, label=f"IQR inferior={limits['lower']:.2f}")
        ax.axvline(limits["upper"], color=INK, linestyle="--", linewidth=2, label=f"IQR superior={limits['upper']:.2f}")
        ax.set(
            title=f"Distribución de {variable}\nCasos observados: n={len(observed):,}; límites IQR descriptivos",
            xlabel=units[variable],
            ylabel="Frecuencia",
        )
        ax.legend(frameon=False)
        style_axis(ax)
        hist_filename = f"{hist_number:02d}_{variable}_hist_iqr.png"
        descriptions[hist_filename] = f"Histograma de {variable} con límites IQR."
        save_figure(fig, hist_filename)

        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.boxplot(
            observed,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": LIGHT_BLUE, "edgecolor": BLUE},
            medianprops={"color": INK, "linewidth": 2},
            whiskerprops={"color": BLUE},
            capprops={"color": BLUE},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": ORANGE, "markeredgecolor": ORANGE, "alpha": 0.35},
        )
        ax.set(
            title=f"Boxplot de {variable}\nValores originales observados; sin tratamiento",
            xlabel=units[variable],
            yticks=[],
        )
        style_axis(ax)
        box_filename = f"{box_number:02d}_{variable}_boxplot.png"
        descriptions[box_filename] = f"Boxplot descriptivo de {variable}."
        save_figure(fig, box_filename)

    any_domain = domain_masks["q4"] | domain_masks["q5"] | domain_masks["imc_diag"]
    complete = q4.notna() & q5.notna()
    regular = complete & ~any_domain
    flagged = complete & any_domain
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(q4.loc[regular], q5.loc[regular], s=10, color=LIGHT_BLUE, alpha=0.28, edgecolors="none", label=f"Sin domain flag (n={int(regular.sum()):,})")
    ax.scatter(q4.loc[flagged], q5.loc[flagged], s=24, facecolors="none", edgecolors=ORANGE, linewidths=0.9, alpha=0.9, label=f"DOMAIN_REVIEW_FLAG (n={int(flagged.sum()):,})")
    ax.set(
        title="Altura y peso observados\nCasos completos; banderas de dominio destacadas sin modificar valores",
        xlabel="q4 — altura (m)",
        ylabel="q5 — peso (kg)",
    )
    ax.legend(frameon=False)
    style_axis(ax)
    descriptions["07_q4_vs_q5_domain_flags.png"] = "Scatter q4 vs q5 con banderas de dominio."
    save_figure(fig, "07_q4_vs_q5_domain_flags.png")

    imc = series_by_variable["imc_diag"]
    by_target = [imc.loc[target.eq(value)].dropna() for value in (0, 1)]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    boxes = ax.boxplot(
        by_target,
        labels=["0", "1"],
        patch_artist=True,
        boxprops={"facecolor": LIGHT_BLUE, "edgecolor": BLUE},
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": ORANGE, "markeredgecolor": ORANGE, "alpha": 0.3},
    )
    boxes["boxes"][1].set_facecolor("white")
    boxes["boxes"][1].set_hatch("///")
    ax.set(
        title="IMC diagnóstico por target\nSólo casos q4/q5 completos; distribución descriptiva",
        xlabel=TARGET_NAME,
        ylabel="IMC diagnóstico",
    )
    style_axis(ax)
    descriptions["08_imc_diag_by_target_boxplot.png"] = "Boxplot de IMC diagnóstico por target."
    save_figure(fig, "08_imc_diag_by_target_boxplot.png")

    positions = np.arange(len(overlap))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_iqr = ax.bar(positions - width, overlap["iqr_n"], width, color=BLUE, label="IQR")
    bars_z3 = ax.bar(positions, overlap["z3_n"], width, facecolor="white", edgecolor=BLUE, hatch="///", label="±3SD")
    bars_domain = ax.bar(positions + width, overlap["domain_flag_n"], width, color=ORANGE, label="Domain review")
    ax.set_xticks(positions, overlap["variable"])
    ax.set(
        title="Candidatos por método diagnóstico\nConteos sobre valores observados; las reglas no implican error",
        ylabel="Cantidad marcada",
    )
    for bars in (bars_iqr, bars_z3, bars_domain):
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.legend(frameon=False)
    style_axis(ax)
    descriptions["09_outlier_method_counts.png"] = "Comparación de cantidades marcadas por IQR, ±3SD y dominio."
    save_figure(fig, "09_outlier_method_counts.png")

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

    code_to_days, mapping_consistent = build_q49_code_to_days(df)
    if not mapping_consistent:
        raise AssertionError("Mapping q49/texto_q49 inconsistente; B4 detenido.")
    target = build_target(df, code_to_days)
    target_mask = target.notna()
    df_target = df.loc[target_mask].copy()
    target_observed = target.loc[target_mask].astype("int8")

    q4_before = df_target["q4"].copy(deep=True)
    q5_before = df_target["q5"].copy(deep=True)
    q4_numeric = pd.to_numeric(df_target["q4"], errors="coerce")
    q5_numeric = pd.to_numeric(df_target["q5"], errors="coerce")

    q4_nonnumeric = df_target["q4"].notna() & q4_numeric.isna()
    q5_nonnumeric = df_target["q5"].notna() & q5_numeric.isna()
    q4_infinite = q4_numeric.notna() & ~np.isfinite(q4_numeric)
    q5_infinite = q5_numeric.notna() & ~np.isfinite(q5_numeric)
    q4_nonpositive = q4_numeric.notna() & q4_numeric.le(0)
    q5_nonpositive = q5_numeric.notna() & q5_numeric.le(0)

    complete_pairs = df_target["q4"].notna() & df_target["q5"].notna()
    valid_imc = (
        q4_numeric.notna()
        & q5_numeric.notna()
        & np.isfinite(q4_numeric)
        & np.isfinite(q5_numeric)
        & q4_numeric.gt(0)
        & q5_numeric.gt(0)
    )
    imc_diag = pd.Series(np.nan, index=df_target.index, dtype="float64", name="imc_diag")
    imc_diag.loc[valid_imc] = q5_numeric.loc[valid_imc] / q4_numeric.loc[valid_imc].pow(2)

    series_by_variable = {
        "q4": q4_numeric,
        "q5": q5_numeric,
        "imc_diag": imc_diag,
    }
    profiles = pd.DataFrame(
        [numeric_profile(series, variable) for variable, series in series_by_variable.items()]
    )

    iqr_rows: list[dict[str, object]] = []
    z3_rows: list[dict[str, object]] = []
    iqr_masks: dict[str, pd.Series] = {}
    z3_masks: dict[str, pd.Series] = {}
    for variable, series in series_by_variable.items():
        iqr_row, iqr_mask = iqr_diagnosis(series, variable)
        z3_row, z3_mask = z3_diagnosis(series, variable)
        iqr_rows.append(iqr_row)
        z3_rows.append(z3_row)
        iqr_masks[variable] = iqr_mask
        z3_masks[variable] = z3_mask
    iqr_table = pd.DataFrame(iqr_rows)
    z3_table = pd.DataFrame(z3_rows)
    iqr_table.to_csv(IQR_PATH, index=False, encoding="utf-8-sig")
    z3_table.to_csv(Z3_PATH, index=False, encoding="utf-8-sig")

    domain_table, domain_masks = build_domain_flags(series_by_variable)
    domain_table.to_csv(DOMAIN_PATH, index=False, encoding="utf-8-sig")

    overlap = build_overlap(series_by_variable, iqr_masks, z3_masks, domain_masks)
    overlap.to_csv(OVERLAP_PATH, index=False, encoding="utf-8-sig")

    any_domain = domain_masks["q4"] | domain_masks["q5"] | domain_masks["imc_diag"]
    representation_error = (
        q4_nonnumeric
        | q5_nonnumeric
        | q4_infinite
        | q5_infinite
        | q4_nonpositive
        | q5_nonpositive
    )
    imc_infinite = imc_diag.notna() & ~np.isfinite(imc_diag)
    unexpected_imc_nan = valid_imc & imc_diag.isna()
    imc_recalculated = q5_numeric.loc[valid_imc] / q4_numeric.loc[valid_imc].pow(2)
    imc_inconsistent = pd.Series(False, index=df_target.index)
    imc_inconsistent.loc[valid_imc] = ~np.isclose(
        imc_diag.loc[valid_imc],
        imc_recalculated,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    )
    calculation_error = imc_infinite | unexpected_imc_nan | imc_inconsistent
    demonstrated_error = representation_error | calculation_error
    case_mask = any_domain | demonstrated_error
    case_review = pd.DataFrame(
        {
            "record": df_target.loc[case_mask, "record"],
            "target": target_observed.loc[case_mask],
            "q4": q4_numeric.loc[case_mask],
            "q5": q5_numeric.loc[case_mask],
            "imc_diag": imc_diag.loc[case_mask],
            "flag_q4_iqr": iqr_masks["q4"].loc[case_mask],
            "flag_q4_z3": z3_masks["q4"].loc[case_mask],
            "flag_q4_domain": domain_masks["q4"].loc[case_mask],
            "flag_q5_iqr": iqr_masks["q5"].loc[case_mask],
            "flag_q5_z3": z3_masks["q5"].loc[case_mask],
            "flag_q5_domain": domain_masks["q5"].loc[case_mask],
            "flag_imc_iqr": iqr_masks["imc_diag"].loc[case_mask],
            "flag_imc_z3": z3_masks["imc_diag"].loc[case_mask],
            "flag_imc_domain": domain_masks["imc_diag"].loc[case_mask],
            "error_representacion_demostrado": representation_error.loc[case_mask],
            "error_calculo_demostrado": calculation_error.loc[case_mask],
            "error_demostrado": demonstrated_error.loc[case_mask],
        }
    )
    flag_columns = [column for column in case_review if column.startswith("flag_")]
    case_review["flag_count"] = case_review[flag_columns].sum(axis=1).astype(int)
    case_review["review_class"] = np.where(
        case_review["error_demostrado"],
        "REQUIERE_REVISION",
        "VALOR_EXTREMO_NO_ERROR_DEMOSTRADO",
    )
    case_review["evidence"] = np.select(
        [
            case_review["error_representacion_demostrado"],
            case_review["error_calculo_demostrado"],
        ],
        [
            "Valor no numérico, infinito o no positivo en q4/q5.",
            "IMC infinito, ausente inesperadamente o inconsistente con q5/q4**2.",
        ],
        default="Activa al menos una regla descriptiva pre-registrada de dominio.",
    )
    case_review = case_review.sort_values(
        ["error_demostrado", "flag_count", "imc_diag"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    case_review.to_csv(CASE_REVIEW_PATH, index=False, encoding="utf-8-sig")

    target_flags = build_target_flags(
        target_observed,
        series_by_variable,
        iqr_masks,
        domain_masks,
    )
    target_flags.to_csv(TARGET_FLAGS_PATH, index=False, encoding="utf-8-sig")

    coherence = pd.DataFrame(
        [
            ("q4_non_numeric", int(q4_nonnumeric.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("q5_non_numeric", int(q5_nonnumeric.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("q4_infinite", int(q4_infinite.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("q5_infinite", int(q5_infinite.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("q4_nonpositive", int(q4_nonpositive.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("q5_nonpositive", int(q5_nonpositive.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("imc_infinite", int(imc_infinite.sum()), "ERROR_DE_REPRESENTACION_DEMOSTRADO"),
            ("unexpected_imc_nan", int(unexpected_imc_nan.sum()), "ERROR_DE_CALCULO_DEMOSTRADO"),
            ("imc_formula_inconsistent", int(imc_inconsistent.sum()), "ERROR_DE_CALCULO_DEMOSTRADO"),
            ("duplicate_record", int(df_target["record"].duplicated().sum()), "INCONSISTENCIA_DE_IDENTIFICADOR"),
        ],
        columns=["check", "affected_n", "classification_if_positive"],
    )
    coherence["evidence"] = coherence.apply(
        lambda row: (
            "Sin casos detectados."
            if row["affected_n"] == 0
            else f"{int(row['affected_n'])} casos cumplen el chequeo {row['check']}."
        ),
        axis=1,
    )
    coherence.to_csv(COHERENCE_PATH, index=False, encoding="utf-8-sig")
    demonstrated_error_n = int(demonstrated_error.sum())

    domain_any = domain_table.loc[domain_table["rule"].eq("ANY_DOMAIN_REVIEW_FLAG")].set_index("variable")
    summary_rows = []
    for variable in series_by_variable:
        profile = profiles.loc[profiles["variable"].eq(variable)].iloc[0]
        iqr_row = iqr_table.loc[iqr_table["variable"].eq(variable)].iloc[0]
        z3_row = z3_table.loc[z3_table["variable"].eq(variable)].iloc[0]
        summary_rows.append(
            {
                "variable": variable,
                "observed_n": int(profile["observed_n"]),
                "missing_n": int(profile["missing_n"]),
                "min": profile["min"],
                "q1": profile["q1"],
                "median": profile["median"],
                "q3": profile["q3"],
                "max": profile["max"],
                "mean": profile["mean"],
                "std": profile["std"],
                "skewness": profile["skewness"],
                "iqr_lower": iqr_row["lower"],
                "iqr_upper": iqr_row["upper"],
                "iqr_flagged_n": int(iqr_row["flagged_n"]),
                "z3_lower": z3_row["lower"],
                "z3_upper": z3_row["upper"],
                "z3_flagged_n": int(z3_row["flagged_n"]),
                "domain_flag_n": int(domain_any.loc[variable, "flagged_n"]),
                "final_policy": OUTLIERS_POLICY_B4,
            }
        )
    outlier_summary = pd.DataFrame(summary_rows)
    outlier_summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    figure_descriptions = create_figures(
        series_by_variable,
        target_observed,
        iqr_table,
        iqr_masks,
        z3_masks,
        domain_masks,
        q4_numeric,
        q5_numeric,
        overlap,
    )

    expected_artifacts = [
        IQR_PATH,
        Z3_PATH,
        DOMAIN_PATH,
        OVERLAP_PATH,
        CASE_REVIEW_PATH,
        SUMMARY_PATH,
        TARGET_FLAGS_PATH,
        COHERENCE_PATH,
    ] + [FIGURE_DIR / filename for filename in figure_descriptions]

    checks = {
        "shape_crudo_56981x309": df.shape == EXPECTED_SHAPE,
        "universo_b4_55551": df_target.shape == (55_551, 309),
        "q4_sin_modificacion": df_target["q4"].equals(q4_before),
        "q5_sin_modificacion": df_target["q5"].equals(q5_before),
        "missing_q4_q5_20209": int(df_target["q4"].isna().sum()) == 20_209 and int(df_target["q5"].isna().sum()) == 20_209,
        "coausencia_q4_q5_exacta": df_target["q4"].isna().equals(df_target["q5"].isna()),
        "imc_observados_igual_pares_completos": int(imc_diag.notna().sum()) == int(complete_pairs.sum()),
        "sin_imc_si_falta_q4_o_q5": bool(imc_diag.loc[~complete_pairs].isna().all()),
        "sin_imputacion": not OPERATION_FLAGS["imputation"] and df_target.isna().equals(df.loc[target_mask].isna()),
        "sin_eliminacion_filas": not OPERATION_FLAGS["row_deletion"] and len(df_target) == 55_551,
        "sin_modificacion_features": not OPERATION_FLAGS["feature_modification"] and list(df_target.columns) == list(df.columns),
        "sin_split": not OPERATION_FLAGS["split"],
        "sin_fit_transformadores": not OPERATION_FLAGS["transformer_fit"],
        "sin_scaling_normalizacion": not OPERATION_FLAGS["scaling"] and not OPERATION_FLAGS["normalization"],
        "sin_feature_engineering_persistente": not OPERATION_FLAGS["persistent_feature_engineering"] and "imc_diag" not in df_target.columns,
        "sin_smote": not OPERATION_FLAGS["smote"],
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_encoding_discretizacion_modelado": not OPERATION_FLAGS["encoding"] and not OPERATION_FLAGS["model_discretization"],
        "sin_entrenamiento": not OPERATION_FLAGS["model_training"],
        "politica_final_correcta": OUTLIERS_POLICY_B4 == "RETENER_Y_MARCAR_SIN_MODIFICAR",
        "errores_demostrados_con_evidencia_o_cero": demonstrated_error_n == 0 or bool(coherence.loc[coherence["affected_n"].gt(0), "evidence"].str.strip().ne("").all()),
        "calculo_imc_coherente": int(unexpected_imc_nan.sum()) == 0 and int(imc_inconsistent.sum()) == 0,
        "artefactos_existen": all(path.exists() for path in expected_artifacts),
    }

    print("=== RECONSTRUCCION B4 ===")
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
    print(f"Pares q4/q5 completos: {int(complete_pairs.sum())}")
    print(f"IMC diagnóstico observado: {int(imc_diag.notna().sum())}")
    print(f"IMC diagnóstico faltante: {int(imc_diag.isna().sum())}")

    print("\n=== RESUMEN q4 / q5 / IMC DIAGNOSTICO ===")
    print(profiles.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== IQR ===")
    print(iqr_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== ±3 SD ===")
    print(z3_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== DOMAIN REVIEW FLAGS ===")
    print(domain_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\n=== SOLAPAMIENTO ENTRE METODOS ===")
    print(overlap.to_string(index=False))

    print("\n=== CASOS PRIORITARIOS ===")
    print(f"Casos guardados: {len(case_review)}")
    print(f"Errores de representación demostrados: {int(case_review['error_representacion_demostrado'].sum())}")
    if not case_review.empty:
        print(case_review.head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\n=== COHERENCIA ===")
    print(coherence.to_string(index=False))
    print(f"Total de errores demostrados: {demonstrated_error_n}")

    print("\n=== FLAGS POR TARGET ===")
    print(target_flags.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Interpretación: diferencias descriptivas; no se infiere causalidad.")

    print("\n=== LIMITACION COMPLETE CASES ===")
    print("IMC sólo puede calcularse donde q4 y q5 están observados.")
    print("B3 diagnosticó la ausencia q4/q5 como MAR_HIPOTESIS_RESPALDADA.")
    print("Por lo tanto, la distribución observada de IMC corresponde al subconjunto con medición y no debe presentarse como si representara sin reservas a los 55.551 casos.")

    print("\n=== POLITICA B4 ===")
    print(f"OUTLIERS_POLICY_B4 = {OUTLIERS_POLICY_B4}")
    print("Se conservan todos los valores; las banderas son artefactos diagnósticos temporales.")
    print("No se eliminó, winsorizó, reemplazó, transformó ni convirtió ningún extremo en NaN.")
    print("Si un límite data-dependent se usara después: fit en TRAIN y aplicación del mismo criterio a TRAIN/TEST.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)
    print("\n=== GRAFICOS ===")
    for filename, description in figure_descriptions.items():
        print(f"{filename}: {description}")

    print("\n=== CHECKS B4 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B4 detenido; checks fallidos: {failed}")
    print("B4 finalizado.")


if __name__ == "__main__":
    main()
