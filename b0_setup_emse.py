from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd


CSV_FILENAME = "EMSE_DatosAbiertos.csv"
EXPECTED_SHAPE = (56_981, 309)
CSV_ENCODING = "utf-8-sig"
CSV_SEPARATOR = ","
CONTINUOUS_COLS = ["q4", "q5"]
SAMPLE_METADATA_EXPECTED = ["weight", "stratum", "psu", "sitio", "record"]
SAME_DOMAIN_NOT_HARD_LEAKAGE = ["q50", "q51"]


def locate_csv() -> Path:
    matches = sorted(Path(__file__).resolve().parent.rglob(CSV_FILENAME))
    if len(matches) != 1:
        raise RuntimeError(
            f"Se esperaba un unico {CSV_FILENAME}; encontrados: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def percent(count: int, total: int) -> float:
    return 100 * count / total


def print_list(label: str, values: list[str]) -> None:
    print(f"{label} ({len(values)}): {json.dumps(values, ensure_ascii=False)}")


def exact_functional_derivatives(
    df: pd.DataFrame, source_col: str, candidate_cols: list[str]
) -> list[str]:
    derivatives: list[str] = []
    for col in candidate_cols:
        pair = df[[source_col, col]]
        complete = pair.dropna()
        same_missingness = pair[source_col].isna().equals(pair[col].isna())
        deterministic = (
            not complete.empty
            and complete.groupby(source_col, observed=True)[col]
            .nunique(dropna=False)
            .le(1)
            .all()
        )
        if same_missingness and deterministic and complete[col].nunique() > 1:
            derivatives.append(col)
    return derivatives


def numeric_summary(df: pd.DataFrame, col: str) -> dict[str, object]:
    series = df[col]
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_numeric = int((series.notna() & numeric.isna()).sum())
    assert invalid_numeric == 0, f"{col} contiene valores no numericos"
    missing = int(series.isna().sum())
    return {
        "dtype_pandas": str(series.dtype),
        "count": int(numeric.count()),
        "faltantes": missing,
        "faltantes_pct": percent(missing, len(df)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "media": float(numeric.mean()),
        "mediana": float(numeric.median()),
        "valores_unicos": int(series.nunique(dropna=True)),
    }


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

    memory_mib = df.memory_usage(index=True, deep=True).sum() / 1024**2

    print("=== CARGA ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Encoding: {CSV_ENCODING}")
    print(f"Separador: {CSV_SEPARATOR!r}")
    print(f"Shape: {df.shape}")
    print(f"Numero total de columnas: {df.shape[1]}")
    print(f"Memoria aproximada DataFrame (deep): {memory_mib:.2f} MiB")
    if read_warnings:
        print("Warnings de lectura:")
        for item in read_warnings:
            print(f"- {item.category.__name__}: {item.message}")
    else:
        print("Warnings de lectura: ninguno")

    if df.shape != EXPECTED_SHAPE:
        raise AssertionError(
            f"Shape real {df.shape} != shape esperado {EXPECTED_SHAPE}. B0 detenido."
        )

    response_code_cols = [
        col for col in df.columns if re.fullmatch(r"q\d+", col)
    ]
    response_text_cols = [
        col for col in df.columns if re.fullmatch(r"texto_q\d+", col)
    ]
    derived_qn_cols = [col for col in df.columns if col.startswith("qn")]

    layer_a = set(response_code_cols)
    layer_b = set(response_text_cols)
    layer_c = set(derived_qn_cols)
    structural_unclassified_cols = [
        col for col in df.columns if col not in layer_a | layer_b | layer_c
    ]

    sample_metadata_cols = [
        col for col in SAMPLE_METADATA_EXPECTED if col in df.columns
    ]
    missing_sample_metadata_cols = [
        col for col in SAMPLE_METADATA_EXPECTED if col not in df.columns
    ]
    unclassified_to_confirm_cols = [
        col
        for col in structural_unclassified_cols
        if col not in sample_metadata_cols
    ]

    q49_exact_qn_derivatives = exact_functional_derivatives(
        df, "q49", derived_qn_cols
    )
    leakage_cols_known = q49_exact_qn_derivatives
    same_domain_not_hard_leakage_cols = [
        col for col in SAME_DOMAIN_NOT_HARD_LEAKAGE if col in df.columns
    ]

    categorical_candidate_cols = [
        col
        for col in response_code_cols + response_text_cols + derived_qn_cols
        if col not in set(CONTINUOUS_COLS + leakage_cols_known)
    ]

    print("\n=== INVENTARIO ESTRUCTURAL ===")
    print_list("Capa A - respuestas codificadas qN", response_code_cols)
    print_list("Capa B - texto de respuesta texto_qN", response_text_cols)
    print_list("Capa C - derivadas qn*", derived_qn_cols)
    print_list("Columnas fuera de las tres capas", structural_unclassified_cols)

    print("\n=== METADATA MUESTRAL / ADMINISTRATIVA ===")
    print_list("Metadata encontrada", sample_metadata_cols)
    print_list("Metadata esperada no encontrada", missing_sample_metadata_cols)
    print("columna | dtype | unicos_sin_na | faltantes | faltantes_pct")
    for col in sample_metadata_cols:
        missing = int(df[col].isna().sum())
        print(
            f"{col} | {df[col].dtype} | {df[col].nunique(dropna=True)} | "
            f"{missing} | {percent(missing, len(df)):.4f}%"
        )

    print("\n=== TIPADO CONCEPTUAL INICIAL ===")
    print_list("Continuas originales verificadas", CONTINUOUS_COLS)
    for col in CONTINUOUS_COLS:
        summary = numeric_summary(df, col)
        print(
            f"{col}: dtype={summary['dtype_pandas']}; "
            f"count={summary['count']}; "
            f"faltantes={summary['faltantes']} "
            f"({summary['faltantes_pct']:.4f}%); "
            f"min={summary['min']:.6g}; max={summary['max']:.6g}; "
            f"media={summary['media']:.6g}; mediana={summary['mediana']:.6g}; "
            f"unicos={summary['valores_unicos']}"
        )
    other_qn_unique = df[
        [col for col in response_code_cols if col not in CONTINUOUS_COLS]
    ].nunique(dropna=True)
    max_other_unique = int(other_qn_unique.max())
    max_other_unique_cols = other_qn_unique[
        other_qn_unique.eq(max_other_unique)
    ].index.tolist()
    print(
        "Chequeo de respaldo: maximo de valores unicos entre otros qN="
        f"{max_other_unique}, columnas={max_other_unique_cols}"
    )
    print(
        "Contrato conceptual B0: dtype fisico de pandas != tipo estadistico; "
        "todo lo sustantivo restante queda como categorico inicial."
    )

    q4_missing = df["q4"].isna()
    q5_missing = df["q5"].isna()
    both_missing = int((q4_missing & q5_missing).sum())
    only_q4_missing = int((q4_missing & ~q5_missing).sum())
    only_q5_missing = int((~q4_missing & q5_missing).sum())

    print("\n=== CHEQUEO DE FALTANTES q4/q5 ===")
    print(
        f"q4 faltantes: {int(q4_missing.sum())} "
        f"({percent(int(q4_missing.sum()), len(df)):.4f}%)"
    )
    print(
        f"q5 faltantes: {int(q5_missing.sum())} "
        f"({percent(int(q5_missing.sum()), len(df)):.4f}%)"
    )
    print(
        f"Ambas faltan: {both_missing} "
        f"({percent(both_missing, len(df)):.4f}%)"
    )
    print(f"Solo falta q4: {only_q4_missing}")
    print(f"Solo falta q5: {only_q5_missing}")

    print("\n=== LEAKAGE CONOCIDO ===")
    print_list(
        "qn* con dependencia funcional exacta y misma mascara de faltantes que q49",
        q49_exact_qn_derivatives,
    )
    print_list("Leakage conocido detectado", leakage_cols_known)
    print_list(
        "Mismo dominio, no leakage duro confirmado",
        same_domain_not_hard_leakage_cols,
    )
    print("Mapeos empiricos q49 -> leakage conocido:")
    for col in leakage_cols_known:
        mapping = (
            df[["q49", col]]
            .drop_duplicates()
            .sort_values(["q49", col], na_position="last")
        )
        print(f"{col}: {mapping.to_dict(orient='records')}")

    print("\n=== INVENTARIOS REUTILIZABLES ===")
    print_list("response_code_cols", response_code_cols)
    print_list("response_text_cols", response_text_cols)
    print_list("derived_qn_cols", derived_qn_cols)
    print_list("continuous_cols", CONTINUOUS_COLS)
    print_list("sample_metadata_cols", sample_metadata_cols)
    print_list("leakage_cols_known", leakage_cols_known)
    print_list("categorical_candidate_cols", categorical_candidate_cols)
    print_list("unclassified_to_confirm_cols", unclassified_to_confirm_cols)

    checks = {
        "shape_esperado": df.shape == EXPECTED_SHAPE,
        "existe_q4": "q4" in df.columns,
        "existe_q5": "q5" in df.columns,
        "existe_q49": "q49" in df.columns,
        "existe_qnpa5g": "qnpa5g" in df.columns,
        "capas_sin_superposicion": not (
            (layer_a & layer_b) | (layer_a & layer_c) | (layer_b & layer_c)
        ),
        "todas_columnas_explicadas": (
            layer_a | layer_b | layer_c | set(structural_unclassified_cols)
        )
        == set(df.columns),
        "qnpa5g_en_leakage": "qnpa5g" in leakage_cols_known,
        "qnpa7g_en_leakage": "qnpa7g" in leakage_cols_known,
        "q50_q51_no_excluidas_como_leakage_duro": not (
            set(same_domain_not_hard_leakage_cols) & set(leakage_cols_known)
        ),
    }
    for name, passed in checks.items():
        assert passed, f"Fallo check: {name}"

    print("\n=== ASSERTS / CHECKS ===")
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FALLO'}")
    print(f"Resultado: {len(checks)}/{len(checks)} checks OK")


if __name__ == "__main__":
    main()
