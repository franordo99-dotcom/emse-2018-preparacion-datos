from __future__ import annotations

import json
import re
import sys
import unicodedata
import warnings

import pandas as pd

from b0_setup_emse import (
    CSV_ENCODING,
    CSV_SEPARATOR,
    EXPECTED_SHAPE,
    exact_functional_derivatives,
    locate_csv,
)


TARGET_NAME = "target_pa_oms5"
TARGET_VALUE_SOURCE_COLS = ["q49"]
TARGET_SEMANTICS_COLS = ["texto_q49"]
EXPECTED_TARGET_PROXY_COLS = ["qn49", "qnpa5g", "qnpa7g"]
SAME_DOMAIN_REVIEW_EXPECTED = ["q50", "q51"]


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.strip().casefold())


def build_q49_code_to_days(df: pd.DataFrame) -> tuple[dict[object, int], bool]:
    observed = df.loc[df["q49"].notna(), ["q49", "texto_q49"]]
    no_missing_labels = observed["texto_q49"].notna().all()
    one_label_per_code = (
        observed.groupby("q49", observed=True)["texto_q49"]
        .nunique(dropna=False)
        .eq(1)
        .all()
    )

    mapping: dict[object, int] = {}
    for code, label in observed.drop_duplicates().itertuples(index=False, name=None):
        match = re.fullmatch(r"(\d+)\s+dias?", normalize_label(label))
        if match is None:
            raise AssertionError(
                f"No se pudo interpretar texto_q49={label!r} para codigo {code!r}"
            )
        mapping[code] = int(match.group(1))

    complete_days = set(mapping.values()) == set(range(8))
    one_code_per_day = len(set(mapping.values())) == len(mapping)
    consistent = bool(
        no_missing_labels
        and one_label_per_code
        and complete_days
        and one_code_per_day
    )
    return mapping, consistent


def build_target(df: pd.DataFrame, code_to_days: dict[object, int]) -> pd.Series:
    real_days = df["q49"].map(code_to_days)
    target = pd.Series(pd.NA, index=df.index, dtype="Int8", name=TARGET_NAME)
    observed = real_days.notna()
    target.loc[observed] = real_days.loc[observed].ge(5).astype("int8")
    return target


def build_yes_no_code_mapping(
    df: pd.DataFrame, code_col: str, text_col: str
) -> tuple[dict[object, int], bool]:
    observed = df.loc[df[code_col].notna(), [code_col, text_col]]
    no_missing_labels = observed[text_col].notna().all()
    one_label_per_code = (
        observed.groupby(code_col, observed=True)[text_col]
        .nunique(dropna=False)
        .eq(1)
        .all()
    )

    mapping: dict[object, int] = {}
    for code, label in observed.drop_duplicates().itertuples(index=False, name=None):
        normalized = normalize_label(label)
        if normalized not in {"si", "no"}:
            raise AssertionError(
                f"Etiqueta no binaria en {text_col}: {label!r}"
            )
        mapping[code] = 1 if normalized == "si" else 0

    consistent = bool(
        no_missing_labels
        and one_label_per_code
        and set(mapping.values()) == {0, 1}
        and len(mapping) == 2
    )
    return mapping, consistent


def is_exact_text_counterpart(
    df: pd.DataFrame, code_col: str, text_col: str
) -> bool:
    pairs = pd.DataFrame(
        {
            "code": df[code_col].astype("string").fillna("<NA>"),
            "text": df[text_col].astype("string").fillna("<NA>"),
        }
    ).drop_duplicates()
    code_to_text = pairs.groupby("code")["text"].nunique().eq(1).all()
    text_to_code = pairs.groupby("text")["code"].nunique().eq(1).all()
    return bool(code_to_text and text_to_code)


def print_json_list(label: str, values: list[str]) -> None:
    print(f"{label} ({len(values)}): {json.dumps(values, ensure_ascii=False)}")


def format_code(value: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


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

    required_cols = [
        "q49",
        "texto_q49",
        "qn49",
        "texto_qn49",
        "qnpa5g",
        "texto_qnpa5g",
        "qnpa7g",
        "texto_qnpa7g",
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise AssertionError(f"Columnas B1 faltantes: {missing_required}")
    if df.shape != EXPECTED_SHAPE:
        raise AssertionError(
            f"Shape real {df.shape} != esperado {EXPECTED_SHAPE}. B1 detenido."
        )

    code_to_days, q49_mapping_consistent = build_q49_code_to_days(df)
    if not q49_mapping_consistent:
        raise AssertionError("Mapping q49 -> texto_q49 inconsistente")

    target = build_target(df, code_to_days)
    q49_missing = int(df["q49"].isna().sum())
    valid_target = int(target.notna().sum())
    positives = int(target.eq(1).sum())
    negatives = int(target.eq(0).sum())

    qnpa5g_code_to_binary, qnpa5g_mapping_consistent = (
        build_yes_no_code_mapping(df, "qnpa5g", "texto_qnpa5g")
    )
    qnpa5g_binary = df["qnpa5g"].map(qnpa5g_code_to_binary).astype("Int8")
    comparable = target.notna() & qnpa5g_binary.notna()
    matches = int(target.loc[comparable].eq(qnpa5g_binary.loc[comparable]).sum())
    discrepancies = int(comparable.sum()) - matches
    agreement_pct = 100 * matches / int(comparable.sum())
    contingency = pd.crosstab(
        target.loc[comparable],
        qnpa5g_binary.loc[comparable],
        rownames=[TARGET_NAME],
        colnames=["qnpa5g_binario"],
    )

    both_observed = int((target.notna() & qnpa5g_binary.notna()).sum())
    both_missing = int((target.isna() & qnpa5g_binary.isna()).sum())
    target_only_missing = int((target.isna() & qnpa5g_binary.notna()).sum())
    qnpa5g_only_missing = int((target.notna() & qnpa5g_binary.isna()).sum())

    derived_qn_cols = [col for col in df.columns if col.startswith("qn")]
    exact_q49_qn_derivatives = exact_functional_derivatives(
        df, "q49", derived_qn_cols
    )
    target_proxy_cols = [
        col
        for col in EXPECTED_TARGET_PROXY_COLS
        if col in exact_q49_qn_derivatives
    ]
    target_source_cols = TARGET_VALUE_SOURCE_COLS.copy()

    counterpart_checks: dict[str, bool] = {}
    target_text_proxy_cols: list[str] = []
    for code_col in target_source_cols + target_proxy_cols:
        text_col = f"texto_{code_col}"
        exact = text_col in df.columns and is_exact_text_counterpart(
            df, code_col, text_col
        )
        counterpart_checks[text_col] = exact
        if exact:
            target_text_proxy_cols.append(text_col)

    hard_leakage_cols = list(
        dict.fromkeys(
            target_source_cols + target_proxy_cols + target_text_proxy_cols
        )
    )
    same_domain_review_cols = [
        col for col in SAME_DOMAIN_REVIEW_EXPECTED if col in df.columns
    ]
    same_domain_review_text_cols = [
        f"texto_{col}"
        for col in same_domain_review_cols
        if f"texto_{col}" in df.columns
    ]
    interpretation_only_text_cols = [
        col for col in df.columns if col.startswith("texto_")
    ]
    future_feature_review_cols = [
        col
        for col in df.columns
        if col not in set(hard_leakage_cols + interpretation_only_text_cols)
    ]

    target_observed_index = target.index[target.notna()]
    df_target = df.loc[target_observed_index].copy()

    checks = {
        "shape_crudo": df.shape == EXPECTED_SHAPE,
        "existe_q49": "q49" in df.columns,
        "existe_texto_q49": "texto_q49" in df.columns,
        "existe_qnpa5g": "qnpa5g" in df.columns,
        "mapping_q49_texto_consistente": q49_mapping_consistent,
        "target_solo_0_1_nan": set(target.dropna().unique()) <= {0, 1},
        "target_nan_iff_q49_nan": target.isna().equals(df["q49"].isna()),
        "qnpa5g_no_usada_para_target": (
            "qnpa5g" not in TARGET_VALUE_SOURCE_COLS + TARGET_SEMANTICS_COLS
        ),
        "concordancia_target_qnpa5g_reportada": int(comparable.sum()) > 0,
        "q49_excluida_de_features": (
            "q49" in hard_leakage_cols
            and "q49" not in future_feature_review_cols
        ),
        "proxies_duros_excluidos_de_features": not (
            set(target_proxy_cols) & set(future_feature_review_cols)
        ),
        "q50_q51_no_son_leakage_duro": not (
            set(same_domain_review_cols) & set(hard_leakage_cols)
        ),
        "proxies_esperados_detectados": set(EXPECTED_TARGET_PROXY_COLS)
        == set(target_proxy_cols),
        "textos_target_proxy_equivalentes": all(counterpart_checks.values()),
        "mapping_qnpa5g_consistente": qnpa5g_mapping_consistent,
    }

    print("=== CARGA B1 ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Shape crudo: {df.shape}")
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

    print("\n=== MAPPING q49 ===")
    print("codigo q49 | texto_q49 | dias reales interpretados")
    q49_pairs = df[["q49", "texto_q49"]].drop_duplicates().copy()
    q49_pairs["_sort"] = q49_pairs["q49"].fillna(float("inf"))
    for code, label, _ in q49_pairs.sort_values("_sort").itertuples(
        index=False, name=None
    ):
        days = code_to_days.get(code)
        print(f"{format_code(code)} | {label} | {days if days is not None else 'NA'}")

    print("\n=== TARGET ===")
    print(f"Nombre: {TARGET_NAME}")
    print(f"Filas totales: {len(df)}")
    print(
        f"q49 faltantes: {q49_missing} "
        f"({100 * q49_missing / len(df):.4f}%)"
    )
    print(f"Casos con target valido: {valid_target}")
    print(f"Positivos: {positives} ({100 * positives / valid_target:.4f}%)")
    print(f"Negativos: {negatives} ({100 * negatives / valid_target:.4f}%)")

    print("\n=== VALIDACION INDEPENDIENTE qnpa5g ===")
    print("codigo qnpa5g | texto_qnpa5g | clase interpretada")
    qnpa_pairs = df[["qnpa5g", "texto_qnpa5g"]].drop_duplicates().copy()
    qnpa_pairs["_sort"] = qnpa_pairs["qnpa5g"].fillna(float("inf"))
    for code, label, _ in qnpa_pairs.sort_values("_sort").itertuples(
        index=False, name=None
    ):
        binary = qnpa5g_code_to_binary.get(code)
        print(
            f"{format_code(code)} | {label} | "
            f"{binary if binary is not None else 'NA'}"
        )
    print("Tabla de contingencia:")
    print(contingency.to_string())
    print(f"Filas comparables: {int(comparable.sum())}")
    print(f"Coincidencias: {matches}")
    print(f"Discrepancias: {discrepancies}")
    print(f"Concordancia: {agreement_pct:.4f}%")
    print("Patron de faltantes:")
    print(f"- ambos observados: {both_observed}")
    print(f"- ambos faltantes: {both_missing}")
    print(f"- solo target faltante: {target_only_missing}")
    print(f"- solo qnpa5g faltante: {qnpa5g_only_missing}")
    if discrepancies:
        discrepancy_rows = df.loc[
            comparable & target.ne(qnpa5g_binary),
            [
                "record",
                "q49",
                "texto_q49",
                "qnpa5g",
                "texto_qnpa5g",
            ],
        ].copy()
        discrepancy_rows[TARGET_NAME] = target.loc[discrepancy_rows.index]
        print("Observaciones discrepantes:")
        print(discrepancy_rows.to_string(index=False))

    print("\n=== MAPA DEFINITIVO DE LEAKAGE B1 ===")
    print_json_list("target_source_cols", target_source_cols)
    print_json_list("target_proxy_cols", target_proxy_cols)
    print("Caracterizacion empirica de proxies respecto de q49:")
    for proxy_col in target_proxy_cols:
        text_col = f"texto_{proxy_col}"
        mapping = (
            df[["q49", proxy_col, text_col]]
            .drop_duplicates()
            .sort_values(["q49", proxy_col], na_position="last")
        )
        print(f"{proxy_col}: dependencia deterministica directa=True")
        print(mapping.to_string(index=False))
    for text_col, exact in counterpart_checks.items():
        print(f"Contraparte textual {text_col}: equivalencia exacta={exact}")
    print_json_list("target_text_proxy_cols", target_text_proxy_cols)
    print_json_list("hard_leakage_cols", hard_leakage_cols)
    print_json_list("same_domain_review_cols", same_domain_review_cols)
    print_json_list(
        "same_domain_review_text_cols (solo interpretacion)",
        same_domain_review_text_cols,
    )
    print_json_list(
        "interpretation_only_text_cols", interpretation_only_text_cols
    )
    print(
        "Principio: texto_* es capa interpretativa/documental; no constituye "
        "una feature adicional simultanea a su codigo equivalente."
    )

    print("\n=== UNIVERSO CON TARGET OBSERVADO ===")
    print(f"Shape df_target: {df_target.shape}")
    print(f"Longitud target observado: {len(target_observed_index)}")

    print("\n=== ASSERTS / CHECKS ===")
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FALLO'}")
    passed_count = sum(checks.values())
    print(f"Resultado: {passed_count}/{len(checks)} checks OK")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"B1 detenido; checks fallidos: {failed}")


if __name__ == "__main__":
    main()
