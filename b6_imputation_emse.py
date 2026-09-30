from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, EXPECTED_SHAPE, locate_csv
from b1_target_emse import TARGET_NAME, build_q49_code_to_days, build_target


BASE_DIR = Path(__file__).resolve().parent
B5_MANIFEST_PATH = BASE_DIR / "b5_split_manifest.json"
B5_FEATURE_INVENTORY_PATH = BASE_DIR / "b5_feature_inventory.csv"
B5_TRAIN_RECORDS_PATH = BASE_DIR / "b5_train_records.csv"
B5_TEST_RECORDS_PATH = BASE_DIR / "b5_test_records.csv"

CONTINUOUS_PARAMS_PATH = BASE_DIR / "b6_continuous_imputation_params.csv"
MISSING_BEFORE_AFTER_PATH = BASE_DIR / "b6_missing_before_after.csv"
INDICATOR_SUMMARY_PATH = BASE_DIR / "b6_indicator_summary.csv"
CATEGORICAL_FILL_SUMMARY_PATH = BASE_DIR / "b6_categorical_missing_fill_summary.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b6_feature_inventory.csv"
MANIFEST_PATH = BASE_DIR / "b6_imputation_manifest.json"
X_TRAIN_PATH = BASE_DIR / "b6_X_train.pkl"
X_TEST_PATH = BASE_DIR / "b6_X_test.pkl"
Y_TRAIN_PATH = BASE_DIR / "b6_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b6_y_test.pkl"

EXPECTED_UNIVERSE_N = 55_551
EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_INPUT_FEATURE_N = 148
EXPECTED_OUTPUT_FEATURE_N = 149
CONTINUOUS_COLS = ["q4", "q5"]
INDICATOR_COL = "q4q5_faltaba"
CATEGORICAL_FILL_VALUE = "sin_dato"
FIT_PARTITION = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True

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
METADATA_COLS = ["record", "psu", "stratum", "weight", "sitio"]
UNKNOWN_CATEGORICAL_COLS = [
    "q28",
    "q34",
    "q40",
    "q45",
    "q59",
    "q60",
    "q69",
    "q70",
    "q71",
    "q72",
    "q79",
]

OPERATION_FLAGS = {
    "split_executed": False,
    "test_used_for_fit": False,
    "mode_used": False,
    "test_categories_learned": False,
    "scaling": False,
    "encoding": False,
    "discretization": False,
    "outlier_treatment": False,
    "feature_engineering_other_than_required_indicator": False,
    "smote": False,
    "pca": False,
    "feature_selection": False,
    "model_training": False,
}


def membership_hash(records: pd.Series) -> str:
    payload = "\n".join(sorted(records.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_b5_assignment(path: Path) -> pd.DataFrame:
    assignment = pd.read_csv(path, encoding="utf-8-sig")
    required = ["split_position", "original_index", "record"]
    if assignment.columns.tolist() != required:
        raise AssertionError(f"Estructura inesperada en {path.name}: {assignment.columns.tolist()}")
    assignment = assignment.sort_values("split_position", kind="stable").reset_index(drop=True)
    if assignment["split_position"].tolist() != list(range(len(assignment))):
        raise AssertionError(f"split_position no es continuo en {path.name}")
    if not assignment["original_index"].is_unique:
        raise AssertionError(f"original_index duplicado en {path.name}")
    return assignment


def observed_values_unchanged(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: list[str],
) -> bool:
    for column in columns:
        observed = before[column].notna()
        if not before.loc[observed, column].astype(object).eq(
            after.loc[observed, column].astype(object)
        ).all():
            return False
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b5_manifest = json.loads(B5_MANIFEST_PATH.read_text(encoding="utf-8"))
    b5_inventory = pd.read_csv(B5_FEATURE_INVENTORY_PATH, encoding="utf-8-sig")
    train_assignment = read_b5_assignment(B5_TRAIN_RECORDS_PATH)
    test_assignment = read_b5_assignment(B5_TEST_RECORDS_PATH)

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
        raise AssertionError("Mapping q49/texto_q49 inconsistente; B6 detenido.")
    target = build_target(df, code_to_days)
    target_mask = target.notna()
    df_target = df.loc[target_mask].copy()
    y = target.loc[target_mask].astype("int8").rename(TARGET_NAME)

    inventory_included = (
        b5_inventory["included_in_X_b5"].astype(str).str.strip().str.lower().eq("true")
    )
    feature_cols = b5_inventory.loc[inventory_included, "column"].tolist()
    manifest_feature_cols = b5_manifest["feature_columns"]

    train_indices = train_assignment["original_index"].tolist()
    test_indices = test_assignment["original_index"].tolist()
    train_index_set = set(train_indices)
    test_index_set = set(test_indices)
    universe_index_set = set(df_target.index)

    boundary_checks = {
        "shape_crudo": df.shape == EXPECTED_SHAPE,
        "universo": len(df_target) == EXPECTED_UNIVERSE_N,
        "train_n": len(train_assignment) == EXPECTED_TRAIN_N,
        "test_n": len(test_assignment) == EXPECTED_TEST_N,
        "indices_disjuntos": train_index_set.isdisjoint(test_index_set),
        "union_igual_universo": train_index_set | test_index_set == universe_index_set,
        "feature_inventory_igual_manifest": feature_cols == manifest_feature_cols,
        "train_record_igual_csv": train_assignment["record"].astype(str).tolist()
        == df_target.loc[train_indices, "record"].astype(str).tolist(),
        "test_record_igual_csv": test_assignment["record"].astype(str).tolist()
        == df_target.loc[test_indices, "record"].astype(str).tolist(),
        "train_hash_igual_b5": membership_hash(train_assignment["record"])
        == b5_manifest["train_record_membership_sha256"],
        "test_hash_igual_b5": membership_hash(test_assignment["record"])
        == b5_manifest["test_record_membership_sha256"],
        "test_congelado_b5": b5_manifest["TEST_FROZEN_AFTER_B5"] is True,
    }
    if not all(boundary_checks.values()):
        failed = [name for name, passed in boundary_checks.items() if not passed]
        raise AssertionError(f"Frontera B5 inválida; B6 detenido: {failed}")

    X_train_raw = df_target.loc[train_indices, feature_cols].copy()
    X_test_raw = df_target.loc[test_indices, feature_cols].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    X_train_before_b6 = X_train_raw.copy(deep=True)
    X_test_before_b6 = X_test_raw.copy(deep=True)
    X_train_b6 = X_train_raw.copy(deep=True)
    X_test_b6 = X_test_raw.copy(deep=True)

    categorical_cols = [column for column in feature_cols if column not in CONTINUOUS_COLS]
    train_missing_q4 = X_train_before_b6["q4"].isna()
    train_missing_q5 = X_train_before_b6["q5"].isna()
    test_missing_q4 = X_test_before_b6["q4"].isna()
    test_missing_q5 = X_test_before_b6["q5"].isna()
    if not train_missing_q4.equals(train_missing_q5):
        raise AssertionError("q4/q5 no tienen co-ausencia exacta en train; B6 detenido.")
    if not test_missing_q4.equals(test_missing_q5):
        raise AssertionError("q4/q5 no tienen co-ausencia exacta en test; B6 detenido.")

    X_train_b6[INDICATOR_COL] = train_missing_q4.astype("int8")
    X_test_b6[INDICATOR_COL] = test_missing_q4.astype("int8")

    median_q4_train = float(X_train_before_b6["q4"].median())
    median_q5_train = float(X_train_before_b6["q5"].median())
    learned_medians = {"q4": median_q4_train, "q5": median_q5_train}

    continuous_params = pd.DataFrame(
        [
            {
                "column": column,
                "strategy": "median",
                "fit_partition": "train",
                "fit_observed_n": int(X_train_before_b6[column].notna().sum()),
                "fit_missing_n": int(X_train_before_b6[column].isna().sum()),
                "learned_value": learned_medians[column],
            }
            for column in CONTINUOUS_COLS
        ]
    )

    for column in CONTINUOUS_COLS:
        X_train_b6.loc[X_train_before_b6[column].isna(), column] = learned_medians[column]
        X_test_b6.loc[X_test_before_b6[column].isna(), column] = learned_medians[column]

    categorical_rows: list[dict[str, object]] = []
    for column in categorical_cols:
        train_missing = X_train_before_b6[column].isna()
        test_missing = X_test_before_b6[column].isna()
        if train_missing.any() or test_missing.any():
            X_train_b6[column] = X_train_b6[column].astype(object)
            X_test_b6[column] = X_test_b6[column].astype(object)
        X_train_b6.loc[train_missing, column] = CATEGORICAL_FILL_VALUE
        X_test_b6.loc[test_missing, column] = CATEGORICAL_FILL_VALUE
        categorical_rows.append(
            {
                "column": column,
                "strategy": "fixed_category_sin_dato",
                "fit_partition": "NO_FIT_FIXED_RULE",
                "train_missing_before": int(train_missing.sum()),
                "test_missing_before": int(test_missing.sum()),
                "train_cells_replaced": int(train_missing.sum()),
                "test_cells_replaced": int(test_missing.sum()),
                "train_missing_after": int(X_train_b6[column].isna().sum()),
                "test_missing_after": int(X_test_b6[column].isna().sum()),
                "train_dtype_before": str(X_train_before_b6[column].dtype),
                "train_dtype_after": str(X_train_b6[column].dtype),
                "test_dtype_before": str(X_test_before_b6[column].dtype),
                "test_dtype_after": str(X_test_b6[column].dtype),
            }
        )
    categorical_summary = pd.DataFrame(categorical_rows)

    indicator_rows: list[dict[str, object]] = []
    for partition, indicator in {
        "train": X_train_b6[INDICATOR_COL],
        "test": X_test_b6[INDICATOR_COL],
    }.items():
        counts = indicator.value_counts().sort_index()
        for value in (0, 1):
            indicator_rows.append(
                {
                    "partition": partition,
                    "indicator": INDICATOR_COL,
                    "value": value,
                    "n": int(counts.get(value, 0)),
                    "pct_partition": float(100 * counts.get(value, 0) / len(indicator)),
                }
            )
    indicator_summary = pd.DataFrame(indicator_rows)

    missing_rows: list[dict[str, object]] = []
    for partition, before, after in [
        ("train", X_train_before_b6, X_train_b6),
        ("test", X_test_before_b6, X_test_b6),
    ]:
        for stage, frame in [("before", before), ("after", after)]:
            stage_categorical_cols = [
                column for column in frame.columns if column not in CONTINUOUS_COLS + [INDICATOR_COL]
            ]
            groups = {
                "all_features": frame.columns.tolist(),
                "q4": ["q4"],
                "q5": ["q5"],
                "categorical_features": stage_categorical_cols,
            }
            for group, columns in groups.items():
                cell_n = len(frame) * len(columns)
                missing_n = int(frame[columns].isna().sum().sum())
                missing_rows.append(
                    {
                        "partition": partition,
                        "stage": stage,
                        "column_group": group,
                        "n_rows": len(frame),
                        "n_features": len(columns),
                        "cell_n": cell_n,
                        "missing_n": missing_n,
                        "missing_pct_cells": float(100 * missing_n / cell_n),
                    }
                )
    missing_before_after = pd.DataFrame(missing_rows)

    b6_feature_inventory = b5_inventory.loc[inventory_included].copy()
    b6_feature_inventory["output_position"] = range(len(b6_feature_inventory))
    b6_feature_inventory["source"] = "B5"
    b6_feature_inventory["b6_action"] = b6_feature_inventory["column"].map(
        lambda column: "train_median_imputation" if column in CONTINUOUS_COLS else "fixed_category_sin_dato"
    )
    indicator_inventory = pd.DataFrame(
        [
            {
                "column": INDICATOR_COL,
                "layer": "b6_technical",
                "role": "missing_indicator",
                "statistical_type": "binary_categorical",
                "model_candidate_b2": False,
                "included_in_X_b5": False,
                "exclusion_reason": "",
                "output_position": len(b6_feature_inventory),
                "source": "B6_FIXED_RULE",
                "b6_action": "derived_from_original_joint_missing_mask",
            }
        ]
    )
    b6_feature_inventory = pd.concat(
        [b6_feature_inventory, indicator_inventory], ignore_index=True
    )

    continuous_params.to_csv(CONTINUOUS_PARAMS_PATH, index=False, encoding="utf-8-sig")
    missing_before_after.to_csv(MISSING_BEFORE_AFTER_PATH, index=False, encoding="utf-8-sig")
    indicator_summary.to_csv(INDICATOR_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    categorical_summary.to_csv(CATEGORICAL_FILL_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    b6_feature_inventory.to_csv(FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig")

    X_train_b6.to_pickle(X_TRAIN_PATH)
    X_test_b6.to_pickle(X_TEST_PATH)
    y_train.to_pickle(Y_TRAIN_PATH)
    y_test.to_pickle(Y_TEST_PATH)

    output_paths = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    manifest = {
        "block": "B6",
        "source_split_manifest": B5_MANIFEST_PATH.name,
        "source_split_random_state": b5_manifest["random_state"],
        "source_train_record_membership_sha256": b5_manifest[
            "train_record_membership_sha256"
        ],
        "source_test_record_membership_sha256": b5_manifest[
            "test_record_membership_sha256"
        ],
        "universe_n": len(df_target),
        "train_n": len(X_train_b6),
        "test_n": len(X_test_b6),
        "input_feature_n": len(feature_cols),
        "output_feature_n": X_train_b6.shape[1],
        "continuous_strategy": "train_median_plus_joint_missing_indicator",
        "learned_train_medians": learned_medians,
        "categorical_strategy": "fixed_category_sin_dato",
        "categorical_fill_value": CATEGORICAL_FILL_VALUE,
        "indicator": INDICATOR_COL,
        "FIT_PARTITION": FIT_PARTITION,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "split_reexecuted": False,
        "output_files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    expected_artifacts = [
        CONTINUOUS_PARAMS_PATH,
        MISSING_BEFORE_AFTER_PATH,
        INDICATOR_SUMMARY_PATH,
        CATEGORICAL_FILL_SUMMARY_PATH,
        FEATURE_INVENTORY_PATH,
        MANIFEST_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    train_categorical_missing_before = int(
        X_train_before_b6[categorical_cols].isna().sum().sum()
    )
    test_categorical_missing_before = int(
        X_test_before_b6[categorical_cols].isna().sum().sum()
    )
    new_features = set(X_train_b6.columns) - set(X_train_before_b6.columns)
    removed_features = set(X_train_before_b6.columns) - set(X_train_b6.columns)

    checks = {
        "universo_reconstruido_55551": len(df_target) == EXPECTED_UNIVERSE_N,
        "train_44440": len(X_train_raw) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_raw) == EXPECTED_TEST_N,
        "membership_coincide_exactamente_B5": all(boundary_checks.values()),
        "no_se_reejecuto_split": not OPERATION_FLAGS["split_executed"],
        "test_sigue_congelado": TEST_FROZEN_AFTER_B5 and b5_manifest["TEST_FROZEN_AFTER_B5"],
        "entrada_148_features": X_train_raw.shape[1] == X_test_raw.shape[1] == EXPECTED_INPUT_FEATURE_N,
        "salida_149_features": X_train_b6.shape[1] == X_test_b6.shape[1] == EXPECTED_OUTPUT_FEATURE_N,
        "unica_nueva_feature_indicador": new_features == {INDICATOR_COL} and not removed_features,
        "qn40_fuera": "qn40" not in X_train_b6.columns,
        "leakage_fuera": not (set(HARD_LEAKAGE_COLS) & set(X_train_b6.columns)),
        "target_fuera": TARGET_NAME not in X_train_b6.columns,
        "texto_fuera": not any(column.startswith("texto_") for column in X_train_b6.columns),
        "metadata_fuera": not (set(METADATA_COLS) & set(X_train_b6.columns)),
        "q50_q51_presentes": {"q50", "q51"}.issubset(X_train_b6.columns),
        "unknown_11_presentes": set(UNKNOWN_CATEGORICAL_COLS).issubset(X_train_b6.columns),
        "coausencia_q4_q5_train": train_missing_q4.equals(train_missing_q5),
        "coausencia_q4_q5_test": test_missing_q4.equals(test_missing_q5),
        "mediana_q4_fit_train_only": continuous_params.loc[
            continuous_params["column"].eq("q4"), "fit_partition"
        ].eq("train").all() and median_q4_train == X_train_before_b6["q4"].median(),
        "mediana_q5_fit_train_only": continuous_params.loc[
            continuous_params["column"].eq("q5"), "fit_partition"
        ].eq("train").all() and median_q5_train == X_train_before_b6["q5"].median(),
        "q4_sin_nan_train_test": not X_train_b6["q4"].isna().any() and not X_test_b6["q4"].isna().any(),
        "q5_sin_nan_train_test": not X_train_b6["q5"].isna().any() and not X_test_b6["q5"].isna().any(),
        "q4_observados_sin_cambios": observed_values_unchanged(X_train_before_b6, X_train_b6, ["q4"]) and observed_values_unchanged(X_test_before_b6, X_test_b6, ["q4"]),
        "q5_observados_sin_cambios": observed_values_unchanged(X_train_before_b6, X_train_b6, ["q5"]) and observed_values_unchanged(X_test_before_b6, X_test_b6, ["q5"]),
        "indicador_igual_mascara_original": X_train_b6[INDICATOR_COL].equals(train_missing_q4.astype("int8")) and X_test_b6[INDICATOR_COL].equals(test_missing_q4.astype("int8")),
        "categoricas_nan_train_a_sin_dato": all(
            X_train_b6.loc[X_train_before_b6[column].isna(), column].eq(CATEGORICAL_FILL_VALUE).all()
            for column in categorical_cols
        ),
        "categoricas_nan_test_a_sin_dato": all(
            X_test_b6.loc[X_test_before_b6[column].isna(), column].eq(CATEGORICAL_FILL_VALUE).all()
            for column in categorical_cols
        ),
        "no_se_uso_moda": not OPERATION_FLAGS["mode_used"],
        "no_categorias_aprendidas_test": not OPERATION_FLAGS["test_categories_learned"],
        "categoricas_observadas_sin_cambios": observed_values_unchanged(X_train_before_b6, X_train_b6, categorical_cols) and observed_values_unchanged(X_test_before_b6, X_test_b6, categorical_cols),
        "missing_total_train_despues_0": int(X_train_b6.isna().sum().sum()) == 0,
        "missing_total_test_despues_0": int(X_test_b6.isna().sum().sum()) == 0,
        "sin_scaling": not OPERATION_FLAGS["scaling"],
        "sin_encoding": not OPERATION_FLAGS["encoding"],
        "sin_discretizacion": not OPERATION_FLAGS["discretization"],
        "sin_outlier_treatment": not OPERATION_FLAGS["outlier_treatment"],
        "sin_smote": not OPERATION_FLAGS["smote"],
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "test_used_for_fit_false": TEST_USED_FOR_FIT is False and not OPERATION_FLAGS["test_used_for_fit"],
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "pickles_recargables_y_alineados": (
            pd.read_pickle(X_TRAIN_PATH).index.equals(y_train.index)
            and pd.read_pickle(X_TEST_PATH).index.equals(y_test.index)
            and pd.read_pickle(Y_TRAIN_PATH).equals(y_train)
            and pd.read_pickle(Y_TEST_PATH).equals(y_test)
        ),
    }

    print("=== FRONTERA B5 RECONSTRUIDA ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Shape crudo: {df.shape}")
    print(f"Universo: {len(df_target)}")
    print(f"Train: {len(X_train_raw)}")
    print(f"Test: {len(X_test_raw)}")
    print(f"Features entrada: {len(feature_cols)}")
    print(f"random_state heredado B5: {b5_manifest['random_state']}")
    print(f"Train membership SHA-256: {membership_hash(train_assignment['record'])}")
    print(f"Test membership SHA-256: {membership_hash(test_assignment['record'])}")
    print("Membership e índices: coincidencia exacta con B5")
    print("Split reejecutado: False")
    print(
        "Warnings de lectura: "
        + (
            "; ".join(f"{item.category.__name__}: {item.message}" for item in read_warnings)
            if read_warnings
            else "ninguno"
        )
    )

    print("\n=== ANTES DE IMPUTAR ===")
    print(f"Train shape: {X_train_before_b6.shape}")
    print(f"Train NaN total: {int(X_train_before_b6.isna().sum().sum())}")
    print(f"Train q4 NaN: {int(train_missing_q4.sum())}")
    print(f"Train q5 NaN: {int(train_missing_q5.sum())}")
    print(f"Train categóricas NaN: {train_categorical_missing_before}")
    print(f"Test shape: {X_test_before_b6.shape}")
    print(f"Test NaN total: {int(X_test_before_b6.isna().sum().sum())}")
    print(f"Test q4 NaN: {int(test_missing_q4.sum())}")
    print(f"Test q5 NaN: {int(test_missing_q5.sum())}")
    print(f"Test categóricas NaN: {test_categorical_missing_before}")

    print("\n=== q4/q5 E INDICADOR ===")
    print(f"Mediana q4 aprendida en train: {median_q4_train}")
    print(f"Mediana q5 aprendida en train: {median_q5_train}")
    print(f"q4 imputados train/test: {int(train_missing_q4.sum())}/{int(test_missing_q4.sum())}")
    print(f"q5 imputados train/test: {int(train_missing_q5.sum())}/{int(test_missing_q5.sum())}")
    print(indicator_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    missing_categorical = categorical_summary.loc[
        categorical_summary["train_missing_before"].gt(0)
        | categorical_summary["test_missing_before"].gt(0)
    ].copy()
    missing_categorical["total_cells_replaced"] = (
        missing_categorical["train_cells_replaced"]
        + missing_categorical["test_cells_replaced"]
    )
    top_categorical = missing_categorical.sort_values(
        ["total_cells_replaced", "column"], ascending=[False, True]
    ).head(10)
    print("\n=== CATEGÓRICAS ===")
    print(f"Categóricas totales: {len(categorical_cols)}")
    print(f"Categóricas con missing: {len(missing_categorical)}")
    print(f"Celdas reemplazadas train: {int(categorical_summary['train_cells_replaced'].sum())}")
    print(f"Celdas reemplazadas test: {int(categorical_summary['test_cells_replaced'].sum())}")
    print(f"Categoría fija: {CATEGORICAL_FILL_VALUE!r}")
    print("Top 10 por celdas reemplazadas:")
    print(
        top_categorical[
            ["column", "train_cells_replaced", "test_cells_replaced", "total_cells_replaced"]
        ].to_string(index=False)
    )

    print("\n=== DESPUÉS DE B6 ===")
    print(f"X_train_b6 shape: {X_train_b6.shape}")
    print(f"X_test_b6 shape: {X_test_b6.shape}")
    print(f"Features salida: {X_train_b6.shape[1]}")
    print(f"Missing total train: {int(X_train_b6.isna().sum().sum())}")
    print(f"Missing total test: {int(X_test_b6.isna().sum().sum())}")

    print("\n=== FIT / TRANSFORM ===")
    print("FIT q4 median: TRAIN ONLY")
    print("FIT q5 median: TRAIN ONLY")
    print("q4q5_faltaba: FIXED RULE")
    print("sin_dato: FIXED RULE")
    print("TEST_USED_FOR_FIT = False")
    print("TRANSFORM medianas train: TRAIN + TEST")
    print("TRANSFORM reglas fijas: TRAIN + TEST")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B6 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B6 detenido; checks fallidos: {failed}")
    print(f"Resultado: {sum(checks.values())}/{len(checks)} checks OK. B6 finalizado; no se ejecutó B7.")


if __name__ == "__main__":
    main()
