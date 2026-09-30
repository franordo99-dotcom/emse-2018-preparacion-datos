from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from b1_target_emse import TARGET_NAME


BASE_DIR = Path(__file__).resolve().parent
B5_MANIFEST_PATH = BASE_DIR / "b5_split_manifest.json"
B6_MANIFEST_PATH = BASE_DIR / "b6_imputation_manifest.json"
B6_FEATURE_INVENTORY_PATH = BASE_DIR / "b6_feature_inventory.csv"
B6_X_TRAIN_PATH = BASE_DIR / "b6_X_train.pkl"
B6_X_TEST_PATH = BASE_DIR / "b6_X_test.pkl"
B6_Y_TRAIN_PATH = BASE_DIR / "b6_y_train.pkl"
B6_Y_TEST_PATH = BASE_DIR / "b6_y_test.pkl"

SCALER_PARAMS_PATH = BASE_DIR / "b7_scaler_params.csv"
SCALING_SUMMARY_PATH = BASE_DIR / "b7_scaling_summary.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b7_feature_inventory.csv"
MANIFEST_PATH = BASE_DIR / "b7_scaling_manifest.json"
X_TRAIN_PATH = BASE_DIR / "b7_X_train.pkl"
X_TEST_PATH = BASE_DIR / "b7_X_test.pkl"
Y_TRAIN_PATH = BASE_DIR / "b7_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b7_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_FEATURE_N = 149
SCALED_COLS = ["q4", "q5"]
INDICATOR_COL = "q4q5_faltaba"
DDOF = 0
FIT_PARTITION = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
OUTLIER_POLICY = "RETENER_Y_MARCAR_SIN_MODIFICAR"
FUTURE_IMC_POLICY = "FIT_STANDARD_SCALER_ON_TRAIN_ONLY_IF_IMC_CREATED_IN_B9"
TOLERANCE = 1e-10

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
    "imputation_reexecuted": False,
    "test_used_for_fit": False,
    "categorical_scaling": False,
    "indicator_scaling": False,
    "encoding": False,
    "discretization": False,
    "feature_engineering": False,
    "imc_created": False,
    "outlier_treatment": False,
    "smote": False,
    "pca": False,
    "feature_selection": False,
    "model_training": False,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_count(frame: pd.DataFrame, columns: list[str], value: object) -> int:
    return sum(int(frame[column].eq(value).sum()) for column in columns)


def scale_train_only(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_scaled = train.copy(deep=True)
    test_scaled = test.copy(deep=True)

    if importlib.util.find_spec("sklearn") is not None:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train[SCALED_COLS])
        train_scaled.loc[:, SCALED_COLS] = scaler.transform(train[SCALED_COLS])
        test_scaled.loc[:, SCALED_COLS] = scaler.transform(test[SCALED_COLS])
        means = pd.Series(scaler.mean_, index=SCALED_COLS)
        variances = pd.Series(scaler.var_, index=SCALED_COLS)
        scales = pd.Series(scaler.scale_, index=SCALED_COLS)
        engine = "sklearn.preprocessing.StandardScaler"
    else:
        means = train[SCALED_COLS].mean()
        variances = train[SCALED_COLS].var(ddof=DDOF)
        scales = variances.pow(0.5)
        scales = scales.mask(scales.eq(0), 1.0)
        train_scaled.loc[:, SCALED_COLS] = (
            train[SCALED_COLS] - means
        ) / scales
        test_scaled.loc[:, SCALED_COLS] = (
            test[SCALED_COLS] - means
        ) / scales
        engine = "pandas_manual_sklearn_StandardScaler_equivalent"

    params = pd.DataFrame(
        [
            {
                "column": column,
                "fit_partition": "train",
                "mean_train": float(means.loc[column]),
                "variance_train": float(variances.loc[column]),
                "scale_train": float(scales.loc[column]),
                "ddof": DDOF,
                "engine": engine,
            }
            for column in SCALED_COLS
        ]
    )
    return train_scaled, test_scaled, params, engine


def build_scaling_summary(
    before_partitions: dict[str, pd.DataFrame],
    after_partitions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for partition in ("train", "test"):
        before = before_partitions[partition]
        after = after_partitions[partition]
        for column in SCALED_COLS:
            rows.append(
                {
                    "partition": partition,
                    "column": column,
                    "mean_before": float(before[column].mean()),
                    "std_before_ddof0": float(before[column].std(ddof=DDOF)),
                    "min_before": float(before[column].min()),
                    "max_before": float(before[column].max()),
                    "mean_after": float(after[column].mean()),
                    "std_after_ddof0": float(after[column].std(ddof=DDOF)),
                    "min_after": float(after[column].min()),
                    "max_after": float(after[column].max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b5_manifest = json.loads(B5_MANIFEST_PATH.read_text(encoding="utf-8"))
    b6_manifest = json.loads(B6_MANIFEST_PATH.read_text(encoding="utf-8"))
    b6_feature_inventory = pd.read_csv(
        B6_FEATURE_INVENTORY_PATH, encoding="utf-8-sig"
    )

    source_paths = [
        B6_X_TRAIN_PATH,
        B6_X_TEST_PATH,
        B6_Y_TRAIN_PATH,
        B6_Y_TEST_PATH,
    ]
    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    expected_source_hashes = {
        name: data["sha256"] for name, data in b6_manifest["output_files"].items()
    }
    source_checks = {
        "source_hashes_match_b6": source_hashes == expected_source_hashes,
        "b6_source_block": b6_manifest["block"] == "B6",
        "b6_test_not_used_for_fit": b6_manifest["TEST_USED_FOR_FIT"] is False,
        "b6_test_frozen": b6_manifest["TEST_FROZEN_AFTER_B5"] is True,
        "b5_test_frozen": b5_manifest["TEST_FROZEN_AFTER_B5"] is True,
    }
    if not all(source_checks.values()):
        failed = [name for name, passed in source_checks.items() if not passed]
        raise AssertionError(f"Entrada B6 inválida; B7 detenido: {failed}")

    X_train_b6 = pd.read_pickle(B6_X_TRAIN_PATH)
    X_test_b6 = pd.read_pickle(B6_X_TEST_PATH)
    y_train_b6 = pd.read_pickle(B6_Y_TRAIN_PATH)
    y_test_b6 = pd.read_pickle(B6_Y_TEST_PATH)

    prefit_checks = {
        "train_44440": len(X_train_b6) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b6) == EXPECTED_TEST_N,
        "train_149_features": X_train_b6.shape[1] == EXPECTED_FEATURE_N,
        "test_149_features": X_test_b6.shape[1] == EXPECTED_FEATURE_N,
        "train_X_y_aligned": X_train_b6.index.equals(y_train_b6.index),
        "test_X_y_aligned": X_test_b6.index.equals(y_test_b6.index),
        "train_test_indices_disjoint": set(X_train_b6.index).isdisjoint(
            set(X_test_b6.index)
        ),
        "train_missing_0": int(X_train_b6.isna().sum().sum()) == 0,
        "test_missing_0": int(X_test_b6.isna().sum().sum()) == 0,
        "scaled_columns_present": set(SCALED_COLS).issubset(X_train_b6.columns),
        "indicator_present": INDICATOR_COL in X_train_b6.columns,
        "feature_inventory_149": len(b6_feature_inventory) == EXPECTED_FEATURE_N,
    }
    if not all(prefit_checks.values()):
        failed = [name for name, passed in prefit_checks.items() if not passed]
        raise AssertionError(f"Precondiciones B7 inválidas; B7 detenido: {failed}")

    X_train_before_b7 = X_train_b6.copy(deep=True)
    X_test_before_b7 = X_test_b6.copy(deep=True)
    y_train = y_train_b6.copy(deep=True)
    y_test = y_test_b6.copy(deep=True)

    X_train_b7, X_test_b7, scaler_params, engine = scale_train_only(
        X_train_before_b7, X_test_before_b7
    )
    scaling_summary = build_scaling_summary(
        {"train": X_train_before_b7, "test": X_test_before_b7},
        {"train": X_train_b7, "test": X_test_b7},
    )

    nonscaled_cols = [
        column for column in X_train_before_b7.columns if column not in SCALED_COLS
    ]
    categorical_cols = [
        column for column in nonscaled_cols if column != INDICATOR_COL
    ]
    train_sin_dato_before = value_count(
        X_train_before_b7, categorical_cols, "sin_dato"
    )
    test_sin_dato_before = value_count(
        X_test_before_b7, categorical_cols, "sin_dato"
    )
    train_sin_dato_after = value_count(X_train_b7, categorical_cols, "sin_dato")
    test_sin_dato_after = value_count(X_test_b7, categorical_cols, "sin_dato")

    b7_feature_inventory = b6_feature_inventory.copy()
    b7_feature_inventory["scaled_b7"] = b7_feature_inventory["column"].isin(
        SCALED_COLS
    )
    b7_feature_inventory["b7_action"] = b7_feature_inventory["column"].map(
        lambda column: (
            "standard_scaler_train_fit" if column in SCALED_COLS else "unchanged"
        )
    )

    scaler_params.to_csv(SCALER_PARAMS_PATH, index=False, encoding="utf-8-sig")
    scaling_summary.to_csv(SCALING_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    b7_feature_inventory.to_csv(
        FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig"
    )

    X_train_b7.to_pickle(X_TRAIN_PATH)
    X_test_b7.to_pickle(X_TEST_PATH)
    y_train.to_pickle(Y_TRAIN_PATH)
    y_test.to_pickle(Y_TEST_PATH)

    output_paths = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    manifest = {
        "block": "B7",
        "source_block": "B6",
        "source_manifest": B6_MANIFEST_PATH.name,
        "source_output_hashes": source_hashes,
        "train_n": len(X_train_b7),
        "test_n": len(X_test_b7),
        "input_feature_n": X_train_before_b7.shape[1],
        "output_feature_n": X_train_b7.shape[1],
        "scaler": "StandardScaler",
        "scaler_semantics": "sklearn_StandardScaler_equivalent",
        "scaled_columns": SCALED_COLS,
        "FIT_PARTITION": FIT_PARTITION,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "TRANSFORM_PARTITIONS": ["train", "test"],
        "engine": engine,
        "ddof": DDOF,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "split_reexecuted": False,
        "outlier_policy_inherited": OUTLIER_POLICY,
        "outlier_scaling_limitation": (
            "q5 mean and scale may be influenced by its retained right tail and extremes"
        ),
        "future_imc_policy": FUTURE_IMC_POLICY,
        "output_files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    expected_artifacts = [
        SCALER_PARAMS_PATH,
        SCALING_SUMMARY_PATH,
        FEATURE_INVENTORY_PATH,
        MANIFEST_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    params_by_column = scaler_params.set_index("column")
    train_after = scaling_summary.loc[
        scaling_summary["partition"].eq("train")
    ].set_index("column")
    scaled_feature_set = set(
        b7_feature_inventory.loc[b7_feature_inventory["scaled_b7"], "column"]
    )

    checks = {
        "train_44440": len(X_train_b7) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b7) == EXPECTED_TEST_N,
        "indices_train_iguales_B6": X_train_b7.index.equals(X_train_b6.index),
        "indices_test_iguales_B6": X_test_b7.index.equals(X_test_b6.index),
        "y_train_test_iguales_B6": y_train.equals(y_train_b6)
        and y_test.equals(y_test_b6),
        "test_congelado": TEST_FROZEN_AFTER_B5
        and b5_manifest["TEST_FROZEN_AFTER_B5"]
        and b6_manifest["TEST_FROZEN_AFTER_B5"],
        "no_nuevo_split": not OPERATION_FLAGS["split_executed"],
        "input_train_149": X_train_before_b7.shape[1] == EXPECTED_FEATURE_N,
        "input_test_149": X_test_before_b7.shape[1] == EXPECTED_FEATURE_N,
        "output_train_149": X_train_b7.shape[1] == EXPECTED_FEATURE_N,
        "output_test_149": X_test_b7.shape[1] == EXPECTED_FEATURE_N,
        "missing_train_antes_0": int(X_train_before_b7.isna().sum().sum()) == 0,
        "missing_test_antes_0": int(X_test_before_b7.isna().sum().sum()) == 0,
        "missing_train_despues_0": int(X_train_b7.isna().sum().sum()) == 0,
        "missing_test_despues_0": int(X_test_b7.isna().sum().sum()) == 0,
        "columnas_escaladas_exactamente_q4_q5": scaled_feature_set == set(SCALED_COLS),
        "media_q4_aprendida_solo_train": params_by_column.loc["q4", "fit_partition"]
        == "train"
        and params_by_column.loc["q4", "mean_train"]
        == X_train_before_b7["q4"].mean(),
        "scale_q4_aprendido_solo_train": params_by_column.loc["q4", "fit_partition"]
        == "train"
        and params_by_column.loc["q4", "scale_train"]
        == X_train_before_b7["q4"].std(ddof=DDOF),
        "media_q5_aprendida_solo_train": params_by_column.loc["q5", "fit_partition"]
        == "train"
        and params_by_column.loc["q5", "mean_train"]
        == X_train_before_b7["q5"].mean(),
        "scale_q5_aprendido_solo_train": params_by_column.loc["q5", "fit_partition"]
        == "train"
        and params_by_column.loc["q5", "scale_train"]
        == X_train_before_b7["q5"].std(ddof=DDOF),
        "test_used_for_fit_false": TEST_USED_FOR_FIT is False
        and not OPERATION_FLAGS["test_used_for_fit"],
        "ddof_0": DDOF == 0 and scaler_params["ddof"].eq(0).all(),
        "q4_train_mean_cero": abs(train_after.loc["q4", "mean_after"]) < TOLERANCE,
        "q4_train_std_uno": abs(train_after.loc["q4", "std_after_ddof0"] - 1) < TOLERANCE,
        "q5_train_mean_cero": abs(train_after.loc["q5", "mean_after"]) < TOLERANCE,
        "q5_train_std_uno": abs(train_after.loc["q5", "std_after_ddof0"] - 1) < TOLERANCE,
        "indicador_no_cambio": X_train_b7[INDICATOR_COL].equals(
            X_train_before_b7[INDICATOR_COL]
        )
        and X_test_b7[INDICATOR_COL].equals(X_test_before_b7[INDICATOR_COL])
        and set(X_train_b7[INDICATOR_COL].unique()) == {0, 1},
        "categoricas_no_cambiaron": X_train_b7[categorical_cols].equals(
            X_train_before_b7[categorical_cols]
        )
        and X_test_b7[categorical_cols].equals(X_test_before_b7[categorical_cols]),
        "sin_dato_preservado": train_sin_dato_after == train_sin_dato_before
        and test_sin_dato_after == test_sin_dato_before,
        "q50_q51_preservadas": X_train_b7[["q50", "q51"]].equals(
            X_train_before_b7[["q50", "q51"]]
        )
        and X_test_b7[["q50", "q51"]].equals(
            X_test_before_b7[["q50", "q51"]]
        ),
        "unknown_11_preservadas": X_train_b7[UNKNOWN_CATEGORICAL_COLS].equals(
            X_train_before_b7[UNKNOWN_CATEGORICAL_COLS]
        )
        and X_test_b7[UNKNOWN_CATEGORICAL_COLS].equals(
            X_test_before_b7[UNKNOWN_CATEGORICAL_COLS]
        ),
        "leakage_sigue_fuera": not (set(HARD_LEAKAGE_COLS) & set(X_train_b7.columns)),
        "metadata_sigue_fuera": not (set(METADATA_COLS) & set(X_train_b7.columns)),
        "target_sigue_fuera": TARGET_NAME not in X_train_b7.columns,
        "sin_encoding": not OPERATION_FLAGS["encoding"],
        "sin_discretizacion": not OPERATION_FLAGS["discretization"],
        "sin_feature_engineering": not OPERATION_FLAGS["feature_engineering"],
        "sin_imc": not OPERATION_FLAGS["imc_created"]
        and not any("imc" in column.lower() for column in X_train_b7.columns),
        "sin_outlier_treatment": not OPERATION_FLAGS["outlier_treatment"],
        "sin_smote": not OPERATION_FLAGS["smote"],
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "pickles_recargables_y_alineados": pd.read_pickle(X_TRAIN_PATH).index.equals(
            pd.read_pickle(Y_TRAIN_PATH).index
        )
        and pd.read_pickle(X_TEST_PATH).index.equals(pd.read_pickle(Y_TEST_PATH).index)
        and pd.read_pickle(X_TRAIN_PATH)[nonscaled_cols].equals(
            X_train_before_b7[nonscaled_cols]
        )
        and pd.read_pickle(X_TEST_PATH)[nonscaled_cols].equals(
            X_test_before_b7[nonscaled_cols]
        ),
        "hashes_fuente_B6_verificados": all(source_checks.values()),
    }

    print("=== FRONTERA B6 CANÓNICA ===")
    print(f"Train: {len(X_train_b6)}")
    print(f"Test: {len(X_test_b6)}")
    print(f"Features: {X_train_b6.shape[1]}")
    print("Índices X/y: alineados")
    print("Hashes de los cuatro pickles B6: verificados")
    print("Split reejecutado: False")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")
    print(f"B6 TEST_USED_FOR_FIT = {b6_manifest['TEST_USED_FOR_FIT']}")

    print("\n=== STANDARD SCALER ===")
    print(f"Engine: {engine}")
    print("scaler_semantics: sklearn_StandardScaler_equivalent")
    print(f"ddof: {DDOF}")
    print(
        scaler_params.to_string(index=False, float_format=lambda value: f"{value:.12f}")
    )

    print("\n=== TRAIN DESPUÉS ===")
    print(
        scaling_summary.loc[
            scaling_summary["partition"].eq("train"),
            ["column", "mean_after", "std_after_ddof0", "min_after", "max_after"],
        ].to_string(index=False, float_format=lambda value: f"{value:.12f}")
    )

    print("\n=== TEST DESPUÉS ===")
    print(
        scaling_summary.loc[
            scaling_summary["partition"].eq("test"),
            ["column", "mean_after", "std_after_ddof0", "min_after", "max_after"],
        ].to_string(index=False, float_format=lambda value: f"{value:.12f}")
    )
    print("Test no debe quedar exactamente centrado en 0 ni con std=1: no participó del fit.")

    print("\n=== PRESERVACIÓN ===")
    print(f"q4q5_faltaba intacto: {checks['indicador_no_cambio']}")
    print(f"Categóricas intactas: {checks['categoricas_no_cambiaron']}")
    print(f"sin_dato train preservados: {train_sin_dato_after}")
    print(f"sin_dato test preservados: {test_sin_dato_after}")
    print(f"Features train/test: {X_train_b7.shape[1]}/{X_test_b7.shape[1]}")
    print(
        f"Missing total train/test: {int(X_train_b7.isna().sum().sum())}/"
        f"{int(X_test_b7.isna().sum().sum())}"
    )

    print("\n=== FIT / TRANSFORM ===")
    print("FIT StandardScaler(q4,q5): TRAIN ONLY")
    print("TRANSFORM StandardScaler: TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")
    print("q4q5_faltaba: NOT SCALED")
    print("CATEGORICAL_FEATURES: NOT SCALED")

    print("\n=== OUTLIERS ===")
    print(f"OUTLIERS_POLICY_B4 = {OUTLIER_POLICY}")
    print("No hubo eliminación, winsorización, clipping ni outlier→NaN.")
    print("Limitación: la cola derecha y extremos retenidos de q5 pueden influir en su media y escala.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B7 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B7 detenido; checks fallidos: {failed}")
    print(f"Resultado: {sum(checks.values())}/{len(checks)} checks OK. B7 finalizado; no se ejecutó B8.")


if __name__ == "__main__":
    main()
