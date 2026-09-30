from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from b1_target_emse import TARGET_NAME


BASE_DIR = Path(__file__).resolve().parent
B5_MANIFEST_PATH = BASE_DIR / "b5_split_manifest.json"
B6_MANIFEST_PATH = BASE_DIR / "b6_imputation_manifest.json"
B8_MANIFEST_PATH = BASE_DIR / "b8_encoding_manifest.json"
B8_FEATURE_INVENTORY_PATH = BASE_DIR / "b8_feature_inventory.csv"
B6_X_TRAIN_PATH = BASE_DIR / "b6_X_train.pkl"
B6_X_TEST_PATH = BASE_DIR / "b6_X_test.pkl"
B6_Y_TRAIN_PATH = BASE_DIR / "b6_y_train.pkl"
B6_Y_TEST_PATH = BASE_DIR / "b6_y_test.pkl"
B8_X_TRAIN_PATH = BASE_DIR / "b8_X_train.pkl"
B8_X_TEST_PATH = BASE_DIR / "b8_X_test.pkl"
B8_Y_TRAIN_PATH = BASE_DIR / "b8_y_train.pkl"
B8_Y_TEST_PATH = BASE_DIR / "b8_y_test.pkl"

RAW_SUMMARY_PATH = BASE_DIR / "b9_imc_raw_summary.csv"
BY_INDICATOR_PATH = BASE_DIR / "b9_imc_by_missing_indicator.csv"
SCALER_PARAMS_PATH = BASE_DIR / "b9_imc_scaler_params.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b9_feature_inventory.csv"
ENGINEERING_SUMMARY_PATH = BASE_DIR / "b9_feature_engineering_summary.csv"
MANIFEST_PATH = BASE_DIR / "b9_feature_engineering_manifest.json"
X_TRAIN_PATH = BASE_DIR / "b9_X_train.pkl"
X_TEST_PATH = BASE_DIR / "b9_X_test.pkl"
Y_TRAIN_PATH = BASE_DIR / "b9_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b9_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
IMC_FEATURE = "imc"
INDICATOR_COL = "q4q5_faltaba"
IMC_SOURCE = "B6_IMPUTED_UNSCALED_Q4_Q5"
IMC_FORMULA = "q5_kg / q4_m**2"
IMC_FORMULA_TYPE = "FIXED_DETERMINISTIC_RULE"
IMC_SCALER = "StandardScaler_equivalent"
IMC_DISCRETIZATION_B9 = "NONE"
FIT_IMC_SCALER = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
DDOF = 0
ENGINE = "pandas_manual_sklearn_StandardScaler_equivalent"
OUTLIER_POLICY = "RETENER_Y_MARCAR_SIN_MODIFICAR"
Q50_Q51_STATUS = "SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE"
UNKNOWN_AUDIT_STATUS = "SEMANTIC_REVIEW_APPLIED"
PATCH_REASON = "ORDINAL_SEMANTIC_REVIEW"
PATCH_SOURCE = "REVISION_SEMANTICA_ORDINALES"
B10_STATUS = "SMOTE_DEMONSTRATION_ONLY"
DOWNSTREAM_TRAIN_SOURCE = "B9_ORIGINAL_TRAIN"
DOWNSTREAM_SMOTE_SOURCE_ALLOWED = False
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
    "imputation_executed": False,
    "reencoding": False,
    "q4_q5_rescaled": False,
    "imc_separate_imputation": False,
    "discretization": False,
    "outlier_treatment": False,
    "smote": False,
    "pca": False,
    "feature_selection": False,
    "model_training": False,
    "test_used_for_fit": False,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_hashes(paths: list[Path], manifest: dict[str, object]) -> dict[str, str]:
    actual = {path.name: file_sha256(path) for path in paths}
    expected = {
        name: data["sha256"]
        for name, data in manifest["output_files"].items()
    }
    if actual != expected:
        raise AssertionError(f"Hashes fuente inválidos: actual={actual}, expected={expected}")
    return actual


def describe_imc(partition: str, values: pd.Series) -> dict[str, object]:
    finite = np.isfinite(values.to_numpy(dtype="float64"))
    return {
        "partition": partition,
        "n": len(values),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std_ddof0": float(values.std(ddof=DDOF)),
        "nan_n": int(values.isna().sum()),
        "inf_n": int((~finite & values.notna().to_numpy()).sum()),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b5_manifest = json.loads(B5_MANIFEST_PATH.read_text(encoding="utf-8"))
    b6_manifest = json.loads(B6_MANIFEST_PATH.read_text(encoding="utf-8"))
    b8_manifest = json.loads(B8_MANIFEST_PATH.read_text(encoding="utf-8"))
    b8_inventory = pd.read_csv(B8_FEATURE_INVENTORY_PATH, encoding="utf-8-sig")
    expected_input_feature_n = int(b8_manifest["output_feature_n"])
    expected_output_feature_n = expected_input_feature_n + 1
    unknown_audit_status = b8_manifest["unknown_categorical_audit_status"]

    b6_paths = [
        B6_X_TRAIN_PATH,
        B6_X_TEST_PATH,
        B6_Y_TRAIN_PATH,
        B6_Y_TEST_PATH,
    ]
    b8_paths = [
        B8_X_TRAIN_PATH,
        B8_X_TEST_PATH,
        B8_Y_TRAIN_PATH,
        B8_Y_TEST_PATH,
    ]
    b6_hashes = verify_manifest_hashes(b6_paths, b6_manifest)
    b8_hashes = verify_manifest_hashes(b8_paths, b8_manifest)

    X_train_b6 = pd.read_pickle(B6_X_TRAIN_PATH)
    X_test_b6 = pd.read_pickle(B6_X_TEST_PATH)
    y_train_b6 = pd.read_pickle(B6_Y_TRAIN_PATH)
    y_test_b6 = pd.read_pickle(B6_Y_TEST_PATH)
    X_train_b8 = pd.read_pickle(B8_X_TRAIN_PATH)
    X_test_b8 = pd.read_pickle(B8_X_TEST_PATH)
    y_train_b8 = pd.read_pickle(B8_Y_TRAIN_PATH)
    y_test_b8 = pd.read_pickle(B8_Y_TEST_PATH)

    boundary_checks = {
        "train_n": len(X_train_b6) == len(X_train_b8) == EXPECTED_TRAIN_N,
        "test_n": len(X_test_b6) == len(X_test_b8) == EXPECTED_TEST_N,
        "train_indices": X_train_b6.index.equals(X_train_b8.index),
        "test_indices": X_test_b6.index.equals(X_test_b8.index),
        "train_y": y_train_b6.equals(y_train_b8),
        "test_y": y_test_b6.equals(y_test_b8),
        "train_X_y": X_train_b8.index.equals(y_train_b8.index),
        "test_X_y": X_test_b8.index.equals(y_test_b8.index),
        "same_b5_membership": (
            b6_manifest["source_train_record_membership_sha256"]
            == b5_manifest["train_record_membership_sha256"]
            and b6_manifest["source_test_record_membership_sha256"]
            == b5_manifest["test_record_membership_sha256"]
        ),
        "test_frozen": (
            b5_manifest["TEST_FROZEN_AFTER_B5"]
            and b6_manifest["TEST_FROZEN_AFTER_B5"]
            and b8_manifest["TEST_FROZEN_AFTER_B5"]
        ),
        "b8_imc_source_contract": b8_manifest["IMC_SOURCE_FOR_B9"] == IMC_SOURCE,
        "b8_features_expected": X_train_b8.shape[1]
        == X_test_b8.shape[1]
        == expected_input_feature_n,
        "b8_inventory_expected": len(b8_inventory) == expected_input_feature_n,
        "b8_patch_contract": b8_manifest["PATCH_REASON"] == PATCH_REASON
        and b8_manifest["PATCH_SOURCE"] == PATCH_SOURCE,
    }
    if not all(boundary_checks.values()):
        failed = [name for name, passed in boundary_checks.items() if not passed]
        raise AssertionError(f"Frontera B6/B8 inválida; B9 detenido: {failed}")

    physical_checks = {
        "q4_no_nan": not X_train_b6["q4"].isna().any()
        and not X_test_b6["q4"].isna().any(),
        "q5_no_nan": not X_train_b6["q5"].isna().any()
        and not X_test_b6["q5"].isna().any(),
        "q4_positive": X_train_b6["q4"].gt(0).all()
        and X_test_b6["q4"].gt(0).all(),
        "q5_positive": X_train_b6["q5"].gt(0).all()
        and X_test_b6["q5"].gt(0).all(),
        "indicator_equal": X_train_b6[INDICATOR_COL].equals(
            X_train_b8[INDICATOR_COL]
        )
        and X_test_b6[INDICATOR_COL].equals(X_test_b8[INDICATOR_COL]),
    }
    if not all(physical_checks.values()):
        failed = [name for name, passed in physical_checks.items() if not passed]
        raise AssertionError(f"Fuente física B6 inválida; B9 detenido: {failed}")

    imc_raw_train = X_train_b6["q5"] / X_train_b6["q4"].pow(2)
    imc_raw_test = X_test_b6["q5"] / X_test_b6["q4"].pow(2)
    imc_raw_train.name = "imc_raw"
    imc_raw_test.name = "imc_raw"

    raw_summary = pd.DataFrame(
        [
            describe_imc("train", imc_raw_train),
            describe_imc("test", imc_raw_test),
        ]
    )
    by_indicator_rows: list[dict[str, object]] = []
    for partition, values, indicator in [
        ("train", imc_raw_train, X_train_b6[INDICATOR_COL]),
        ("test", imc_raw_test, X_test_b6[INDICATOR_COL]),
    ]:
        for indicator_value in (0, 1):
            group = values.loc[indicator.eq(indicator_value)]
            row = describe_imc(partition, group)
            row["q4q5_faltaba"] = indicator_value
            by_indicator_rows.append(row)
    by_indicator = pd.DataFrame(by_indicator_rows)[
        [
            "partition",
            "q4q5_faltaba",
            "n",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "std_ddof0",
            "nan_n",
            "inf_n",
        ]
    ]

    mean_imc_train = float(imc_raw_train.mean())
    variance_imc_train = float(imc_raw_train.var(ddof=DDOF))
    scale_imc_train = float(np.sqrt(variance_imc_train))
    if not scale_imc_train > 0:
        raise AssertionError("IMC train tiene escala no positiva; B9 detenido.")
    imc_scaled_train = (imc_raw_train - mean_imc_train) / scale_imc_train
    imc_scaled_test = (imc_raw_test - mean_imc_train) / scale_imc_train

    scaler_params = pd.DataFrame(
        [
            {
                "feature": IMC_FEATURE,
                "source_q4": "B6 q4 imputed unscaled",
                "source_q5": "B6 q5 imputed unscaled",
                "formula": "q5 / q4**2",
                "fit_partition": "train",
                "mean_train": mean_imc_train,
                "variance_train": variance_imc_train,
                "scale_train": scale_imc_train,
                "ddof": DDOF,
                "engine": ENGINE,
                "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
            }
        ]
    )

    X_train_b9 = X_train_b8.copy(deep=True)
    X_test_b9 = X_test_b8.copy(deep=True)
    X_train_b9[IMC_FEATURE] = imc_scaled_train
    X_test_b9[IMC_FEATURE] = imc_scaled_test
    y_train = y_train_b8.copy(deep=True)
    y_test = y_test_b8.copy(deep=True)

    b9_inventory = b8_inventory.copy()
    b9_inventory["b9_action"] = "unchanged_from_B8"
    imc_inventory = pd.DataFrame(
        [
            {
                "output_column": IMC_FEATURE,
                "source_column": "B6:q4+B6:q5",
                "output_position": len(b9_inventory),
                "output_kind": "continuous_numeric",
                "encoding": "physical_formula_then_train_standard_scaler",
                "source_category": "",
                "audit_status": "CONFIRMED_B9",
                "b9_action": "new_feature_imc_scaled",
            }
        ]
    )
    b9_inventory = pd.concat([b9_inventory, imc_inventory], ignore_index=True)

    output_train_array = X_train_b9.to_numpy(dtype="float64")
    output_test_array = X_test_b9.to_numpy(dtype="float64")
    engineering_summary = pd.DataFrame(
        [
            {
                "input_feature_n": X_train_b8.shape[1],
                "output_feature_n": X_train_b9.shape[1],
                "new_feature": IMC_FEATURE,
                "imc_source": IMC_SOURCE,
                "imc_formula": IMC_FORMULA,
                "imc_formula_type": IMC_FORMULA_TYPE,
                "imc_scaler": IMC_SCALER,
                "fit_partition": FIT_IMC_SCALER,
                "test_used_for_fit": TEST_USED_FOR_FIT,
                "imc_discretization": IMC_DISCRETIZATION_B9,
                "train_imc_scaled_mean": float(imc_scaled_train.mean()),
                "train_imc_scaled_std_ddof0": float(imc_scaled_train.std(ddof=DDOF)),
                "test_imc_scaled_mean": float(imc_scaled_test.mean()),
                "test_imc_scaled_std_ddof0": float(imc_scaled_test.std(ddof=DDOF)),
                "missing_train_after": int(X_train_b9.isna().sum().sum()),
                "missing_test_after": int(X_test_b9.isna().sum().sum()),
                "inf_train_after": int(np.isinf(output_train_array).sum()),
                "inf_test_after": int(np.isinf(output_test_array).sum()),
                "q50_q51_status": Q50_Q51_STATUS,
                "unknown_11_status": unknown_audit_status,
            }
        ]
    )

    raw_summary.to_csv(RAW_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    by_indicator.to_csv(BY_INDICATOR_PATH, index=False, encoding="utf-8-sig")
    scaler_params.to_csv(SCALER_PARAMS_PATH, index=False, encoding="utf-8-sig")
    b9_inventory.to_csv(FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig")
    engineering_summary.to_csv(
        ENGINEERING_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )
    X_train_b9.to_pickle(X_TRAIN_PATH)
    X_test_b9.to_pickle(X_TEST_PATH)
    y_train.to_pickle(Y_TRAIN_PATH)
    y_test.to_pickle(Y_TEST_PATH)

    output_paths = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    manifest = {
        "block": "B9",
        "source_predictive_block": "B8",
        "physical_source_block": "B6",
        "source_predictive_hashes": b8_hashes,
        "physical_source_hashes": b6_hashes,
        "train_n": len(X_train_b9),
        "test_n": len(X_test_b9),
        "input_feature_n": X_train_b8.shape[1],
        "output_feature_n": X_train_b9.shape[1],
        "new_features": [IMC_FEATURE],
        "IMC_SOURCE": IMC_SOURCE,
        "IMC_FORMULA": IMC_FORMULA,
        "IMC_FORMULA_TYPE": IMC_FORMULA_TYPE,
        "IMC_SCALER": IMC_SCALER,
        "FIT_IMC_SCALER": FIT_IMC_SCALER,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "ddof": DDOF,
        "engine": ENGINE,
        "IMC_DISCRETIZATION_B9": IMC_DISCRETIZATION_B9,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "outlier_policy_inherited": OUTLIER_POLICY,
        "q50_q51_status": Q50_Q51_STATUS,
        "unknown_11_status": unknown_audit_status,
        "PATCH_REASON": PATCH_REASON,
        "PATCH_SOURCE": PATCH_SOURCE,
        "B10_STATUS": B10_STATUS,
        "DOWNSTREAM_TRAIN_SOURCE": DOWNSTREAM_TRAIN_SOURCE,
        "DOWNSTREAM_TRAIN_N": EXPECTED_TRAIN_N,
        "DOWNSTREAM_SMOTE_SOURCE_ALLOWED": DOWNSTREAM_SMOTE_SOURCE_ALLOWED,
        "output_files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    expected_artifacts = [
        RAW_SUMMARY_PATH,
        BY_INDICATOR_PATH,
        SCALER_PARAMS_PATH,
        FEATURE_INVENTORY_PATH,
        ENGINEERING_SUMMARY_PATH,
        MANIFEST_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]
    original_columns = X_train_b8.columns.tolist()
    new_train_features = set(X_train_b9.columns) - set(original_columns)
    new_test_features = set(X_test_b9.columns) - set(original_columns)
    unknown_output_cols = b8_inventory.loc[
        b8_inventory["source_column"].isin(UNKNOWN_CATEGORICAL_COLS),
        "output_column",
    ].tolist()
    exact_formula_train = X_train_b6["q5"] / X_train_b6["q4"].pow(2)
    exact_formula_test = X_test_b6["q5"] / X_test_b6["q4"].pow(2)

    checks = {
        "train_44440": len(X_train_b9) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b9) == EXPECTED_TEST_N,
        "indices_train_B6_B8_identicos": X_train_b6.index.equals(X_train_b8.index),
        "indices_test_B6_B8_identicos": X_test_b6.index.equals(X_test_b8.index),
        "y_B6_B8_identico": y_train_b6.equals(y_train_b8)
        and y_test_b6.equals(y_test_b8),
        "test_congelado": TEST_FROZEN_AFTER_B5 and boundary_checks["test_frozen"],
        "sin_split_nuevo": not OPERATION_FLAGS["split_executed"],
        "q4_B6_sin_nan": physical_checks["q4_no_nan"],
        "q5_B6_sin_nan": physical_checks["q5_no_nan"],
        "q4_B6_positiva": physical_checks["q4_positive"],
        "q5_B6_positiva": physical_checks["q5_positive"],
        "imc_desde_B6_no_B8": IMC_SOURCE == "B6_IMPUTED_UNSCALED_Q4_Q5"
        and b8_manifest["IMC_SOURCE_FOR_B9"] == IMC_SOURCE,
        "formula_exacta_q5_q4_cuadrado": imc_raw_train.equals(exact_formula_train)
        and imc_raw_test.equals(exact_formula_test),
        "imc_raw_sin_nan": not imc_raw_train.isna().any()
        and not imc_raw_test.isna().any(),
        "imc_raw_sin_inf": np.isfinite(imc_raw_train.to_numpy()).all()
        and np.isfinite(imc_raw_test.to_numpy()).all(),
        "formula_imc_sin_fit": IMC_FORMULA_TYPE == "FIXED_DETERMINISTIC_RULE",
        "mean_imc_train_only": scaler_params.loc[0, "fit_partition"] == "train"
        and scaler_params.loc[0, "mean_train"] == imc_raw_train.mean(),
        "scale_imc_train_only": scaler_params.loc[0, "fit_partition"] == "train"
        and scaler_params.loc[0, "scale_train"] == imc_raw_train.std(ddof=DDOF),
        "ddof_0": DDOF == 0 and scaler_params.loc[0, "ddof"] == 0,
        "test_used_for_fit_false": TEST_USED_FOR_FIT is False
        and not OPERATION_FLAGS["test_used_for_fit"],
        "input_B8_patch": X_train_b8.shape[1]
        == X_test_b8.shape[1]
        == expected_input_feature_n,
        "output_B9_patch": X_train_b9.shape[1]
        == X_test_b9.shape[1]
        == expected_output_feature_n,
        "entrada_nuevo_B8_patch": boundary_checks["b8_patch_contract"],
        "unica_nueva_feature_imc": new_train_features
        == new_test_features
        == {IMC_FEATURE},
        "mismas_columnas_train_test": X_train_b9.columns.equals(X_test_b9.columns),
        "mismo_orden_train_test": X_train_b9.columns.tolist()
        == X_test_b9.columns.tolist(),
        "imc_train_mean_cero": abs(float(imc_scaled_train.mean())) < TOLERANCE,
        "imc_train_std_uno": abs(float(imc_scaled_train.std(ddof=DDOF)) - 1)
        < TOLERANCE,
        "imc_sin_nan": not X_train_b9[IMC_FEATURE].isna().any()
        and not X_test_b9[IMC_FEATURE].isna().any(),
        "imc_sin_inf": np.isfinite(X_train_b9[IMC_FEATURE].to_numpy()).all()
        and np.isfinite(X_test_b9[IMC_FEATURE].to_numpy()).all(),
        "salida_sin_nan": int(X_train_b9.isna().sum().sum()) == 0
        and int(X_test_b9.isna().sum().sum()) == 0,
        "salida_sin_inf": np.isfinite(output_train_array).all()
        and np.isfinite(output_test_array).all(),
        "B8_train_all_features_intactas": X_train_b9[original_columns].equals(X_train_b8),
        "B8_test_all_features_intactas": X_test_b9[original_columns].equals(X_test_b8),
        "q4_intacta_B8": X_train_b9["q4"].equals(X_train_b8["q4"])
        and X_test_b9["q4"].equals(X_test_b8["q4"]),
        "q5_intacta_B8": X_train_b9["q5"].equals(X_train_b8["q5"])
        and X_test_b9["q5"].equals(X_test_b8["q5"]),
        "indicador_intacto": X_train_b9[INDICATOR_COL].equals(
            X_train_b8[INDICATOR_COL]
        )
        and X_test_b9[INDICATOR_COL].equals(X_test_b8[INDICATOR_COL]),
        "q50_q51_intactas": X_train_b9[["q50", "q51"]].equals(
            X_train_b8[["q50", "q51"]]
        )
        and X_test_b9[["q50", "q51"]].equals(X_test_b8[["q50", "q51"]]),
        "unknown_codificadas_intactas": X_train_b9[unknown_output_cols].equals(
            X_train_b8[unknown_output_cols]
        )
        and X_test_b9[unknown_output_cols].equals(X_test_b8[unknown_output_cols]),
        "y_intacto": y_train.equals(y_train_b8) and y_test.equals(y_test_b8),
        "leakage_fuera": not any(
            source in HARD_LEAKAGE_COLS
            or output in HARD_LEAKAGE_COLS
            or any(output.startswith(f"{blocked}__") for blocked in HARD_LEAKAGE_COLS)
            for source, output in zip(
                b9_inventory["source_column"],
                b9_inventory["output_column"],
                strict=True,
            )
        ),
        "metadata_fuera": not any(
            source in METADATA_COLS for source in b9_inventory["source_column"]
        ),
        "target_fuera": TARGET_NAME not in X_train_b9.columns,
        "texto_fuera": not any(
            source.startswith("texto_") for source in b9_inventory["source_column"]
        ),
        "sin_nueva_imputacion": not OPERATION_FLAGS["imputation_executed"]
        and not OPERATION_FLAGS["imc_separate_imputation"],
        "sin_reencoding": not OPERATION_FLAGS["reencoding"],
        "sin_discretizacion": IMC_DISCRETIZATION_B9 == "NONE"
        and not OPERATION_FLAGS["discretization"],
        "sin_outlier_treatment": not OPERATION_FLAGS["outlier_treatment"],
        "sin_smote": not OPERATION_FLAGS["smote"],
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "params_imc_persistidos": SCALER_PARAMS_PATH.exists(),
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "pickles_recargables_alineados": pd.read_pickle(X_TRAIN_PATH).index.equals(
            pd.read_pickle(Y_TRAIN_PATH).index
        )
        and pd.read_pickle(X_TEST_PATH).index.equals(pd.read_pickle(Y_TEST_PATH).index)
        and pd.read_pickle(X_TRAIN_PATH).columns.equals(
            pd.read_pickle(X_TEST_PATH).columns
        ),
        "hashes_fuente_B6_B8_verificados": bool(b6_hashes) and bool(b8_hashes),
    }

    print("=== FRONTERA B6 ↔ B8 ===")
    print(f"Train: {len(X_train_b8)}")
    print(f"Test: {len(X_test_b8)}")
    print("Índices B6/B8 train/test: idénticos")
    print("Targets B6/B8: idénticos")
    print("Hashes B6 y B8: verificados")
    print("Membership B5: consistente")
    print("Split reejecutado: False")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")

    print("\n=== FUENTE FÍSICA IMC ===")
    print("q4: B6 imputada, sin escalar, unidad metros")
    print("q5: B6 imputada, sin escalar, unidad kg")
    print(f"q4 rango train/test: {X_train_b6['q4'].min()}–{X_train_b6['q4'].max()} / {X_test_b6['q4'].min()}–{X_test_b6['q4'].max()}")
    print(f"q5 rango train/test: {X_train_b6['q5'].min()}–{X_train_b6['q5'].max()} / {X_test_b6['q5'].min()}–{X_test_b6['q5'].max()}")
    print(f"Fórmula: {IMC_FORMULA}")

    print("\n=== IMC RAW ===")
    print(raw_summary.to_string(index=False, float_format=lambda value: f"{value:.12f}"))
    print("\nIMC raw por q4q5_faltaba:")
    print(by_indicator.to_string(index=False, float_format=lambda value: f"{value:.12f}"))

    print("\n=== SCALER IMC ===")
    print(scaler_params.to_string(index=False, float_format=lambda value: f"{value:.12f}"))

    print("\n=== IMC ESCALADO ===")
    scaled_summary = pd.DataFrame(
        [
            {
                "partition": "train",
                "mean": imc_scaled_train.mean(),
                "std_ddof0": imc_scaled_train.std(ddof=DDOF),
                "min": imc_scaled_train.min(),
                "max": imc_scaled_train.max(),
            },
            {
                "partition": "test",
                "mean": imc_scaled_test.mean(),
                "std_ddof0": imc_scaled_test.std(ddof=DDOF),
                "min": imc_scaled_test.min(),
                "max": imc_scaled_test.max(),
            },
        ]
    )
    print(scaled_summary.to_string(index=False, float_format=lambda value: f"{value:.12f}"))
    print("Test no debe quedar exactamente en mean=0/std=1: no participó del fit.")

    print("\n=== SALIDA B9 ===")
    print(f"X_train_b9 shape: {X_train_b9.shape}")
    print(f"X_test_b9 shape: {X_test_b9.shape}")
    print(f"Única nueva feature: {sorted(new_train_features)}")
    print(f"Numeric only: {all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_b9.dtypes)}")
    print(f"Missing train/test: {int(X_train_b9.isna().sum().sum())}/{int(X_test_b9.isna().sum().sum())}")
    print(f"Inf train/test: {int(np.isinf(output_train_array).sum())}/{int(np.isinf(output_test_array).sum())}")

    print("\n=== FIT / TRANSFORM ===")
    print("IMC FORMULA: q5_kg / q4_m**2")
    print("IMC SOURCE: B6 IMPUTED UNSCALED q4/q5")
    print("IMC FORMULA FIT: NONE — FIXED RULE")
    print("FIT StandardScaler(imc): TRAIN ONLY")
    print("TRANSFORM StandardScaler(imc): TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")
    print("IMC_DISCRETIZATION_B9 = NONE")

    print("\n=== PRESERVACIÓN ===")
    print(f"{len(original_columns)} features B8 train intactas: {checks['B8_train_all_features_intactas']}")
    print(f"{len(original_columns)} features B8 test intactas: {checks['B8_test_all_features_intactas']}")
    print(f"q4/q5/indicador intactos: {checks['q4_intacta_B8'] and checks['q5_intacta_B8'] and checks['indicador_intacto']}")
    print(f"q50/q51 status: {Q50_Q51_STATUS}; encoding ordinal auditado en patch")
    print(f"Unknown 11 status: {unknown_audit_status}")
    print(f"OUTLIERS_POLICY_B4 = {OUTLIER_POLICY}")
    print("No hubo clipping, winsorización, eliminación ni outlier→NaN.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B9 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B9 detenido; checks fallidos: {failed}")
    print(f"Resultado: {sum(checks.values())}/{len(checks)} checks OK. B9 regenerado; no se ejecutó B10 ni B11.")
    print("SEMANTIC_PATCH_APPLIED = True")
    print(f"B10_STATUS = {B10_STATUS}")
    print(f"DOWNSTREAM_TRAIN_SOURCE = {DOWNSTREAM_TRAIN_SOURCE}")
    print(f"DOWNSTREAM_TRAIN_N = {EXPECTED_TRAIN_N}")
    print(f"DOWNSTREAM_SMOTE_SOURCE_ALLOWED = {DOWNSTREAM_SMOTE_SOURCE_ALLOWED}")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")


if __name__ == "__main__":
    main()
