from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE

from b1_target_emse import TARGET_NAME


BASE_DIR = Path(__file__).resolve().parent
B9_MANIFEST_PATH = BASE_DIR / "b9_feature_engineering_manifest.json"
B9_FEATURE_INVENTORY_PATH = BASE_DIR / "b9_feature_inventory.csv"
B9_X_TRAIN_PATH = BASE_DIR / "b9_X_train.pkl"
B9_X_TEST_PATH = BASE_DIR / "b9_X_test.pkl"
B9_Y_TRAIN_PATH = BASE_DIR / "b9_y_train.pkl"
B9_Y_TEST_PATH = BASE_DIR / "b9_y_test.pkl"

CLASS_DISTRIBUTION_PATH = BASE_DIR / "b10_class_distribution.csv"
SAMPLE_ORIGIN_PATH = BASE_DIR / "b10_sample_origin.csv"
SMOTE_SUMMARY_PATH = BASE_DIR / "b10_smote_summary.csv"
INTERPOLATION_DIAGNOSTIC_PATH = (
    BASE_DIR / "b10_smote_categorical_interpolation_diagnostic.csv"
)
FEATURE_INVENTORY_PATH = BASE_DIR / "b10_feature_inventory.csv"
MANIFEST_PATH = BASE_DIR / "b10_smote_manifest.json"
X_TRAIN_SMOTE_PATH = BASE_DIR / "b10_X_train_smote.pkl"
Y_TRAIN_SMOTE_PATH = BASE_DIR / "b10_y_train_smote.pkl"
X_TEST_PATH = BASE_DIR / "b10_X_test.pkl"
Y_TEST_PATH = BASE_DIR / "b10_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_FEATURE_N = 446
SAMPLING_STRATEGY = "auto"
RANDOM_STATE = 42
K_NEIGHBORS = 5
METHOD = "SMOTE"
IMPLEMENTATION = "imblearn.over_sampling.SMOTE"
FIT_RESAMPLE_PARTITION = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_RESAMPLED = False
TEST_FROZEN_AFTER_B5 = True
LIMITATION = (
    "Vanilla SMOTE interpolates encoded categorical dimensions; "
    "synthetic categorical coordinates may be fractional."
)
UNKNOWN_AUDIT_STATUS = "PENDIENTE_REVISION_SEMANTICA"
Q50_Q51_STATUS = "SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE"

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
    "new_split": False,
    "test_passed_to_smote": False,
    "new_imputation": False,
    "new_scaling": False,
    "reencoding": False,
    "discretization": False,
    "outlier_treatment": False,
    "new_feature_engineering": False,
    "synthetic_rounding": False,
    "smotenc_used": False,
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


def verify_b9_hashes(
    paths: list[Path], manifest: dict[str, object]
) -> dict[str, str]:
    actual = {path.name: file_sha256(path) for path in paths}
    expected = {
        name: data["sha256"]
        for name, data in manifest["output_files"].items()
    }
    if actual != expected:
        raise AssertionError(
            f"Hashes B9 inválidos: actual={actual}, expected={expected}"
        )
    return actual


def class_distribution(
    partition: str, stage: str, target: pd.Series
) -> list[dict[str, object]]:
    counts = target.value_counts().sort_index()
    return [
        {
            "partition": partition,
            "stage": stage,
            "target": int(label),
            "n": int(count),
            "pct": float(count / len(target) * 100),
        }
        for label, count in counts.items()
    ]


def target_counts(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def diagnostic_row(
    feature_group: str,
    feature_n: int,
    values: np.ndarray,
    include_unit_interval: bool,
) -> dict[str, object]:
    flat = values.ravel()
    exactly_zero = int(np.count_nonzero(flat == 0))
    exactly_one = int(np.count_nonzero(flat == 1))
    fractional = int(
        np.count_nonzero(
            ~np.isclose(flat, np.round(flat), rtol=0, atol=1e-12)
        )
    )
    return {
        "feature_group": feature_group,
        "feature_n": feature_n,
        "synthetic_rows": values.shape[0],
        "total_cells": int(flat.size),
        "exactly_0_n": exactly_zero,
        "exactly_1_n": exactly_one,
        "strictly_between_0_1_n": (
            int(np.count_nonzero((flat > 0) & (flat < 1)))
            if include_unit_interval
            else pd.NA
        ),
        "outside_0_1_n": (
            int(np.count_nonzero((flat < 0) | (flat > 1)))
            if include_unit_interval
            else pd.NA
        ),
        "fractional_n": fractional,
        "audit_status": "PENDIENTE_REVISION_SEMANTICA",
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b9_manifest = json.loads(B9_MANIFEST_PATH.read_text(encoding="utf-8"))
    b9_inventory = pd.read_csv(
        B9_FEATURE_INVENTORY_PATH, encoding="utf-8-sig"
    )
    b9_paths = [
        B9_X_TRAIN_PATH,
        B9_X_TEST_PATH,
        B9_Y_TRAIN_PATH,
        B9_Y_TEST_PATH,
    ]
    b9_hashes = verify_b9_hashes(b9_paths, b9_manifest)

    X_train_b9 = pd.read_pickle(B9_X_TRAIN_PATH)
    X_test_b9 = pd.read_pickle(B9_X_TEST_PATH)
    y_train_b9 = pd.read_pickle(B9_Y_TRAIN_PATH)
    y_test_b9 = pd.read_pickle(B9_Y_TEST_PATH)

    train_array = X_train_b9.to_numpy(dtype="float64")
    test_array = X_test_b9.to_numpy(dtype="float64")
    pre_checks = {
        "train_n": len(X_train_b9) == EXPECTED_TRAIN_N,
        "test_n": len(X_test_b9) == EXPECTED_TEST_N,
        "train_features": X_train_b9.shape[1] == EXPECTED_FEATURE_N,
        "test_features": X_test_b9.shape[1] == EXPECTED_FEATURE_N,
        "train_xy_aligned": X_train_b9.index.equals(y_train_b9.index),
        "test_xy_aligned": X_test_b9.index.equals(y_test_b9.index),
        "indices_disjoint": X_train_b9.index.intersection(X_test_b9.index).empty,
        "same_columns": X_train_b9.columns.equals(X_test_b9.columns),
        "numeric_only": all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_b9.dtypes
        ),
        "no_nan": int(X_train_b9.isna().sum().sum()) == 0
        and int(X_test_b9.isna().sum().sum()) == 0,
        "no_inf": np.isfinite(train_array).all()
        and np.isfinite(test_array).all(),
        "test_frozen": b9_manifest["TEST_FROZEN_AFTER_B5"] is True,
        "b9_test_not_fit": b9_manifest["TEST_USED_FOR_FIT"] is False,
        "inventory_446": len(b9_inventory) == EXPECTED_FEATURE_N,
        "target_binary": set(y_train_b9.unique()) == {0, 1}
        and set(y_test_b9.unique()) == {0, 1},
    }
    if not all(pre_checks.values()):
        failed = [name for name, passed in pre_checks.items() if not passed]
        raise AssertionError(f"Entrada B9 inválida; B10 detenido: {failed}")

    train_before_counts = target_counts(y_train_b9)
    test_counts = target_counts(y_test_b9)
    train_prevalence_before = float(y_train_b9.mean() * 100)
    test_prevalence_before = float(y_test_b9.mean() * 100)

    smote = SMOTE(
        sampling_strategy=SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
        k_neighbors=K_NEIGHBORS,
    )
    # imblearn reconstruye los dtypes pandas de entrada; una copia float64
    # evita truncar silenciosamente las interpolaciones de columnas enteras.
    X_train_smote_input = X_train_b9.astype("float64", copy=True)
    if not np.array_equal(X_train_smote_input.to_numpy(), train_array):
        raise AssertionError(
            "La normalización técnica de dtype alteró valores; B10 detenido."
        )
    X_resampled_raw, y_resampled_raw = smote.fit_resample(
        X_train_smote_input, y_train_b9
    )

    if not isinstance(X_resampled_raw, pd.DataFrame):
        X_resampled_raw = pd.DataFrame(
            X_resampled_raw, columns=X_train_b9.columns
        )
    if not isinstance(y_resampled_raw, pd.Series):
        y_resampled_raw = pd.Series(y_resampled_raw, name=y_train_b9.name)
    X_resampled_raw = X_resampled_raw.reset_index(drop=True)
    y_resampled_raw = y_resampled_raw.reset_index(drop=True)

    original_train_n = len(X_train_b9)
    resampled_train_n = len(X_resampled_raw)
    synthetic_n = resampled_train_n - original_train_n
    original_values_unchanged = np.array_equal(
        X_resampled_raw.iloc[:original_train_n].to_numpy(dtype="float64"),
        train_array,
    ) and np.array_equal(
        y_resampled_raw.iloc[:original_train_n].to_numpy(),
        y_train_b9.to_numpy(),
    )
    if not original_values_unchanged:
        raise AssertionError(
            "SMOTE no preservó las observaciones originales al inicio; "
            "B10 detenido."
        )

    original_labels = [
        f"original_train_{position:06d}"
        for position in range(original_train_n)
    ]
    synthetic_labels = [
        f"synthetic_smote_{position:06d}"
        for position in range(1, synthetic_n + 1)
    ]
    technical_index = pd.Index(
        original_labels + synthetic_labels, name="b10_sample_id"
    )
    X_train_smote = X_resampled_raw.copy()
    y_train_smote = y_resampled_raw.copy()
    X_train_smote.index = technical_index
    y_train_smote.index = technical_index
    y_train_smote.name = y_train_b9.name

    sample_origin = pd.DataFrame(
        {
            "resampled_position": np.arange(resampled_train_n),
            "sample_origin": (
                ["ORIGINAL_TRAIN"] * original_train_n
                + ["SYNTHETIC_SMOTE"] * synthetic_n
            ),
            "original_index_if_applicable": (
                [str(value) for value in X_train_b9.index]
                + [pd.NA] * synthetic_n
            ),
        }
    )

    synthetic = X_resampled_raw.iloc[original_train_n:]
    onehot_cols = b9_inventory.loc[
        b9_inventory["output_kind"].eq("onehot_numeric"), "output_column"
    ].tolist()
    ordinal_cols = b9_inventory.loc[
        b9_inventory["output_kind"].eq("ordinal_numeric"), "output_column"
    ].tolist()
    indicator_cols = b9_inventory.loc[
        b9_inventory["output_kind"].eq("binary_numeric"), "output_column"
    ].tolist()
    continuous_cols = b9_inventory.loc[
        b9_inventory["output_kind"].eq("continuous_numeric"), "output_column"
    ].tolist()
    if continuous_cols != ["q4", "q5", "imc"] or indicator_cols != [
        "q4q5_faltaba"
    ]:
        raise AssertionError(
            "Roles B9 inesperados para continuas/indicador; B10 detenido."
        )

    interpolation_diagnostic = pd.DataFrame(
        [
            diagnostic_row(
                "onehot_numeric",
                len(onehot_cols),
                synthetic[onehot_cols].to_numpy(dtype="float64"),
                include_unit_interval=True,
            ),
            diagnostic_row(
                "ordinal_numeric",
                len(ordinal_cols),
                synthetic[ordinal_cols].to_numpy(dtype="float64"),
                include_unit_interval=False,
            ),
            diagnostic_row(
                "binary_indicator",
                len(indicator_cols),
                synthetic[indicator_cols].to_numpy(dtype="float64"),
                include_unit_interval=True,
            ),
        ]
    )

    train_after_counts = target_counts(y_train_smote)
    train_prevalence_after = float(y_train_smote.mean() * 100)
    class_distribution_table = pd.DataFrame(
        class_distribution("TRAIN", "BEFORE_SMOTE", y_train_b9)
        + class_distribution("TRAIN", "AFTER_SMOTE", y_train_smote)
        + class_distribution("TEST", "UNTOUCHED", y_test_b9)
    )

    smote_summary = pd.DataFrame(
        [
            {
                "method": METHOD,
                "implementation": IMPLEMENTATION,
                "imbalanced_learn_version": imblearn.__version__,
                "scikit_learn_version": sklearn.__version__,
                "sampling_strategy": SAMPLING_STRATEGY,
                "random_state": RANDOM_STATE,
                "k_neighbors": K_NEIGHBORS,
                "fit_resample_partition": FIT_RESAMPLE_PARTITION,
                "smote_input_dtype": "float64_numeric_copy",
                "test_used_for_fit": TEST_USED_FOR_FIT,
                "test_resampled": TEST_RESAMPLED,
                "original_train_n": original_train_n,
                "synthetic_train_n": synthetic_n,
                "resampled_train_n": resampled_train_n,
                "test_n": len(X_test_b9),
                "input_feature_n": X_train_b9.shape[1],
                "output_feature_n": X_train_smote.shape[1],
                "train_positive_pct_before": train_prevalence_before,
                "train_positive_pct_after": train_prevalence_after,
                "test_positive_pct_untouched": test_prevalence_before,
                "encoded_space_limitation": LIMITATION,
            }
        ]
    )

    b10_inventory = b9_inventory.copy()
    b10_inventory["b10_action"] = (
        "unchanged_feature_space_smote_rows_only"
    )

    class_distribution_table.to_csv(
        CLASS_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig"
    )
    sample_origin.to_csv(
        SAMPLE_ORIGIN_PATH, index=False, encoding="utf-8-sig"
    )
    smote_summary.to_csv(
        SMOTE_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )
    interpolation_diagnostic.to_csv(
        INTERPOLATION_DIAGNOSTIC_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    b10_inventory.to_csv(
        FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig"
    )
    X_train_smote.to_pickle(X_TRAIN_SMOTE_PATH)
    y_train_smote.to_pickle(Y_TRAIN_SMOTE_PATH)
    shutil.copy2(B9_X_TEST_PATH, X_TEST_PATH)
    shutil.copy2(B9_Y_TEST_PATH, Y_TEST_PATH)

    X_test_b10 = pd.read_pickle(X_TEST_PATH)
    y_test_b10 = pd.read_pickle(Y_TEST_PATH)
    x_test_hash_equal = file_sha256(X_TEST_PATH) == file_sha256(B9_X_TEST_PATH)
    y_test_hash_equal = file_sha256(Y_TEST_PATH) == file_sha256(B9_Y_TEST_PATH)
    x_test_data_equal = X_test_b10.equals(X_test_b9)
    y_test_data_equal = y_test_b10.equals(y_test_b9)

    output_files = [
        CLASS_DISTRIBUTION_PATH,
        SAMPLE_ORIGIN_PATH,
        SMOTE_SUMMARY_PATH,
        INTERPOLATION_DIAGNOSTIC_PATH,
        FEATURE_INVENTORY_PATH,
        X_TRAIN_SMOTE_PATH,
        Y_TRAIN_SMOTE_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]
    manifest = {
        "block": "B10",
        "source_block": "B9",
        "source_hashes": b9_hashes,
        "train_original_n": original_train_n,
        "train_resampled_n": resampled_train_n,
        "synthetic_n": synthetic_n,
        "test_n": len(X_test_b10),
        "input_feature_n": X_train_b9.shape[1],
        "output_feature_n": X_train_smote.shape[1],
        "method": METHOD,
        "implementation": IMPLEMENTATION,
        "imbalanced_learn_version": imblearn.__version__,
        "scikit_learn_version": sklearn.__version__,
        "sampling_strategy": SAMPLING_STRATEGY,
        "random_state": RANDOM_STATE,
        "k_neighbors": K_NEIGHBORS,
        "FIT_RESAMPLE_PARTITION": FIT_RESAMPLE_PARTITION,
        "smote_input_dtype": "float64_numeric_copy",
        "smote_input_dtype_reason": (
            "prevent pandas integer dtype restoration from truncating "
            "synthetic interpolations"
        ),
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "TEST_RESAMPLED": TEST_RESAMPLED,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "target_before": train_before_counts,
        "target_after_train": train_after_counts,
        "target_test_untouched": test_counts,
        "SMOTE_ENCODED_SPACE_LIMITATION": LIMITATION,
        "unknown_11_status": UNKNOWN_AUDIT_STATUS,
        "q50_q51_status": Q50_Q51_STATUS,
        "output_files": {
            path.name: {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    expected_artifacts = output_files + [MANIFEST_PATH]
    onehot_diag = interpolation_diagnostic.loc[
        interpolation_diagnostic["feature_group"].eq("onehot_numeric")
    ].iloc[0]
    ordinal_diag = interpolation_diagnostic.loc[
        interpolation_diagnostic["feature_group"].eq("ordinal_numeric")
    ].iloc[0]
    indicator_diag = interpolation_diagnostic.loc[
        interpolation_diagnostic["feature_group"].eq("binary_indicator")
    ].iloc[0]
    unknown_output_cols = b9_inventory.loc[
        b9_inventory["source_column"].isin(UNKNOWN_CATEGORICAL_COLS),
        "output_column",
    ].tolist()
    manifest_required_keys = {
        "block",
        "source_block",
        "train_original_n",
        "train_resampled_n",
        "synthetic_n",
        "test_n",
        "input_feature_n",
        "output_feature_n",
        "method",
        "implementation",
        "sampling_strategy",
        "random_state",
        "k_neighbors",
        "FIT_RESAMPLE_PARTITION",
        "TEST_USED_FOR_FIT",
        "TEST_RESAMPLED",
        "TEST_FROZEN_AFTER_B5",
        "target_before",
        "target_after_train",
        "target_test_untouched",
        "SMOTE_ENCODED_SPACE_LIMITATION",
        "unknown_11_status",
        "q50_q51_status",
        "output_files",
    }

    checks = {
        "train_original_44440": original_train_n == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b10) == EXPECTED_TEST_N,
        "hashes_B9_verificados": bool(b9_hashes),
        "X_y_train_alineados": pre_checks["train_xy_aligned"],
        "X_y_test_alineados": pre_checks["test_xy_aligned"],
        "test_frozen": TEST_FROZEN_AFTER_B5 and pre_checks["test_frozen"],
        "sin_nuevo_split": not OPERATION_FLAGS["new_split"],
        "train_features_446": X_train_b9.shape[1] == EXPECTED_FEATURE_N,
        "test_features_446": X_test_b9.shape[1] == EXPECTED_FEATURE_N,
        "train_test_mismas_columnas": pre_checks["same_columns"],
        "input_sin_nan": pre_checks["no_nan"],
        "input_sin_inf": pre_checks["no_inf"],
        "metodo_SMOTE": METHOD == "SMOTE"
        and IMPLEMENTATION == "imblearn.over_sampling.SMOTE",
        "sampling_strategy_auto": SAMPLING_STRATEGY == "auto",
        "random_state_42": RANDOM_STATE == 42,
        "k_neighbors_5": K_NEIGHBORS == 5,
        "fit_resample_solo_train": FIT_RESAMPLE_PARTITION == "TRAIN_ONLY",
        "test_nunca_fit_resample": TEST_USED_FOR_FIT is False
        and not OPERATION_FLAGS["test_passed_to_smote"],
        "synthetic_n_mayor_0": synthetic_n > 0,
        "train_output_mayor_input": resampled_train_n > original_train_n,
        "train_final_balanceado": len(set(train_after_counts.values())) == 1,
        "target0_igual_target1_final": train_after_counts["0"]
        == train_after_counts["1"],
        "features_siguen_446": X_train_smote.shape[1] == EXPECTED_FEATURE_N,
        "test_n_sigue_11111": len(X_test_b10) == EXPECTED_TEST_N,
        "X_test_B10_igual_B9": x_test_data_equal,
        "y_test_B10_igual_B9": y_test_data_equal,
        "hash_X_test_identico": x_test_hash_equal,
        "hash_y_test_identico": y_test_hash_equal,
        "prevalencia_test_identica": float(y_test_b10.mean())
        == float(y_test_b9.mean()),
        "indices_test_intactos": X_test_b10.index.equals(X_test_b9.index)
        and y_test_b10.index.equals(y_test_b9.index),
        "originales_no_modificados": original_values_unchanged,
        "synthetic_distinguibles": sample_origin["sample_origin"].value_counts().to_dict()
        == {"ORIGINAL_TRAIN": original_train_n, "SYNTHETIC_SMOTE": synthetic_n}
        and all(label.startswith("synthetic_smote_") for label in synthetic_labels),
        "sin_record_reales_fabricados": sample_origin.loc[
            sample_origin["sample_origin"].eq("SYNTHETIC_SMOTE"),
            "original_index_if_applicable",
        ].isna().all(),
        "sin_nuevas_features": X_train_smote.columns.equals(X_train_b9.columns),
        "q50_q51_presentes": set(
            b9_inventory.loc[
                b9_inventory["source_column"].isin(["q50", "q51"]),
                "source_column",
            ]
        )
        == {"q50", "q51"},
        "outputs_unknown_presentes": bool(unknown_output_cols)
        and set(unknown_output_cols).issubset(X_train_smote.columns),
        "leakage_fuera": not any(
            source in HARD_LEAKAGE_COLS
            or output in HARD_LEAKAGE_COLS
            or any(
                output.startswith(f"{blocked}__")
                for blocked in HARD_LEAKAGE_COLS
            )
            for source, output in zip(
                b10_inventory["source_column"],
                b10_inventory["output_column"],
                strict=True,
            )
        ),
        "metadata_fuera": not any(
            source in METADATA_COLS
            for source in b10_inventory["source_column"]
        ),
        "target_fuera_X": TARGET_NAME not in X_train_smote.columns,
        "diagnostico_onehot_generado": int(onehot_diag["feature_n"]) == 401,
        "diagnostico_ordinal_generado": int(ordinal_diag["feature_n"]) == 41,
        "diagnostico_indicador_generado": int(indicator_diag["feature_n"]) == 1,
        "onehot_fuera_0_1_cero": int(onehot_diag["outside_0_1_n"]) == 0,
        "limitacion_encoded_space_registrada": manifest[
            "SMOTE_ENCODED_SPACE_LIMITATION"
        ]
        == LIMITATION,
        "sinteticos_no_redondeados": not OPERATION_FLAGS["synthetic_rounding"]
        and all(dtype == np.dtype("float64") for dtype in X_train_smote.dtypes),
        "sin_SMOTENC_silencioso": not OPERATION_FLAGS["smotenc_used"]
        and smote.__class__.__name__ == "SMOTE",
        "sin_nueva_imputacion": not OPERATION_FLAGS["new_imputation"],
        "sin_nuevo_scaling": not OPERATION_FLAGS["new_scaling"],
        "sin_reencoding": not OPERATION_FLAGS["reencoding"],
        "sin_discretizacion": not OPERATION_FLAGS["discretization"],
        "sin_outlier_treatment": not OPERATION_FLAGS["outlier_treatment"],
        "sin_nuevo_feature_engineering": not OPERATION_FLAGS[
            "new_feature_engineering"
        ],
        "sin_PCA": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "pickles_recargables": pd.read_pickle(X_TRAIN_SMOTE_PATH).equals(
            X_train_smote
        )
        and pd.read_pickle(Y_TRAIN_SMOTE_PATH).equals(y_train_smote)
        and pd.read_pickle(X_TEST_PATH).equals(X_test_b9)
        and pd.read_pickle(Y_TEST_PATH).equals(y_test_b9),
        "manifest_completo": manifest_required_keys.issubset(manifest),
    }

    print("=== FRONTERA B9 ===")
    print(f"Fuente: {B9_MANIFEST_PATH}")
    print("Hashes B9: verificados")
    print(f"Train original: {X_train_b9.shape}")
    print(f"Test: {X_test_b9.shape}")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")

    print("\n=== CLASES ANTES ===")
    print(
        class_distribution_table.loc[
            class_distribution_table["stage"].isin(
                ["BEFORE_SMOTE", "UNTOUCHED"]
            )
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    print("\n=== SMOTE ===")
    print(f"Implementación: {IMPLEMENTATION} {imblearn.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"sampling_strategy: {SAMPLING_STRATEGY}")
    print(f"random_state: {RANDOM_STATE}")
    print(f"k_neighbors: {K_NEIGHBORS}")
    print(f"original_train_n: {original_train_n}")
    print(f"synthetic_train_n: {synthetic_n}")
    print(f"resampled_train_n: {resampled_train_n}")

    print("\n=== CLASES DESPUÉS ===")
    print(
        class_distribution_table.loc[
            class_distribution_table["stage"].eq("AFTER_SMOTE")
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(
        "Test permanece: "
        f"0={test_counts['0']}, 1={test_counts['1']}, "
        f"positivos={test_prevalence_before:.6f}%"
    )

    print("\n=== SHAPES ===")
    print(f"Antes train/test: {X_train_b9.shape} / {X_test_b9.shape}")
    print(f"Después train SMOTE/test intacto: {X_train_smote.shape} / {X_test_b10.shape}")

    print("\n=== DIAGNÓSTICO ENCODED-SPACE ===")
    print(interpolation_diagnostic.to_string(index=False))
    print("SMOTE_ENCODED_SPACE_LIMITATION:")
    print(LIMITATION)

    print("\n=== FIT / RESAMPLE ===")
    print("SMOTE FIT_RESAMPLE: TRAIN ONLY")
    print("TEST RESAMPLED: False")
    print("TEST_USED_FOR_FIT = False")
    print(f"TRAIN PREVALENCE BEFORE: {train_prevalence_before:.6f}%")
    print(f"TRAIN PREVALENCE AFTER: {train_prevalence_after:.6f}%")
    print(f"TEST PREVALENCE UNTOUCHED: {test_prevalence_before:.6f}%")
    print("TEST_FROZEN_AFTER_B5 = True")

    print("\n=== PRESERVACIÓN ===")
    print(f"X_test igualdad byte SHA256: {x_test_hash_equal}")
    print(f"y_test igualdad byte SHA256: {y_test_hash_equal}")
    print(f"X_test igualdad de datos: {x_test_data_equal}")
    print(f"y_test igualdad de datos: {y_test_data_equal}")
    print(f"Observaciones train originales intactas: {original_values_unchanged}")
    print(f"Features: {X_train_smote.shape[1]} (sin cambios)")
    print("Sin imputación, scaling, reencoding, redondeo, PCA, selección o modelo.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B10 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B10 detenido; checks fallidos: {failed}")
    print(
        f"Resultado: {sum(checks.values())}/{len(checks)} checks OK. "
        "B10 finalizado; rama demostrativa sin uso aguas abajo."
    )


if __name__ == "__main__":
    main()
