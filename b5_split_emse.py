from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, EXPECTED_SHAPE, locate_csv
from b1_target_emse import TARGET_NAME, build_q49_code_to_days, build_target


BASE_DIR = Path(__file__).resolve().parent
DATA_DICTIONARY_PATH = BASE_DIR / "b2_data_dictionary.csv"
SPLIT_SUMMARY_PATH = BASE_DIR / "b5_split_summary.csv"
TARGET_DISTRIBUTION_PATH = BASE_DIR / "b5_target_distribution.csv"
PSU_PROFILE_PATH = BASE_DIR / "b5_psu_partition_profile.csv"
MISSING_SUMMARY_PATH = BASE_DIR / "b5_missing_partition_summary.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b5_feature_inventory.csv"
MANIFEST_PATH = BASE_DIR / "b5_split_manifest.json"
TRAIN_RECORDS_PATH = BASE_DIR / "b5_train_records.csv"
TEST_RECORDS_PATH = BASE_DIR / "b5_test_records.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
SHUFFLE = True
SPLIT_STRATEGY = "INDIVIDUAL_RANDOM_STRATIFIED_BY_TARGET"
SPLIT_UNIT = "student_individual"
SPLIT_ENGINE = "numpy.default_rng_stratified_without_replacement"
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
SAMPLE_METADATA_COLS = ["record", "psu", "stratum", "weight", "sitio"]
SAME_DOMAIN_REVIEW_COLS = ["q50", "q51"]
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
CONTRACT_EXCLUSIONS = list(
    dict.fromkeys(HARD_LEAKAGE_COLS + [TARGET_NAME, "qn40"] + SAMPLE_METADATA_COLS)
)

OPERATION_FLAGS = {
    "imputation": False,
    "scaling": False,
    "encoding": False,
    "feature_engineering": False,
    "outlier_treatment": False,
    "smote": False,
    "undersampling": False,
    "pca": False,
    "feature_selection": False,
    "transformer_fit": False,
    "model_training": False,
}


def membership_hash(records: pd.Series) -> str:
    payload = "\n".join(sorted(records.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Una única partición aleatoria estratificada, sin reemplazo."""
    rng = np.random.default_rng(random_state)
    labels = sorted(y.unique().tolist())
    target_test_n = int(np.ceil(len(y) * test_size))
    exact_allocations = y.value_counts().sort_index() * test_size
    test_allocations = np.floor(exact_allocations).astype(int)
    remainder = target_test_n - int(test_allocations.sum())
    fractions = (exact_allocations - test_allocations).sort_values(
        ascending=False, kind="stable"
    )
    for label in fractions.index[:remainder]:
        test_allocations.loc[label] += 1

    train_indices: list[object] = []
    test_indices: list[object] = []
    for label in labels:
        class_indices = y.index[y.eq(label)].to_numpy(copy=True)
        rng.shuffle(class_indices)
        class_test_n = int(test_allocations.loc[label])
        test_indices.extend(class_indices[:class_test_n].tolist())
        train_indices.extend(class_indices[class_test_n:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        X.loc[train_indices].copy(),
        X.loc[test_indices].copy(),
        y.loc[train_indices].copy(),
        y.loc[test_indices].copy(),
    )


def target_distribution(
    partitions: dict[str, pd.Series],
) -> pd.DataFrame:
    global_pct = 100 * partitions["global"].value_counts(normalize=True).sort_index()
    rows: list[dict[str, object]] = []
    for partition, values in partitions.items():
        counts = values.value_counts().sort_index()
        percentages = 100 * values.value_counts(normalize=True).sort_index()
        for target_value in (0, 1):
            rows.append(
                {
                    "partition": partition,
                    "target": target_value,
                    "n": int(counts.loc[target_value]),
                    "pct_partition": float(percentages.loc[target_value]),
                    "difference_pp_vs_global": float(
                        percentages.loc[target_value] - global_pct.loc[target_value]
                    ),
                }
            )
    return pd.DataFrame(rows)


def missing_partition_summary(
    partitions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for partition, frame in partitions.items():
        missing = frame.isna().sum()
        for column in frame.columns:
            rows.append(
                {
                    "partition": partition,
                    "column": column,
                    "n_rows": int(len(frame)),
                    "missing_n": int(missing.loc[column]),
                    "missing_pct": float(100 * missing.loc[column] / len(frame)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["partition", "missing_pct", "column"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_psu_profile(
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
) -> pd.DataFrame:
    train_counts = meta_train["psu"].value_counts(dropna=False).rename("train_n")
    test_counts = meta_test["psu"].value_counts(dropna=False).rename("test_n")
    profile = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int)
    profile.index.name = "psu"
    profile = profile.reset_index()
    profile["total_n"] = profile["train_n"] + profile["test_n"]
    profile["in_train"] = profile["train_n"].gt(0)
    profile["in_test"] = profile["test_n"].gt(0)
    profile["partition_status"] = "shared"
    profile.loc[profile["in_train"] & ~profile["in_test"], "partition_status"] = (
        "train_only"
    )
    profile.loc[~profile["in_train"] & profile["in_test"], "partition_status"] = (
        "test_only"
    )
    return profile.sort_values("psu").reset_index(drop=True)


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
        raise AssertionError("Mapping q49/texto_q49 inconsistente; B5 detenido.")
    target = build_target(df, code_to_days)
    target_mask = target.notna()
    df_target = df.loc[target_mask].copy()
    y = target.loc[target_mask].astype("int8").rename(TARGET_NAME)

    data_dictionary = pd.read_csv(DATA_DICTIONARY_PATH, encoding="utf-8-sig")
    b2_candidate_mask = data_dictionary["model_candidate"].fillna(False).astype(bool)
    initial_feature_cols = data_dictionary.loc[b2_candidate_mask, "column"].tolist()
    interpretation_text_cols = [
        column for column in df.columns if column.startswith("texto_")
    ]
    explicit_exclusion_set = set(CONTRACT_EXCLUSIONS) | set(interpretation_text_cols)
    final_feature_cols = [
        column for column in initial_feature_cols if column not in explicit_exclusion_set
    ]
    X = df_target.loc[:, final_feature_cols].copy()
    metadata_sidecar = df_target.loc[:, SAMPLE_METADATA_COLS].copy()

    feature_inventory = data_dictionary[
        ["column", "layer", "role", "statistical_type", "model_candidate"]
    ].copy()
    feature_inventory = feature_inventory.rename(
        columns={"model_candidate": "model_candidate_b2"}
    )
    feature_inventory["included_in_X_b5"] = feature_inventory["column"].isin(
        final_feature_cols
    )

    def exclusion_reason(column: str) -> str:
        if column in final_feature_cols:
            return ""
        if column in HARD_LEAKAGE_COLS:
            return "HARD_LEAKAGE"
        if column == "qn40":
            return "EXCLUIR_VACIO_INFORMATIVO"
        if column in SAMPLE_METADATA_COLS:
            return "METADATA_SIDECAR"
        if column.startswith("texto_"):
            return "INTERPRETATION_ONLY"
        return "NOT_MODEL_CANDIDATE_B2"

    feature_inventory["exclusion_reason"] = feature_inventory["column"].map(
        exclusion_reason
    )
    feature_inventory.to_csv(
        FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig"
    )

    split_call_count = 0
    split_call_count += 1
    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    meta_train = metadata_sidecar.loc[X_train_raw.index].copy()
    meta_test = metadata_sidecar.loc[X_test_raw.index].copy()

    train_records = pd.DataFrame(
        {
            "split_position": range(len(X_train_raw)),
            "original_index": X_train_raw.index,
            "record": meta_train["record"].to_numpy(),
        }
    )
    test_records = pd.DataFrame(
        {
            "split_position": range(len(X_test_raw)),
            "original_index": X_test_raw.index,
            "record": meta_test["record"].to_numpy(),
        }
    )
    train_records.to_csv(TRAIN_RECORDS_PATH, index=False, encoding="utf-8-sig")
    test_records.to_csv(TEST_RECORDS_PATH, index=False, encoding="utf-8-sig")

    split_summary = pd.DataFrame(
        [
            {
                "partition": "global",
                "n": len(X),
                "pct_universe": 100.0,
                "feature_n": X.shape[1],
                "random_state": RANDOM_STATE,
                "test_size_parameter": TEST_SIZE,
                "shuffle": SHUFFLE,
                "stratify": TARGET_NAME,
                "split_unit": SPLIT_UNIT,
                "test_frozen": TEST_FROZEN_AFTER_B5,
            },
            {
                "partition": "train",
                "n": len(X_train_raw),
                "pct_universe": 100 * len(X_train_raw) / len(X),
                "feature_n": X_train_raw.shape[1],
                "random_state": RANDOM_STATE,
                "test_size_parameter": TEST_SIZE,
                "shuffle": SHUFFLE,
                "stratify": TARGET_NAME,
                "split_unit": SPLIT_UNIT,
                "test_frozen": TEST_FROZEN_AFTER_B5,
            },
            {
                "partition": "test",
                "n": len(X_test_raw),
                "pct_universe": 100 * len(X_test_raw) / len(X),
                "feature_n": X_test_raw.shape[1],
                "random_state": RANDOM_STATE,
                "test_size_parameter": TEST_SIZE,
                "shuffle": SHUFFLE,
                "stratify": TARGET_NAME,
                "split_unit": SPLIT_UNIT,
                "test_frozen": TEST_FROZEN_AFTER_B5,
            },
        ]
    )
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    target_table = target_distribution(
        {"global": y, "train": y_train, "test": y_test}
    )
    target_table.to_csv(
        TARGET_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig"
    )

    psu_profile = build_psu_profile(meta_train, meta_test)
    psu_profile.to_csv(PSU_PROFILE_PATH, index=False, encoding="utf-8-sig")

    missing_summary = missing_partition_summary(
        {"global": X, "train": X_train_raw, "test": X_test_raw}
    )
    missing_summary.to_csv(
        MISSING_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )

    global_positive_pct = float(100 * y.mean())
    train_positive_pct = float(100 * y_train.mean())
    test_positive_pct = float(100 * y_test.mean())
    train_hash = membership_hash(meta_train["record"])
    test_hash = membership_hash(meta_test["record"])

    manifest = {
        "block": "B5",
        "source_csv": csv_path.name,
        "source_shape": list(df.shape),
        "universe_definition": "target_pa_oms5 observed",
        "universe_n": len(X),
        "split_strategy": SPLIT_STRATEGY,
        "split_unit": SPLIT_UNIT,
        "split_engine": SPLIT_ENGINE,
        "test_size": TEST_SIZE,
        "shuffle": SHUFFLE,
        "stratify": TARGET_NAME,
        "random_state": RANDOM_STATE,
        "split_call_count": split_call_count,
        "train_n": len(X_train_raw),
        "test_n": len(X_test_raw),
        "target_prevalence_pct": {
            "global_positive": global_positive_pct,
            "train_positive": train_positive_pct,
            "test_positive": test_positive_pct,
        },
        "initial_candidate_feature_n": len(initial_feature_cols),
        "final_feature_n": len(final_feature_cols),
        "continuous_feature_n": int(
            data_dictionary.loc[
                data_dictionary["column"].isin(final_feature_cols)
                & data_dictionary["statistical_type"].eq("continuous")
            ].shape[0]
        ),
        "categorical_feature_n": int(
            data_dictionary.loc[
                data_dictionary["column"].isin(final_feature_cols)
                & ~data_dictionary["statistical_type"].eq("continuous")
            ].shape[0]
        ),
        "feature_columns": final_feature_cols,
        "contract_exclusions": CONTRACT_EXCLUSIONS,
        "interpretation_text_excluded_n": len(interpretation_text_cols),
        "metadata_sidecar_columns": SAMPLE_METADATA_COLS,
        "same_domain_preserved": SAME_DOMAIN_REVIEW_COLS,
        "unknown_categorical_preserved": UNKNOWN_CATEGORICAL_COLS,
        "psu_policy": "diagnostic_metadata_not_group_split",
        "train_records_file": TRAIN_RECORDS_PATH.name,
        "test_records_file": TEST_RECORDS_PATH.name,
        "train_record_membership_sha256": train_hash,
        "test_record_membership_sha256": test_hash,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "future_fit_rule": "FIT_TRAIN_ONLY_TRANSFORM_TRAIN_AND_TEST",
        "future_sample_modifying_rule": "TRAIN_ONLY",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_index_set = set(X_train_raw.index)
    test_index_set = set(X_test_raw.index)
    universe_index_set = set(X.index)
    train_record_set = set(meta_train["record"])
    test_record_set = set(meta_test["record"])
    loaded_train_records = pd.read_csv(TRAIN_RECORDS_PATH, encoding="utf-8-sig")
    loaded_test_records = pd.read_csv(TEST_RECORDS_PATH, encoding="utf-8-sig")
    loaded_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    psu_train_n = int(psu_profile["in_train"].sum())
    psu_test_n = int(psu_profile["in_test"].sum())
    psu_shared_n = int(psu_profile["partition_status"].eq("shared").sum())
    psu_train_only_n = int(psu_profile["partition_status"].eq("train_only").sum())
    psu_test_only_n = int(psu_profile["partition_status"].eq("test_only").sum())

    expected_artifacts = [
        SPLIT_SUMMARY_PATH,
        TARGET_DISTRIBUTION_PATH,
        PSU_PROFILE_PATH,
        MISSING_SUMMARY_PATH,
        FEATURE_INVENTORY_PATH,
        MANIFEST_PATH,
        TRAIN_RECORDS_PATH,
        TEST_RECORDS_PATH,
    ]

    checks = {
        "shape_crudo_56981x309": df.shape == EXPECTED_SHAPE,
        "universo_55551": len(X) == 55_551,
        "train_mas_test_55551": len(X_train_raw) + len(X_test_raw) == 55_551,
        "train_test_indices_disjuntos": train_index_set.isdisjoint(test_index_set),
        "union_indices_igual_universo": train_index_set | test_index_set == universe_index_set,
        "sin_indices_perdidos_duplicados": (
            X_train_raw.index.is_unique
            and X_test_raw.index.is_unique
            and len(train_index_set) + len(test_index_set) == len(universe_index_set)
        ),
        "y_solo_0_1": set(y.unique()) == {0, 1},
        "prevalencia_global_29_8068": abs(global_positive_pct - 29.8068) < 0.001,
        "prevalencia_train_dentro_0_3pp": abs(train_positive_pct - 29.8068) <= 0.3,
        "prevalencia_test_dentro_0_3pp": abs(test_positive_pct - 29.8068) <= 0.3,
        "q49_fuera_X": "q49" not in X.columns,
        "qn49_fuera_X": "qn49" not in X.columns,
        "qnpa5g_fuera_X": "qnpa5g" not in X.columns,
        "qnpa7g_fuera_X": "qnpa7g" not in X.columns,
        "textos_proxy_target_fuera_X": not (
            set(HARD_LEAKAGE_COLS) & set(X.columns)
        ),
        "target_fuera_X": TARGET_NAME not in X.columns,
        "qn40_fuera_X": "qn40" not in X.columns,
        "todo_texto_interpretativo_fuera_X": not any(
            column.startswith("texto_") for column in X.columns
        ),
        "metadata_fuera_X": not (set(SAMPLE_METADATA_COLS) & set(X.columns)),
        "q50_presente": "q50" in X.columns,
        "q51_presente": "q51" in X.columns,
        "unknown_categorical_preservadas": set(UNKNOWN_CATEGORICAL_COLS).issubset(
            X.columns
        ),
        "candidatas_B2_148": len(initial_feature_cols) == 148,
        "features_finales_148": len(final_feature_cols) == 148,
        "continuas_2_categoricas_146": (
            manifest["continuous_feature_n"] == 2
            and manifest["categorical_feature_n"] == 146
        ),
        "X_y_train_alineados": X_train_raw.index.equals(y_train.index),
        "X_y_test_alineados": X_test_raw.index.equals(y_test.index),
        "sidecar_train_alineado": X_train_raw.index.equals(meta_train.index),
        "sidecar_test_alineado": X_test_raw.index.equals(meta_test.index),
        "record_train_test_disjunto": train_record_set.isdisjoint(test_record_set),
        "sin_imputacion": (
            not OPERATION_FLAGS["imputation"]
            and X_train_raw.equals(X.loc[X_train_raw.index])
            and X_test_raw.equals(X.loc[X_test_raw.index])
            and X_train_raw.isna().sum().add(X_test_raw.isna().sum()).equals(
                X.isna().sum()
            )
        ),
        "sin_scaling": not OPERATION_FLAGS["scaling"],
        "sin_encoding": not OPERATION_FLAGS["encoding"],
        "sin_feature_engineering": not OPERATION_FLAGS["feature_engineering"],
        "sin_tratamiento_outliers": not OPERATION_FLAGS["outlier_treatment"],
        "sin_smote_undersampling": (
            not OPERATION_FLAGS["smote"] and not OPERATION_FLAGS["undersampling"]
        ),
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_fit_transformadores": not OPERATION_FLAGS["transformer_fit"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "split_ejecutado_una_vez": split_call_count == 1,
        "split_no_group_aware": manifest["psu_policy"] == "diagnostic_metadata_not_group_split",
        "test_congelado": TEST_FROZEN_AFTER_B5 and loaded_manifest["TEST_FROZEN_AFTER_B5"],
        "registros_train_persistidos_exactos": (
            set(loaded_train_records["original_index"]) == train_index_set
            and set(loaded_train_records["record"]) == train_record_set
        ),
        "registros_test_persistidos_exactos": (
            set(loaded_test_records["original_index"]) == test_index_set
            and set(loaded_test_records["record"]) == test_record_set
        ),
        "hashes_frontera_correctos": (
            loaded_manifest["train_record_membership_sha256"] == train_hash
            and loaded_manifest["test_record_membership_sha256"] == test_hash
        ),
        "random_state_documentado": (
            isinstance(RANDOM_STATE, int)
            and loaded_manifest["random_state"] == RANDOM_STATE
        ),
        "artefactos_existen": all(path.exists() for path in expected_artifacts),
    }

    print("=== UNIVERSO Y X B5 ===")
    print(f"Ruta CSV: {csv_path}")
    print(f"Shape crudo: {df.shape}")
    print(f"Universo supervisado: {len(X)}")
    print(
        "Warnings de lectura: "
        + (
            "; ".join(f"{item.category.__name__}: {item.message}" for item in read_warnings)
            if read_warnings
            else "ninguno"
        )
    )
    print(f"Features candidatas B2: {len(initial_feature_cols)}")
    print(f"Features finales X B5: {len(final_feature_cols)}")
    print(
        f"Continuas: {manifest['continuous_feature_n']}; "
        f"categóricas: {manifest['categorical_feature_n']}"
    )
    print(f"Exclusiones contractuales: {CONTRACT_EXCLUSIONS}")
    print(f"Columnas texto excluidas: {len(interpretation_text_cols)}")
    print(f"q50/q51 preservadas: {[column for column in SAME_DOMAIN_REVIEW_COLS if column in X]}")
    print(
        "Unknown categorical preservadas: "
        f"{[column for column in UNKNOWN_CATEGORICAL_COLS if column in X]}"
    )

    print("\n=== SPLIT ===")
    print(f"Estrategia: {SPLIT_STRATEGY}")
    print(f"Unidad: {SPLIT_UNIT}")
    print(f"Motor: {SPLIT_ENGINE}")
    print(f"random_state: {RANDOM_STATE}")
    print(f"test_size: {TEST_SIZE}")
    print(f"shuffle: {SHUFFLE}")
    print(f"stratify: {TARGET_NAME}")
    print(f"Train: {len(X_train_raw)} ({100 * len(X_train_raw) / len(X):.6f}%)")
    print(f"Test: {len(X_test_raw)} ({100 * len(X_test_raw) / len(X):.6f}%)")

    print("\n=== TARGET ===")
    print(target_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\n=== PSU: EVIDENCIA DEL SHUFFLE INDIVIDUAL ===")
    print(f"PSU en train: {psu_train_n}")
    print(f"PSU en test: {psu_test_n}")
    print(f"PSU compartidas: {psu_shared_n}")
    print(f"PSU exclusivas train: {psu_train_only_n}")
    print(f"PSU exclusivas test: {psu_test_only_n}")
    print("La presencia compartida no es error: PSU no fue unidad indivisible del split.")

    print("\n=== MISSINGNESS q4/q5 ===")
    q4_q5_missing = missing_summary.loc[
        missing_summary["column"].isin(["q4", "q5"])
        & missing_summary["partition"].isin(["train", "test"])
    ]
    print(q4_q5_missing.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    for partition in ("train", "test"):
        print(f"\nTop missing {partition}:")
        print(
            missing_summary.loc[missing_summary["partition"].eq(partition)]
            .head(10)
            .to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )
    print("No se usaron estas tasas para cambiar semilla ni repetir el split.")

    print("\n=== FROZEN TEST ===")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")
    print("Regla futura: FIT → TRAIN; TRANSFORM → TRAIN + TEST con el mismo objeto.")
    print("Operaciones sample-modifying futuras: TRAIN ONLY.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B5 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B5 detenido; checks fallidos: {failed}")
    print("B5 finalizado.")


if __name__ == "__main__":
    main()
