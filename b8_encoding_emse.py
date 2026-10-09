from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from b1_target_emse import TARGET_NAME


BASE_DIR = Path(__file__).resolve().parent
B7_MANIFEST_PATH = BASE_DIR / "b7_scaling_manifest.json"
B7_FEATURE_INVENTORY_PATH = BASE_DIR / "b7_feature_inventory.csv"
B7_X_TRAIN_PATH = BASE_DIR / "b7_X_train.pkl"
B7_X_TEST_PATH = BASE_DIR / "b7_X_test.pkl"
B7_Y_TRAIN_PATH = BASE_DIR / "b7_y_train.pkl"
B7_Y_TEST_PATH = BASE_DIR / "b7_y_test.pkl"
B6_FEATURE_INVENTORY_PATH = BASE_DIR / "b6_feature_inventory.csv"
B2_DATA_DICTIONARY_PATH = BASE_DIR / "b2_data_dictionary.csv"

ENCODING_PLAN_PATH = BASE_DIR / "b8_encoding_plan.csv"
UNKNOWN_PROPOSALS_PATH = BASE_DIR / "b8_unknown_categorical_proposals.csv"
ORDINAL_MAPPINGS_PATH = BASE_DIR / "b8_ordinal_mappings.csv"
ONEHOT_VOCABULARIES_PATH = BASE_DIR / "b8_onehot_vocabularies.csv"
UNSEEN_TEST_PATH = BASE_DIR / "b8_unseen_test_categories.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b8_feature_inventory.csv"
ENCODING_SUMMARY_PATH = BASE_DIR / "b8_encoding_summary.csv"
MANIFEST_PATH = BASE_DIR / "b8_encoding_manifest.json"
X_TRAIN_PATH = BASE_DIR / "b8_X_train.pkl"
X_TEST_PATH = BASE_DIR / "b8_X_test.pkl"
Y_TRAIN_PATH = BASE_DIR / "b8_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b8_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_INPUT_FEATURE_N = 149
CONTINUOUS_COLS = ["q4", "q5"]
INDICATOR_COL = "q4q5_faltaba"
SIN_DATO = "sin_dato"
ORDINAL_MISSING_SENTINEL = -1
ORDINAL_UNKNOWN_SENTINEL = -2
FIT_PARTITION = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
DISCRETIZATION_B8 = "NONE"
IMC_SOURCE_FOR_B9 = "B6_IMPUTED_UNSCALED_Q4_Q5"
Q50_Q51_STATUS = "SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE"
UNKNOWN_REVIEW_STATUS = "PENDIENTE_REVISION_SEMANTICA"

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

UNKNOWN_DECISIONS = {
    "q28": (
        "Rangos de edad ordenables junto con 'Nunca probó cigarrillos' fuera de la escala",
        "La categoría de no inicio no es un nivel de edad; one-hot evita imponerle una posición",
    ),
    "q34": (
        "Rangos de edad ordenables junto con 'Nunca tomó alcohol' fuera de la escala",
        "La categoría de no inicio no es un nivel de edad; one-hot evita imponerle una posición",
    ),
    "q40": (
        "Rangos de edad ordenables junto con 'Nunca usó drogas' fuera de la escala",
        "La categoría de no inicio no es un nivel de edad; one-hot evita imponerle una posición",
    ),
    "q45": (
        "Edades ordenables junto con 'Nunca tuvo relaciones sexuales' fuera de la escala",
        "La categoría de no inicio no es un nivel de edad; one-hot evita imponerle una posición",
    ),
    "q59": (
        "Niveles educativos ordenables junto con la respuesta 'No sé'",
        "'No sé' no pertenece al orden educativo; one-hot es la representación conservadora",
    ),
    "q60": (
        "Niveles educativos ordenables junto con la respuesta 'No sé'",
        "'No sé' no pertenece al orden educativo; one-hot es la representación conservadora",
    ),
    "q69": (
        "Frecuencia ordenable junto con una categoría explícita de no viaje/no aplicación",
        "La no exposición no es una frecuencia dentro de la misma escala; se evita imponer distancia",
    ),
    "q70": (
        "Conteos ordenables junto con una categoría explícita de no viaje/no aplicación",
        "La no exposición no es un conteo dentro de la misma escala; se evita imponer distancia",
    ),
    "q71": (
        "Frecuencia ordenable junto con una categoría explícita de no uso de bicicleta",
        "La no exposición queda fuera de la escala de frecuencia; one-hot es conservador",
    ),
    "q72": (
        "Frecuencia ordenable junto con una categoría explícita de no viaje en moto",
        "La no exposición queda fuera de la escala de frecuencia; one-hot es conservador",
    ),
    "q79": (
        "Conteos ordenables junto con la respuesta 'No lo sé'",
        "'No lo sé' no pertenece al orden de conteos; one-hot evita asignarle distancia",
    ),
}

OPERATION_FLAGS = {
    "split_executed": False,
    "imputation_executed": False,
    "scaling_executed": False,
    "target_encoding": False,
    "target_used_for_encoding": False,
    "test_used_for_fit": False,
    "refit_for_unseen": False,
    "discretization": False,
    "imc_created": False,
    "b9_feature_engineering": False,
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


def category_to_text(value: object) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def category_sort_key(value: object) -> tuple[int, float | str]:
    if isinstance(value, (np.integer, int, np.floating, float)) and not isinstance(
        value, bool
    ):
        return 0, float(value)
    if value == SIN_DATO:
        return 2, SIN_DATO
    return 1, str(value)


def sorted_train_categories(series: pd.Series) -> list[object]:
    return sorted(series.unique().tolist(), key=category_sort_key)


def safe_category_token(value: object) -> str:
    text = category_to_text(value)
    token = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not token:
        token = "empty"
    return token


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b7_manifest = json.loads(B7_MANIFEST_PATH.read_text(encoding="utf-8"))
    b7_inventory = pd.read_csv(B7_FEATURE_INVENTORY_PATH, encoding="utf-8-sig")
    b6_inventory = pd.read_csv(B6_FEATURE_INVENTORY_PATH, encoding="utf-8-sig")
    b2_dictionary = pd.read_csv(B2_DATA_DICTIONARY_PATH, encoding="utf-8-sig")

    source_paths = [
        B7_X_TRAIN_PATH,
        B7_X_TEST_PATH,
        B7_Y_TRAIN_PATH,
        B7_Y_TEST_PATH,
    ]
    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    expected_source_hashes = {
        name: data["sha256"] for name, data in b7_manifest["output_files"].items()
    }
    if source_hashes != expected_source_hashes:
        raise AssertionError("Hashes de entrada B7 no coinciden; B8 detenido.")

    X_train_b7 = pd.read_pickle(B7_X_TRAIN_PATH)
    X_test_b7 = pd.read_pickle(B7_X_TEST_PATH)
    y_train_b7 = pd.read_pickle(B7_Y_TRAIN_PATH)
    y_test_b7 = pd.read_pickle(B7_Y_TEST_PATH)

    prefit_checks = {
        "train_44440": len(X_train_b7) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b7) == EXPECTED_TEST_N,
        "input_149": X_train_b7.shape[1]
        == X_test_b7.shape[1]
        == EXPECTED_INPUT_FEATURE_N,
        "indices_X_y_train": X_train_b7.index.equals(y_train_b7.index),
        "indices_X_y_test": X_test_b7.index.equals(y_test_b7.index),
        "missing_train_0": int(X_train_b7.isna().sum().sum()) == 0,
        "missing_test_0": int(X_test_b7.isna().sum().sum()) == 0,
        "test_frozen": b7_manifest["TEST_FROZEN_AFTER_B5"] is True,
        "b7_inventory_149": len(b7_inventory) == EXPECTED_INPUT_FEATURE_N,
        "b6_inventory_149": len(b6_inventory) == EXPECTED_INPUT_FEATURE_N,
    }
    if not all(prefit_checks.values()):
        failed = [name for name, passed in prefit_checks.items() if not passed]
        raise AssertionError(f"Precondiciones B8 inválidas; B8 detenido: {failed}")

    X_train_before_b8 = X_train_b7.copy(deep=True)
    X_test_before_b8 = X_test_b7.copy(deep=True)
    y_train = y_train_b7.copy(deep=True)
    y_test = y_test_b7.copy(deep=True)

    statistical_types = b7_inventory.set_index("column")["statistical_type"].to_dict()
    if set(statistical_types) != set(X_train_before_b8.columns):
        raise AssertionError("Inventario B7 y columnas de X no coinciden; B8 detenido.")

    unknown_rows: list[dict[str, object]] = []
    b2_by_column = b2_dictionary.set_index("column")
    for column in UNKNOWN_CATEGORICAL_COLS:
        semantic_scale, reason = UNKNOWN_DECISIONS[column]
        categories = sorted_train_categories(X_train_before_b8[column])
        unknown_rows.append(
            {
                "column": column,
                "observed_train_categories": json.dumps(
                    [category_to_text(value) for value in categories],
                    ensure_ascii=False,
                ),
                "paired_text_column": b2_by_column.loc[
                    column, "paired_text_column"
                ],
                "semantic_scale_detected": semantic_scale,
                "proposed_type": "NOMINAL",
                "confidence": "MEDIA",
                "reason": reason,
                "encoding_to_execute": "ONE_HOT",
                "review_status": UNKNOWN_REVIEW_STATUS,
            }
        )
    unknown_proposals = pd.DataFrame(unknown_rows)
    unknown_type_override = dict.fromkeys(UNKNOWN_CATEGORICAL_COLS, "NOMINAL")

    ordinal_cols: list[str] = []
    onehot_cols: list[str] = []
    plan_rows: list[dict[str, object]] = []
    for column in X_train_before_b8.columns:
        statistical_type = statistical_types[column]
        if column in CONTINUOUS_COLS:
            action = "unchanged_scaled_continuous"
            output_kind = "continuous_numeric"
            fit_required = False
            fit_partition = "NO_FIT"
            notes = "Escalada en B7; B8 no la modifica"
        elif column == INDICATOR_COL:
            action = "unchanged_binary_indicator"
            output_kind = "binary_numeric"
            fit_required = False
            fit_partition = "NO_FIT"
            notes = "Indicador técnico 0/1; no se escala ni codifica"
        else:
            proposed = unknown_type_override.get(column)
            if statistical_type == "ordinal_categorical" or proposed == "ORDINAL":
                ordinal_cols.append(column)
                action = "ordinal_encode"
                output_kind = "ordinal_numeric"
                fit_required = True
                fit_partition = "train"
                notes = (
                    "Orden semántico de códigos; sin_dato=-1; unseen_test=-2"
                )
            else:
                onehot_cols.append(column)
                action = "one_hot"
                output_kind = "onehot_numeric"
                fit_required = True
                fit_partition = "train"
                notes = "Vocabulario train-only; unseen test=all-zero block"
            if column in {"q50", "q51"}:
                notes += f"; {Q50_Q51_STATUS}; pendiente de revisión semántica"
            if column in UNKNOWN_CATEGORICAL_COLS:
                notes += f"; propuesta NOMINAL; {UNKNOWN_REVIEW_STATUS}"
        plan_rows.append(
            {
                "column": column,
                "input_statistical_type": statistical_type,
                "action": action,
                "output_kind": output_kind,
                "fit_required": fit_required,
                "fit_partition": fit_partition,
                "notes": notes,
            }
        )
    encoding_plan = pd.DataFrame(plan_rows)

    ordinal_train = pd.DataFrame(index=X_train_before_b8.index)
    ordinal_test = pd.DataFrame(index=X_test_before_b8.index)
    ordinal_mapping_rows: list[dict[str, object]] = []
    onehot_train_blocks: list[pd.DataFrame] = []
    onehot_test_blocks: list[pd.DataFrame] = []
    onehot_vocabulary_rows: list[dict[str, object]] = []
    unseen_rows: list[dict[str, object]] = []
    onehot_block_checks: list[bool] = []
    ordinal_unseen_checks: list[bool] = []

    for column in ordinal_cols:
        train_categories = sorted_train_categories(X_train_before_b8[column])
        substantive_categories = [
            value for value in train_categories if value != SIN_DATO
        ]
        mapping = {
            value: position for position, value in enumerate(substantive_categories)
        }
        if SIN_DATO in train_categories:
            mapping[SIN_DATO] = ORDINAL_MISSING_SENTINEL

        train_encoded = X_train_before_b8[column].map(mapping)
        test_encoded = X_test_before_b8[column].map(mapping)
        train_unseen_mask = train_encoded.isna()
        test_unseen_mask = test_encoded.isna()
        if train_unseen_mask.any():
            raise AssertionError(f"Mapping ordinal incompleto en train para {column}")
        test_encoded.loc[test_unseen_mask] = ORDINAL_UNKNOWN_SENTINEL
        ordinal_train[column] = train_encoded.astype("int16")
        ordinal_test[column] = test_encoded.astype("int16")
        ordinal_unseen_checks.append(
            ordinal_test.loc[test_unseen_mask, column]
            .eq(ORDINAL_UNKNOWN_SENTINEL)
            .all()
        )

        for position, value in enumerate(substantive_categories):
            ordinal_mapping_rows.append(
                {
                    "column": column,
                    "category": category_to_text(value),
                    "category_kind": "substantive_train_category",
                    "semantic_order_position": position,
                    "encoded_value": mapping[value],
                    "fit_partition": "train",
                }
            )
        if SIN_DATO in mapping:
            ordinal_mapping_rows.append(
                {
                    "column": column,
                    "category": SIN_DATO,
                    "category_kind": "missing_category_sentinel",
                    "semantic_order_position": "",
                    "encoded_value": ORDINAL_MISSING_SENTINEL,
                    "fit_partition": "FIXED_RULE",
                }
            )
        ordinal_mapping_rows.append(
            {
                "column": column,
                "category": "__UNSEEN_TEST__",
                "category_kind": "unknown_category_sentinel",
                "semantic_order_position": "",
                "encoded_value": ORDINAL_UNKNOWN_SENTINEL,
                "fit_partition": "FIXED_RULE",
            }
        )

        unseen_values = sorted(
            X_test_before_b8.loc[test_unseen_mask, column].unique().tolist(),
            key=category_sort_key,
        )
        for value in unseen_values:
            unseen_rows.append(
                {
                    "column": column,
                    "category": category_to_text(value),
                    "test_n": int(X_test_before_b8[column].eq(value).sum()),
                    "encoding_policy": f"ordinal_sentinel_{ORDINAL_UNKNOWN_SENTINEL}_no_refit",
                }
            )

    for column in onehot_cols:
        train_categories = sorted_train_categories(X_train_before_b8[column])
        tokens = [safe_category_token(value) for value in train_categories]
        if len(tokens) != len(set(tokens)):
            raise AssertionError(f"Colisión de tokens one-hot en {column}: {tokens}")
        output_columns = [f"{column}__{token}" for token in tokens]
        train_block = pd.DataFrame(index=X_train_before_b8.index)
        test_block = pd.DataFrame(index=X_test_before_b8.index)
        for position, (value, output_column) in enumerate(
            zip(train_categories, output_columns, strict=True)
        ):
            train_block[output_column] = X_train_before_b8[column].eq(value).astype(
                "uint8"
            )
            test_block[output_column] = X_test_before_b8[column].eq(value).astype(
                "uint8"
            )
            onehot_vocabulary_rows.append(
                {
                    "column": column,
                    "category": category_to_text(value),
                    "category_type": type(value).__name__,
                    "vocabulary_position": position,
                    "output_column": output_column,
                    "fit_partition": "train",
                    "is_sin_dato": value == SIN_DATO,
                }
            )

        seen_mask = X_test_before_b8[column].isin(train_categories)
        unseen_mask = ~seen_mask
        onehot_block_checks.append(train_block.sum(axis=1).eq(1).all())
        onehot_block_checks.append(test_block.loc[seen_mask].sum(axis=1).eq(1).all())
        onehot_block_checks.append(test_block.loc[unseen_mask].sum(axis=1).eq(0).all())
        unseen_values = sorted(
            X_test_before_b8.loc[unseen_mask, column].unique().tolist(),
            key=category_sort_key,
        )
        for value in unseen_values:
            unseen_rows.append(
                {
                    "column": column,
                    "category": category_to_text(value),
                    "test_n": int(X_test_before_b8[column].eq(value).sum()),
                    "encoding_policy": "onehot_all_zero_block_handle_unknown_ignore_no_refit",
                }
            )
        onehot_train_blocks.append(train_block)
        onehot_test_blocks.append(test_block)

    onehot_train = pd.concat(onehot_train_blocks, axis=1)
    onehot_test = pd.concat(onehot_test_blocks, axis=1)
    ordinal_mappings = pd.DataFrame(ordinal_mapping_rows)
    onehot_vocabularies = pd.DataFrame(onehot_vocabulary_rows)
    unseen_test = pd.DataFrame(
        unseen_rows,
        columns=["column", "category", "test_n", "encoding_policy"],
    )

    X_train_b8 = pd.concat(
        [
            X_train_before_b8[CONTINUOUS_COLS].copy(),
            X_train_before_b8[[INDICATOR_COL]].astype("int8").copy(),
            ordinal_train,
            onehot_train,
        ],
        axis=1,
    )
    X_test_b8 = pd.concat(
        [
            X_test_before_b8[CONTINUOUS_COLS].copy(),
            X_test_before_b8[[INDICATOR_COL]].astype("int8").copy(),
            ordinal_test,
            onehot_test,
        ],
        axis=1,
    )
    if not X_train_b8.columns.is_unique or not X_test_b8.columns.is_unique:
        raise AssertionError("Columnas B8 duplicadas; B8 detenido.")

    feature_inventory_rows: list[dict[str, object]] = []
    output_position = 0
    for column in CONTINUOUS_COLS:
        feature_inventory_rows.append(
            {
                "output_column": column,
                "source_column": column,
                "output_position": output_position,
                "output_kind": "continuous_numeric",
                "encoding": "unchanged_scaled_B7",
                "source_category": "",
                "review_status": "CONFIRMED",
            }
        )
        output_position += 1
    feature_inventory_rows.append(
        {
            "output_column": INDICATOR_COL,
            "source_column": INDICATOR_COL,
            "output_position": output_position,
            "output_kind": "binary_numeric",
            "encoding": "unchanged_indicator",
            "source_category": "",
            "review_status": "CONFIRMED",
        }
    )
    output_position += 1
    for column in ordinal_cols:
        feature_inventory_rows.append(
            {
                "output_column": column,
                "source_column": column,
                "output_position": output_position,
                "output_kind": "ordinal_numeric",
                "encoding": "ordinal_train_mapping",
                "source_category": "",
                "review_status": (
                    "PENDIENTE_REVISION_SEMANTICA"
                    if column in {"q50", "q51"}
                    else "B2_ORDINAL"
                ),
            }
        )
        output_position += 1
    vocab_by_output = onehot_vocabularies.set_index("output_column")
    for output_column in onehot_train.columns:
        source_column = vocab_by_output.loc[output_column, "column"]
        feature_inventory_rows.append(
            {
                "output_column": output_column,
                "source_column": source_column,
                "output_position": output_position,
                "output_kind": "onehot_numeric",
                "encoding": "onehot_train_vocabulary",
                "source_category": vocab_by_output.loc[output_column, "category"],
                "review_status": (
                    UNKNOWN_REVIEW_STATUS
                    if source_column in UNKNOWN_CATEGORICAL_COLS
                    else "B2_NOMINAL_OR_BINARY"
                ),
            }
        )
        output_position += 1
    b8_feature_inventory = pd.DataFrame(feature_inventory_rows)

    train_input_sin_dato = sum(
        int(X_train_before_b8[column].eq(SIN_DATO).sum())
        for column in ordinal_cols + onehot_cols
    )
    test_input_sin_dato = sum(
        int(X_test_before_b8[column].eq(SIN_DATO).sum())
        for column in ordinal_cols + onehot_cols
    )
    train_encoded_sin_dato = int(
        ordinal_train.eq(ORDINAL_MISSING_SENTINEL).sum().sum()
    ) + int(
        onehot_train[
            onehot_vocabularies.loc[
                onehot_vocabularies["is_sin_dato"], "output_column"
            ].tolist()
        ].sum().sum()
    )
    test_encoded_sin_dato = int(
        ordinal_test.eq(ORDINAL_MISSING_SENTINEL).sum().sum()
    ) + int(
        onehot_test[
            onehot_vocabularies.loc[
                onehot_vocabularies["is_sin_dato"], "output_column"
            ].tolist()
        ].sum().sum()
    )

    encoding_summary = pd.DataFrame(
        [
            {
                "input_feature_n": X_train_before_b8.shape[1],
                "output_feature_n": X_train_b8.shape[1],
                "continuous_unchanged_n": len(CONTINUOUS_COLS),
                "indicator_unchanged_n": 1,
                "ordinal_input_features_n": len(ordinal_cols),
                "onehot_input_features_n": len(onehot_cols),
                "output_ordinal_columns_n": ordinal_train.shape[1],
                "output_onehot_columns_n": onehot_train.shape[1],
                "onehot_train_vocabulary_categories_n": len(onehot_vocabularies),
                "unseen_test_categories_n": len(unseen_test),
                "unseen_test_observations_n": int(unseen_test["test_n"].sum()),
                "q50_included": "q50" in ordinal_cols,
                "q51_included": "q51" in ordinal_cols,
                "q50_q51_status": Q50_Q51_STATUS,
                "unknown_11_proposals_n": len(unknown_proposals),
                "missing_train_before": int(X_train_before_b8.isna().sum().sum()),
                "missing_test_before": int(X_test_before_b8.isna().sum().sum()),
                "missing_train_after": int(X_train_b8.isna().sum().sum()),
                "missing_test_after": int(X_test_b8.isna().sum().sum()),
            }
        ]
    )

    encoding_plan.to_csv(ENCODING_PLAN_PATH, index=False, encoding="utf-8-sig")
    unknown_proposals.to_csv(
        UNKNOWN_PROPOSALS_PATH, index=False, encoding="utf-8-sig"
    )
    ordinal_mappings.to_csv(
        ORDINAL_MAPPINGS_PATH, index=False, encoding="utf-8-sig"
    )
    onehot_vocabularies.to_csv(
        ONEHOT_VOCABULARIES_PATH, index=False, encoding="utf-8-sig"
    )
    unseen_test.to_csv(UNSEEN_TEST_PATH, index=False, encoding="utf-8-sig")
    b8_feature_inventory.to_csv(
        FEATURE_INVENTORY_PATH, index=False, encoding="utf-8-sig"
    )
    encoding_summary.to_csv(
        ENCODING_SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )

    X_train_b8.to_pickle(X_TRAIN_PATH)
    X_test_b8.to_pickle(X_TEST_PATH)
    y_train.to_pickle(Y_TRAIN_PATH)
    y_test.to_pickle(Y_TEST_PATH)

    output_paths = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    manifest = {
        "block": "B8",
        "source_block": "B7",
        "source_manifest": B7_MANIFEST_PATH.name,
        "source_output_hashes": source_hashes,
        "train_n": len(X_train_b8),
        "test_n": len(X_test_b8),
        "input_feature_n": X_train_before_b8.shape[1],
        "output_feature_n": X_train_b8.shape[1],
        "continuous_columns_unchanged": CONTINUOUS_COLS,
        "indicator_unchanged": INDICATOR_COL,
        "ordinal_input_columns": ordinal_cols,
        "onehot_input_columns": onehot_cols,
        "ordinal_missing_sentinel": ORDINAL_MISSING_SENTINEL,
        "ordinal_unknown_test_sentinel": ORDINAL_UNKNOWN_SENTINEL,
        "onehot_unknown_test_policy": "all_zero_block_handle_unknown_ignore",
        "FIT_PARTITION": FIT_PARTITION,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "TRANSFORM_PARTITIONS": ["train", "test"],
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "target_used_for_encoding": False,
        "DISCRETIZATION_B8": DISCRETIZATION_B8,
        "IMC_SOURCE_FOR_B9": IMC_SOURCE_FOR_B9,
        "q50_included": True,
        "q51_included": True,
        "q50_q51_status": Q50_Q51_STATUS,
        "unknown_categorical_review_status": UNKNOWN_REVIEW_STATUS,
        "output_files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    expected_artifacts = [
        ENCODING_PLAN_PATH,
        UNKNOWN_PROPOSALS_PATH,
        ORDINAL_MAPPINGS_PATH,
        ONEHOT_VOCABULARIES_PATH,
        UNSEEN_TEST_PATH,
        FEATURE_INVENTORY_PATH,
        ENCODING_SUMMARY_PATH,
        MANIFEST_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]
    output_train_array = X_train_b8.to_numpy(dtype="float64")
    output_test_array = X_test_b8.to_numpy(dtype="float64")
    unseen_policy_checks = onehot_block_checks + ordinal_unseen_checks

    checks = {
        "train_44440": len(X_train_b8) == EXPECTED_TRAIN_N,
        "test_11111": len(X_test_b8) == EXPECTED_TEST_N,
        "indices_identicos_B7": X_train_b8.index.equals(X_train_b7.index)
        and X_test_b8.index.equals(X_test_b7.index),
        "y_identico_B7": y_train.equals(y_train_b7) and y_test.equals(y_test_b7),
        "test_congelado": TEST_FROZEN_AFTER_B5
        and b7_manifest["TEST_FROZEN_AFTER_B5"],
        "no_nuevo_split": not OPERATION_FLAGS["split_executed"],
        "input_features_149": X_train_before_b8.shape[1]
        == X_test_before_b8.shape[1]
        == EXPECTED_INPUT_FEATURE_N,
        "missing_input_train_0": int(X_train_before_b8.isna().sum().sum()) == 0,
        "missing_input_test_0": int(X_test_before_b8.isna().sum().sum()) == 0,
        "q4_no_recodificada": X_train_b8["q4"].equals(X_train_before_b8["q4"])
        and X_test_b8["q4"].equals(X_test_before_b8["q4"]),
        "q5_no_recodificada": X_train_b8["q5"].equals(X_train_before_b8["q5"])
        and X_test_b8["q5"].equals(X_test_before_b8["q5"]),
        "indicador_intacto": X_train_b8[INDICATOR_COL].equals(
            X_train_before_b8[INDICATOR_COL]
        )
        and X_test_b8[INDICATOR_COL].equals(X_test_before_b8[INDICATOR_COL]),
        "q50_incluida": "q50" in ordinal_train.columns,
        "q51_incluida": "q51" in ordinal_train.columns,
        "unknown_11_explicitadas": set(unknown_proposals["column"])
        == set(UNKNOWN_CATEGORICAL_COLS),
        "unknown_11_propuesta_y_razon": len(unknown_proposals) == 11
        and unknown_proposals["proposed_type"].isin(["ORDINAL", "NOMINAL"]).all()
        and unknown_proposals["reason"].str.len().gt(0).all(),
        "unknown_11_pendiente_auditoria": unknown_proposals["review_status"]
        .eq(UNKNOWN_REVIEW_STATUS)
        .all(),
        "sin_dato_preservado_encoding": train_encoded_sin_dato
        == train_input_sin_dato
        and test_encoded_sin_dato == test_input_sin_dato,
        "sin_target_encoding": not OPERATION_FLAGS["target_encoding"]
        and not OPERATION_FLAGS["target_used_for_encoding"],
        "sin_discretizacion": DISCRETIZATION_B8 == "NONE"
        and not OPERATION_FLAGS["discretization"],
        "vocabularios_onehot_train_only": onehot_vocabularies["fit_partition"]
        .eq("train")
        .all(),
        "mappings_ordinales_sin_test": not ordinal_mappings["fit_partition"]
        .astype(str)
        .str.contains("test", case=False)
        .any(),
        "test_no_usado_fit": TEST_USED_FOR_FIT is False
        and not OPERATION_FLAGS["test_used_for_fit"],
        "unseen_sin_refit": not OPERATION_FLAGS["refit_for_unseen"]
        and all(unseen_policy_checks),
        "mismas_columnas_train_test": X_train_b8.columns.equals(X_test_b8.columns),
        "mismo_orden_train_test": X_train_b8.columns.tolist()
        == X_test_b8.columns.tolist(),
        "salida_totalmente_numerica": all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_b8.dtypes
        )
        and all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_test_b8.dtypes),
        "sin_nan_train": int(X_train_b8.isna().sum().sum()) == 0,
        "sin_nan_test": int(X_test_b8.isna().sum().sum()) == 0,
        "sin_inf_train": bool(np.isfinite(output_train_array).all()),
        "sin_inf_test": bool(np.isfinite(output_test_array).all()),
        "indices_intactos": X_train_b8.index.equals(X_train_before_b8.index)
        and X_test_b8.index.equals(X_test_before_b8.index),
        "y_intacto": y_train.equals(y_train_b7) and y_test.equals(y_test_b7),
        "leakage_fuera": not any(
            source in HARD_LEAKAGE_COLS
            or any(
                output == blocked or output.startswith(f"{blocked}__")
                for blocked in HARD_LEAKAGE_COLS
            )
            for source, output in zip(
                b8_feature_inventory["source_column"],
                b8_feature_inventory["output_column"],
                strict=True,
            )
        ),
        "target_fuera": TARGET_NAME not in X_train_b8.columns
        and not any(
            column.startswith(f"{TARGET_NAME}__") for column in X_train_b8.columns
        ),
        "metadata_fuera": not any(
            source in METADATA_COLS for source in b8_feature_inventory["source_column"]
        ),
        "texto_fuera": not any(
            source.startswith("texto_")
            for source in b8_feature_inventory["source_column"]
        ),
        "sin_imc": not OPERATION_FLAGS["imc_created"]
        and not any("imc" in column.lower() for column in X_train_b8.columns),
        "sin_feature_engineering_B9": not OPERATION_FLAGS["b9_feature_engineering"],
        "sin_smote": not OPERATION_FLAGS["smote"],
        "sin_pca": not OPERATION_FLAGS["pca"],
        "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
        "sin_modelo": not OPERATION_FLAGS["model_training"],
        "mappings_vocabularios_persistidos": ORDINAL_MAPPINGS_PATH.exists()
        and ONEHOT_VOCABULARIES_PATH.exists()
        and UNSEEN_TEST_PATH.exists(),
        "artefactos_generados": all(path.exists() for path in expected_artifacts),
        "pickles_recargables_alineados": pd.read_pickle(X_TRAIN_PATH).index.equals(
            pd.read_pickle(Y_TRAIN_PATH).index
        )
        and pd.read_pickle(X_TEST_PATH).index.equals(pd.read_pickle(Y_TEST_PATH).index)
        and pd.read_pickle(X_TRAIN_PATH).columns.equals(
            pd.read_pickle(X_TEST_PATH).columns
        ),
        "hashes_fuente_B7_verificados": source_hashes == expected_source_hashes,
    }

    print("=== FRONTERA B7 FINAL ===")
    print(f"Train: {len(X_train_b7)}")
    print(f"Test: {len(X_test_b7)}")
    print(f"Features entrada: {X_train_b7.shape[1]}")
    print("Hashes de los cuatro pickles B7: verificados")
    print("Índices X/y: alineados")
    print("Split reejecutado: False")
    print(f"TEST_FROZEN_AFTER_B5 = {TEST_FROZEN_AFTER_B5}")

    print("\n=== PLAN DE ENCODING ===")
    print(f"Continuas sin cambios: {CONTINUOUS_COLS}")
    print(f"Indicador sin cambios: {INDICATOR_COL}")
    print(f"Ordinales: {len(ordinal_cols)}")
    print(f"Nominales/binarias one-hot: {len(onehot_cols)}")
    print(f"q50 action: {encoding_plan.set_index('column').loc['q50', 'action']}")
    print(f"q51 action: {encoding_plan.set_index('column').loc['q51', 'action']}")

    print("\n=== UNKNOWN CATEGORICAL: PROPUESTAS DE TIPADO ===")
    print(
        unknown_proposals[
            [
                "column",
                "proposed_type",
                "confidence",
                "reason",
                "encoding_to_execute",
                "review_status",
            ]
        ].to_string(index=False)
    )

    print("\n=== q50 / q51 ===")
    print("q50_included = True; encoding = ORDINAL")
    print("q51_included = True; encoding = ORDINAL")
    print(f"q50_q51_status = {Q50_Q51_STATUS}")
    for column in ("q50", "q51"):
        mapping_view = ordinal_mappings.loc[
            ordinal_mappings["column"].eq(column),
            ["category", "encoded_value", "category_kind"],
        ]
        print(f"{column} mapping:")
        print(mapping_view.to_string(index=False))

    print("\n=== ONE-HOT ===")
    print(f"Variables input: {len(onehot_cols)}")
    print(f"Columnas output: {onehot_train.shape[1]}")
    print(f"Categorías de vocabulario train: {len(onehot_vocabularies)}")
    print(f"Categorías unseen en test: {len(unseen_test)}")
    print(f"Observaciones test con categoría unseen: {int(unseen_test['test_n'].sum())}")
    if unseen_test.empty:
        print("Unseen test: ninguna; CSV vacío guardado con encabezados.")
    else:
        print(unseen_test.to_string(index=False))

    print("\n=== ORDINAL ===")
    print(f"Variables input/output: {len(ordinal_cols)}/{ordinal_train.shape[1]}")
    print(f"Mappings sustantivos: {int(ordinal_mappings['category_kind'].eq('substantive_train_category').sum())}")
    print(f"sin_dato sentinel: {ORDINAL_MISSING_SENTINEL}")
    print(f"unseen_test sentinel: {ORDINAL_UNKNOWN_SENTINEL}")
    ordinal_unseen_n = int(
        unseen_test.loc[
            unseen_test["encoding_policy"].str.startswith("ordinal", na=False),
            "test_n",
        ].sum()
    )
    print(f"Observaciones ordinales unseen test: {ordinal_unseen_n}")

    print("\n=== DISCRETIZACIÓN ===")
    print(f"DISCRETIZATION_B8 = {DISCRETIZATION_B8}")

    print("\n=== SALIDA B8 ===")
    print(f"X_train_b8 shape: {X_train_b8.shape}")
    print(f"X_test_b8 shape: {X_test_b8.shape}")
    print(f"Feature count: {X_train_b8.shape[1]}")
    print(f"Numeric only: {checks['salida_totalmente_numerica']}")
    print(
        f"Missing train/test: {int(X_train_b8.isna().sum().sum())}/"
        f"{int(X_test_b8.isna().sum().sum())}"
    )
    print(
        f"Inf train/test: {int(np.isinf(output_train_array).sum())}/"
        f"{int(np.isinf(output_test_array).sum())}"
    )

    print("\n=== FIT / TRANSFORM ===")
    print("FIT one-hot vocabularies: TRAIN ONLY")
    print("FIT ordinal mappings: TRAIN ONLY / semantic fixed order where applicable")
    print("TRANSFORM encoders: TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")
    print("UNSEEN_TEST_CATEGORIES: HANDLE_WITHOUT_REFIT")
    print(f"DISCRETIZATION_B8 = {DISCRETIZATION_B8}")

    print("\n=== RESTRICCIÓN B9 ===")
    print(f"IMC_SOURCE_FOR_B9 = {IMC_SOURCE_FOR_B9}")
    print("B8 no creó IMC ni usó q4/q5 escalados para calcularlo.")

    print("\n=== ARTEFACTOS ===")
    for path in expected_artifacts:
        print(path)

    print("\n=== CHECKS B8 ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"B8 detenido; checks fallidos: {failed}")
    print("B8 finalizado.")


if __name__ == "__main__":
    main()
