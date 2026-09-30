from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from b0_setup_emse import CSV_ENCODING, CSV_SEPARATOR, locate_csv
from b1_target_emse import TARGET_NAME


BASE_DIR = Path(__file__).resolve().parent
B7_MANIFEST_PATH = BASE_DIR / "b7_scaling_manifest.json"
B7_FEATURE_INVENTORY_PATH = BASE_DIR / "b7_feature_inventory.csv"
B7_X_TRAIN_PATH = BASE_DIR / "b7_X_train.pkl"
B7_X_TEST_PATH = BASE_DIR / "b7_X_test.pkl"
B7_Y_TRAIN_PATH = BASE_DIR / "b7_y_train.pkl"
B7_Y_TEST_PATH = BASE_DIR / "b7_y_test.pkl"
B2_DATA_DICTIONARY_PATH = BASE_DIR / "b2_data_dictionary.csv"
PREVIOUS_B8_PLAN_PATH = BASE_DIR / "b8_encoding_plan.csv"

REVIEW_PATH = BASE_DIR / "b8_patch_ordinal_review.csv"
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
ORDINAL_ESCAPE_SENTINEL = -3
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
PATCH_REASON = "ORDINAL_SEMANTIC_REVIEW"
PATCH_SOURCE = "REVISION_SEMANTICA_ORDINALES"
AUDIT_STATUS = "SEMANTIC_REVIEW_APPLIED"
IMC_SOURCE_FOR_B9 = "B6_IMPUTED_UNSCALED_Q4_Q5"
Q50_Q51_STATUS = "SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE"

PREVIOUS_ORDINAL_COLS = [
    "q1", "q3", "q6", "q10", "q15", "q16", "q17", "q22", "q23",
    "q26", "q27", "q29", "q30", "q32", "q35", "q36", "q38",
    "q39", "q41", "q42", "q43", "q46", "q50", "q51", "q52",
    "q53", "q54", "q55", "q56", "q57", "q58", "q61", "q62",
    "q63", "q64", "q65", "q76", "q77", "q78", "q80", "q81",
]
UNKNOWN_CATEGORICAL_COLS = [
    "q28", "q34", "q40", "q45", "q59", "q60", "q69", "q70",
    "q71", "q72", "q79",
]
UNKNOWN_TO_ORDINAL = ["q59", "q60", "q69", "q70", "q71", "q72", "q79"]
UNKNOWN_KEEP_NOMINAL = ["q28", "q34", "q40", "q45"]
HARD_LEAKAGE_COLS = [
    "q49", "qn49", "qnpa5g", "qnpa7g", "texto_q49", "texto_qn49",
    "texto_qnpa5g", "texto_qnpa7g",
]
METADATA_COLS = ["record", "psu", "stratum", "weight", "sitio"]

PREVIOUS_ORDINAL_REASONS: dict[str, str] = {
    "q1": "Las etiquetas son edades crecientes desde 11 años o menos hasta 18 años o más.",
    "q3": "Las etiquetas son grados o años educativos en progresión escolar.",
    "q6": "Las etiquetas forman la frecuencia Nunca→Rara vez→Algunas veces→Casi siempre→Siempre.",
    "q10": "Las etiquetas representan 0 a 7 días en orden creciente.",
    "q15": "Las etiquetas son conteos agrupados desde ninguna hasta 12 o más veces.",
    "q16": "Las etiquetas son conteos agrupados desde ninguna hasta 12 o más veces.",
    "q17": "Las etiquetas son conteos agrupados desde ninguna hasta 12 o más veces.",
    "q22": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q23": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q26": "Las etiquetas son conteos crecientes desde 0 hasta 6 o más veces.",
    "q27": "Las etiquetas son conteos crecientes desde 0 hasta 3 o más.",
    "q29": "Las etiquetas son rangos crecientes de días hasta los 30 días.",
    "q30": "Las etiquetas son rangos crecientes de días hasta los 30 días.",
    "q32": "Las etiquetas son rangos crecientes de días desde 0 hasta los 7 días.",
    "q35": "Las etiquetas son rangos crecientes de días hasta los 30 días.",
    "q36": "Las etiquetas progresan desde ningún consumo hasta 5 tragos o más.",
    "q38": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 10 o más veces.",
    "q39": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 10 o más veces.",
    "q41": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 20 o más veces.",
    "q42": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 20 o más veces.",
    "q43": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 20 o más veces.",
    "q46": "Nunca equivale a cero personas y las restantes etiquetas aumentan hasta 6 o más personas.",
    "q50": "Las etiquetas representan 0 a 7 días en orden creciente.",
    "q51": "Las etiquetas representan días crecientes desde 0 hasta 5 o más.",
    "q52": "Las etiquetas son rangos crecientes de horas por día.",
    "q53": "Las etiquetas son rangos crecientes de días desde 0 hasta 10 o más.",
    "q54": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q55": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q56": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q57": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q58": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q61": "Las etiquetas aumentan desde no consumo hasta 4 o más veces al día.",
    "q62": "Las etiquetas aumentan desde no consumo hasta 4 o más veces al día.",
    "q63": "Las etiquetas aumentan desde no consumo hasta 4 o más veces al día.",
    "q64": "Las etiquetas aumentan desde no consumo hasta 4 o más veces al día.",
    "q65": "Las etiquetas aumentan desde no consumo hasta 4 o más veces al día.",
    "q76": "Las etiquetas forman una escala ordenada de certeza desde definitivamente no hasta definitivamente sí.",
    "q77": "Las etiquetas forman una frecuencia creciente desde Nunca hasta Todos los días.",
    "q78": "Las etiquetas son conteos agrupados crecientes desde 0 hasta 20 o más veces.",
    "q80": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
    "q81": "Las etiquetas forman una escala inequívoca de frecuencia desde Nunca hasta Siempre.",
}

UNKNOWN_SEMANTICS = {
    "q28": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "NOMINAL",
        "encoding": "ONE_HOT",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6", "7", "8"],
        "escape_codes": ["1"],
        "reason": "Mezcla rangos de edad con 'Nunca probó cigarrillos'; Nunca no es una edad y se conserva one-hot para no imponer una distancia artificial.",
    },
    "q34": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "NOMINAL",
        "encoding": "ONE_HOT",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6", "7", "8"],
        "escape_codes": ["1"],
        "reason": "Mezcla rangos de edad con 'Nunca tomó alcohol'; Nunca no es una edad y se conserva one-hot.",
    },
    "q40": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "NOMINAL",
        "encoding": "ONE_HOT",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6", "7", "8"],
        "escape_codes": ["1"],
        "reason": "Mezcla rangos de edad con 'Nunca usó drogas'; Nunca no es una edad y se conserva one-hot.",
    },
    "q45": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "NOMINAL",
        "encoding": "ONE_HOT",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6", "7", "8"],
        "escape_codes": ["1"],
        "reason": "Mezcla edades con 'Nunca tuvo relaciones sexuales'; Nunca no es una edad y se conserva one-hot.",
    },
    "q59": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["1", "2", "3", "4", "5", "6"],
        "escape_codes": ["7"],
        "reason": "Los seis niveles educativos progresan de primaria incompleta a universitario completo; 'No sé' usa sentinel de escape y no integra el orden.",
    },
    "q60": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["1", "2", "3", "4", "5", "6"],
        "escape_codes": ["7"],
        "reason": "Los seis niveles educativos progresan de primaria incompleta a universitario completo; 'No sé' usa sentinel de escape y no integra el orden.",
    },
    "q69": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "MEDIA",
        "ordered_codes": ["2", "3", "4", "5"],
        "escape_codes": ["1", "6"],
        "reason": "Rara vez→Siempre es frecuencia ordenada; los códigos 1 y 6 tienen la misma etiqueta de no viaje y comparten sentinel de escape.",
    },
    "q70": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6"],
        "escape_codes": ["1"],
        "reason": "0→1→2-3→4-5→6+ veces es un conteo ordenado; no haber viajado es escape separado.",
    },
    "q71": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6"],
        "escape_codes": ["1"],
        "reason": "Nunca→Siempre es frecuencia ordenada; no haber manejado bicicleta es escape separado.",
    },
    "q72": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["2", "3", "4", "5", "6"],
        "escape_codes": ["1"],
        "reason": "Nunca→Siempre es frecuencia ordenada; no haber viajado en moto es escape separado.",
    },
    "q79": {
        "structure": "ORDERED_WITH_ESCAPE_CATEGORY",
        "type": "ORDINAL",
        "encoding": "ORDINAL_WITH_SENTINEL",
        "confidence": "ALTA",
        "ordered_codes": ["1", "2", "3"],
        "escape_codes": ["4"],
        "reason": "0→1→2+ veces es un conteo ordenado; 'No lo sé' usa sentinel de escape.",
    },
}

OPERATION_FLAGS = {
    "new_split": False,
    "target_used_for_typing": False,
    "test_used_for_fit": False,
    "imc": False,
    "smote": False,
    "feature_selection": False,
    "pca": False,
    "model": False,
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
    if isinstance(value, (np.integer, int, np.floating, float)) and not isinstance(value, bool):
        return 0, float(value)
    if value == SIN_DATO:
        return 2, SIN_DATO
    return 1, str(value)


def sorted_train_categories(series: pd.Series) -> list[object]:
    return sorted(series.unique().tolist(), key=category_sort_key)


def safe_category_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", category_to_text(value)).strip("_")
    return token or "empty"


def response_label_maps(raw: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for column in columns:
        text_column = f"texto_{column}"
        pairs = raw[[column, text_column]].dropna().drop_duplicates()
        if pairs[column].duplicated().any():
            raise AssertionError(f"Mapping código→texto inconsistente para {column}")
        result[column] = {
            category_to_text(code): str(label)
            for code, label in pairs.sort_values(column).itertuples(index=False)
        }
    return result


def labels_json(label_map: dict[str, str], train_categories: list[object]) -> str:
    labels = {
        category_to_text(value): (
            "sin_dato (categoría fija B6)"
            if value == SIN_DATO
            else label_map[category_to_text(value)]
        )
        for value in train_categories
    }
    return json.dumps(labels, ensure_ascii=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    b7_manifest = json.loads(B7_MANIFEST_PATH.read_text(encoding="utf-8"))
    b7_inventory = pd.read_csv(B7_FEATURE_INVENTORY_PATH, encoding="utf-8-sig")
    b2_dictionary = pd.read_csv(B2_DATA_DICTIONARY_PATH, encoding="utf-8-sig")
    previous_plan = pd.read_csv(PREVIOUS_B8_PLAN_PATH, encoding="utf-8-sig")
    source_paths = [B7_X_TRAIN_PATH, B7_X_TEST_PATH, B7_Y_TRAIN_PATH, B7_Y_TEST_PATH]
    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    expected_hashes = {
        name: data["sha256"] for name, data in b7_manifest["output_files"].items()
    }
    if source_hashes != expected_hashes:
        raise AssertionError("Hashes B7 inválidos; patch detenido.")

    X_train_b7 = pd.read_pickle(B7_X_TRAIN_PATH)
    X_test_b7 = pd.read_pickle(B7_X_TEST_PATH)
    y_train_b7 = pd.read_pickle(B7_Y_TRAIN_PATH)
    y_test_b7 = pd.read_pickle(B7_Y_TEST_PATH)
    previous_ordinal_cols = previous_plan.loc[
        previous_plan["action"].eq("ordinal_encode"), "column"
    ].tolist()
    if previous_ordinal_cols != PREVIOUS_ORDINAL_COLS:
        raise AssertionError("Las ordinales previas no coinciden con el contrato de 41 variables.")

    review_columns = PREVIOUS_ORDINAL_COLS + UNKNOWN_CATEGORICAL_COLS
    usecols = [item for column in review_columns for item in (column, f"texto_{column}")]
    raw = pd.read_csv(
        locate_csv(), usecols=usecols, sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING, low_memory=False,
    )
    label_maps = response_label_maps(raw, review_columns)
    b2_by_column = b2_dictionary.set_index("column")

    review_rows: list[dict[str, object]] = []
    for column in PREVIOUS_ORDINAL_COLS:
        categories = sorted_train_categories(X_train_b7[column])
        review_rows.append({
            "review_scope": "PREVIOUS_ORDINAL",
            "column": column,
            "paired_text_column": b2_by_column.loc[column, "paired_text_column"],
            "train_categories": json.dumps([category_to_text(v) for v in categories], ensure_ascii=False),
            "response_labels": labels_json(label_maps[column], categories),
            "current_type": "ORDINAL",
            "semantic_structure": "ORDERED_SCALE",
            "review_decision": "KEEP_ORDINAL",
            "reason": PREVIOUS_ORDINAL_REASONS[column],
            "new_encoding": "ORDINAL",
            "audit_status": AUDIT_STATUS,
        })
    for column in UNKNOWN_CATEGORICAL_COLS:
        config = UNKNOWN_SEMANTICS[column]
        categories = sorted_train_categories(X_train_b7[column])
        review_rows.append({
            "review_scope": "PREVIOUS_UNKNOWN",
            "column": column,
            "paired_text_column": b2_by_column.loc[column, "paired_text_column"],
            "train_categories": json.dumps([category_to_text(v) for v in categories], ensure_ascii=False),
            "response_labels": labels_json(label_maps[column], categories),
            "current_type": "UNKNOWN_AS_NOMINAL_ONEHOT",
            "semantic_structure": config["structure"],
            "review_decision": (
                "KEEP_ORDINAL" if config["type"] == "ORDINAL"
                else "CHANGE_TO_NOMINAL_ONEHOT"
            ),
            "reason": config["reason"],
            "new_encoding": config["encoding"],
            "audit_status": AUDIT_STATUS,
        })
    ordinal_review = pd.DataFrame(review_rows)
    if ordinal_review["review_decision"].eq("REQUIRES_MANUAL_REVIEW").any():
        raise AssertionError("Hay variables que requieren revisión manual; B9 no puede ejecutarse.")

    unknown_rows: list[dict[str, object]] = []
    for column in UNKNOWN_CATEGORICAL_COLS:
        config = UNKNOWN_SEMANTICS[column]
        categories = sorted_train_categories(X_train_b7[column])
        escape_labels = [label_maps[column][code] for code in config["escape_codes"]]
        unknown_rows.append({
            "column": column,
            "observed_train_categories": json.dumps([category_to_text(v) for v in categories], ensure_ascii=False),
            "paired_text_column": b2_by_column.loc[column, "paired_text_column"],
            "semantic_scale_detected": config["structure"],
            "proposed_type": config["type"],
            "confidence": config["confidence"],
            "reason": config["reason"],
            "encoding_to_execute": config["encoding"],
            "escape_categories": json.dumps(escape_labels, ensure_ascii=False),
            "escape_sentinel": (
                ORDINAL_ESCAPE_SENTINEL if config["type"] == "ORDINAL" else ""
            ),
            "audit_status": AUDIT_STATUS,
        })
    unknown_proposals = pd.DataFrame(unknown_rows)

    final_ordinal_cols = PREVIOUS_ORDINAL_COLS + UNKNOWN_TO_ORDINAL
    categorical_cols = [
        column for column in X_train_b7.columns
        if column not in CONTINUOUS_COLS + [INDICATOR_COL]
    ]
    onehot_cols = [column for column in categorical_cols if column not in final_ordinal_cols]
    statistical_types = b7_inventory.set_index("column")["statistical_type"].to_dict()

    plan_rows: list[dict[str, object]] = []
    for column in X_train_b7.columns:
        if column in CONTINUOUS_COLS:
            action, output_kind, fit_required, fit_partition = (
                "unchanged_scaled_continuous", "continuous_numeric", False, "NO_FIT"
            )
            notes = "Escalada en B7; patch B8 no modifica valores"
        elif column == INDICATOR_COL:
            action, output_kind, fit_required, fit_partition = (
                "unchanged_binary_indicator", "binary_numeric", False, "NO_FIT"
            )
            notes = "Indicador técnico 0/1 sin encoding"
        elif column in final_ordinal_cols:
            action, output_kind, fit_required, fit_partition = (
                "ordinal_encode", "ordinal_numeric", True, "train"
            )
            notes = (
                f"Orden semántico auditado; sin_dato={ORDINAL_MISSING_SENTINEL}; "
                f"escape={ORDINAL_ESCAPE_SENTINEL}; unseen={ORDINAL_UNKNOWN_SENTINEL}"
            )
        else:
            action, output_kind, fit_required, fit_partition = (
                "one_hot", "onehot_numeric", True, "train"
            )
            notes = "Vocabulario train-only; unseen test produce bloque all-zero"
        plan_rows.append({
            "column": column,
            "input_statistical_type": statistical_types[column],
            "action": action,
            "output_kind": output_kind,
            "fit_required": fit_required,
            "fit_partition": fit_partition,
            "notes": notes,
        })
    encoding_plan = pd.DataFrame(plan_rows)

    ordinal_train = pd.DataFrame(index=X_train_b7.index)
    ordinal_test = pd.DataFrame(index=X_test_b7.index)
    ordinal_mapping_rows: list[dict[str, object]] = []
    unseen_rows: list[dict[str, object]] = []
    ordinal_policy_checks: list[bool] = []

    for column in final_ordinal_cols:
        train_categories = sorted_train_categories(X_train_b7[column])
        if column in UNKNOWN_TO_ORDINAL:
            ordered_codes = UNKNOWN_SEMANTICS[column]["ordered_codes"]
            escape_codes = set(UNKNOWN_SEMANTICS[column]["escape_codes"])
        else:
            ordered_codes = list(label_maps[column])
            escape_codes = set()
        order_position = {code: position for position, code in enumerate(ordered_codes)}
        mapping: dict[object, int] = {}
        for value in train_categories:
            code = category_to_text(value)
            if value == SIN_DATO:
                mapping[value] = ORDINAL_MISSING_SENTINEL
            elif code in escape_codes:
                mapping[value] = ORDINAL_ESCAPE_SENTINEL
            elif code in order_position:
                mapping[value] = order_position[code]
            else:
                raise AssertionError(f"Categoría sin semántica ordinal en {column}: {value}")

        train_encoded = X_train_b7[column].map(mapping)
        test_encoded = X_test_b7[column].map(mapping)
        if train_encoded.isna().any():
            raise AssertionError(f"Mapping train incompleto para {column}")
        unseen_mask = test_encoded.isna()
        test_encoded.loc[unseen_mask] = ORDINAL_UNKNOWN_SENTINEL
        ordinal_train[column] = train_encoded.astype("int16")
        ordinal_test[column] = test_encoded.astype("int16")
        ordinal_policy_checks.append(
            ordinal_test.loc[unseen_mask, column].eq(ORDINAL_UNKNOWN_SENTINEL).all()
        )

        for value in train_categories:
            code = category_to_text(value)
            encoded = mapping[value]
            if value == SIN_DATO:
                kind, position, label, fit = (
                    "missing_category_sentinel", "", "sin_dato (categoría fija B6)", "FIXED_RULE"
                )
            elif code in escape_codes:
                kind, position, label, fit = (
                    "semantic_escape_sentinel", "", label_maps[column][code], "SEMANTIC_FIXED_RULE"
                )
            else:
                kind, position, label, fit = (
                    "substantive_ordered_category", order_position[code], label_maps[column][code], "TRAIN_ONLY_SEMANTIC_ORDER"
                )
            ordinal_mapping_rows.append({
                "column": column,
                "category": code,
                "response_label": label,
                "category_kind": kind,
                "semantic_order_position": position,
                "encoded_value": encoded,
                "fit_partition": fit,
                "sentinel_is_not_part_of_order": kind != "substantive_ordered_category",
            })
        ordinal_mapping_rows.append({
            "column": column,
            "category": "__UNSEEN_TEST__",
            "response_label": "Categoría test no observada en train",
            "category_kind": "unknown_category_sentinel",
            "semantic_order_position": "",
            "encoded_value": ORDINAL_UNKNOWN_SENTINEL,
            "fit_partition": "FIXED_RULE",
            "sentinel_is_not_part_of_order": True,
        })
        for value in sorted_train_categories(X_test_b7.loc[unseen_mask, column]):
            unseen_rows.append({
                "column": column,
                "category": category_to_text(value),
                "test_n": int(X_test_b7[column].eq(value).sum()),
                "encoding_policy": f"ordinal_unknown_sentinel_{ORDINAL_UNKNOWN_SENTINEL}_no_refit",
            })

    onehot_train_blocks: list[pd.DataFrame] = []
    onehot_test_blocks: list[pd.DataFrame] = []
    vocabulary_rows: list[dict[str, object]] = []
    onehot_policy_checks: list[bool] = []
    for column in onehot_cols:
        categories = sorted_train_categories(X_train_b7[column])
        tokens = [safe_category_token(value) for value in categories]
        if len(tokens) != len(set(tokens)):
            raise AssertionError(f"Colisión one-hot en {column}")
        train_block = pd.DataFrame(index=X_train_b7.index)
        test_block = pd.DataFrame(index=X_test_b7.index)
        for position, (value, token) in enumerate(zip(categories, tokens, strict=True)):
            output_column = f"{column}__{token}"
            train_block[output_column] = X_train_b7[column].eq(value).astype("uint8")
            test_block[output_column] = X_test_b7[column].eq(value).astype("uint8")
            vocabulary_rows.append({
                "column": column,
                "category": category_to_text(value),
                "category_type": type(value).__name__,
                "vocabulary_position": position,
                "output_column": output_column,
                "fit_partition": "train",
                "is_sin_dato": value == SIN_DATO,
            })
        seen_mask = X_test_b7[column].isin(categories)
        unseen_mask = ~seen_mask
        onehot_policy_checks.extend([
            train_block.sum(axis=1).eq(1).all(),
            test_block.loc[seen_mask].sum(axis=1).eq(1).all(),
            test_block.loc[unseen_mask].sum(axis=1).eq(0).all(),
        ])
        for value in sorted_train_categories(X_test_b7.loc[unseen_mask, column]):
            unseen_rows.append({
                "column": column,
                "category": category_to_text(value),
                "test_n": int(X_test_b7[column].eq(value).sum()),
                "encoding_policy": "onehot_all_zero_handle_unknown_no_refit",
            })
        onehot_train_blocks.append(train_block)
        onehot_test_blocks.append(test_block)

    onehot_train = pd.concat(onehot_train_blocks, axis=1)
    onehot_test = pd.concat(onehot_test_blocks, axis=1)
    ordinal_mappings = pd.DataFrame(ordinal_mapping_rows)
    onehot_vocabularies = pd.DataFrame(vocabulary_rows)
    unseen_test = pd.DataFrame(
        unseen_rows, columns=["column", "category", "test_n", "encoding_policy"]
    )
    X_train_b8 = pd.concat([
        X_train_b7[CONTINUOUS_COLS].copy(),
        X_train_b7[[INDICATOR_COL]].astype("int8").copy(),
        ordinal_train,
        onehot_train,
    ], axis=1)
    X_test_b8 = pd.concat([
        X_test_b7[CONTINUOUS_COLS].copy(),
        X_test_b7[[INDICATOR_COL]].astype("int8").copy(),
        ordinal_test,
        onehot_test,
    ], axis=1)
    if not X_train_b8.columns.is_unique or not X_test_b8.columns.is_unique:
        raise AssertionError("Columnas B8 patch duplicadas.")

    inventory_rows: list[dict[str, object]] = []
    for position, column in enumerate(CONTINUOUS_COLS + [INDICATOR_COL]):
        inventory_rows.append({
            "output_column": column,
            "source_column": column,
            "output_position": position,
            "output_kind": "continuous_numeric" if column in CONTINUOUS_COLS else "binary_numeric",
            "encoding": "unchanged_scaled_B7" if column in CONTINUOUS_COLS else "unchanged_indicator",
            "source_category": "",
            "audit_status": "CONFIRMED",
        })
    position = len(inventory_rows)
    for column in final_ordinal_cols:
        inventory_rows.append({
            "output_column": column,
            "source_column": column,
            "output_position": position,
            "output_kind": "ordinal_numeric",
            "encoding": "ordinal_semantic_mapping_patch",
            "source_category": "",
            "audit_status": AUDIT_STATUS,
        })
        position += 1
    vocab_by_output = onehot_vocabularies.set_index("output_column")
    for output_column in onehot_train.columns:
        inventory_rows.append({
            "output_column": output_column,
            "source_column": vocab_by_output.loc[output_column, "column"],
            "output_position": position,
            "output_kind": "onehot_numeric",
            "encoding": "onehot_train_vocabulary",
            "source_category": vocab_by_output.loc[output_column, "category"],
            "audit_status": AUDIT_STATUS,
        })
        position += 1
    feature_inventory = pd.DataFrame(inventory_rows)

    previous_output_feature_n = 445
    encoding_summary = pd.DataFrame([{
        "input_feature_n": X_train_b7.shape[1],
        "previous_output_feature_n": previous_output_feature_n,
        "output_feature_n": X_train_b8.shape[1],
        "previous_ordinal_input_n": len(PREVIOUS_ORDINAL_COLS),
        "final_ordinal_input_n": len(final_ordinal_cols),
        "previous_onehot_input_n": len(categorical_cols) - len(PREVIOUS_ORDINAL_COLS),
        "final_onehot_input_n": len(onehot_cols),
        "output_ordinal_columns_n": ordinal_train.shape[1],
        "output_onehot_columns_n": onehot_train.shape[1],
        "changed_ordinal_to_nominal_n": 0,
        "changed_nominal_to_ordinal_n": len(UNKNOWN_TO_ORDINAL),
        "unseen_test_categories_n": len(unseen_test),
        "unseen_test_observations_n": int(unseen_test["test_n"].sum()),
        "missing_train_after": int(X_train_b8.isna().sum().sum()),
        "missing_test_after": int(X_test_b8.isna().sum().sum()),
        "patch_reason": PATCH_REASON,
    }])

    with tempfile.TemporaryDirectory(prefix="b8_patch_stage_", dir=BASE_DIR) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        staged = {
            REVIEW_PATH.name: temp_dir / REVIEW_PATH.name,
            ENCODING_PLAN_PATH.name: temp_dir / ENCODING_PLAN_PATH.name,
            UNKNOWN_PROPOSALS_PATH.name: temp_dir / UNKNOWN_PROPOSALS_PATH.name,
            ORDINAL_MAPPINGS_PATH.name: temp_dir / ORDINAL_MAPPINGS_PATH.name,
            ONEHOT_VOCABULARIES_PATH.name: temp_dir / ONEHOT_VOCABULARIES_PATH.name,
            UNSEEN_TEST_PATH.name: temp_dir / UNSEEN_TEST_PATH.name,
            FEATURE_INVENTORY_PATH.name: temp_dir / FEATURE_INVENTORY_PATH.name,
            ENCODING_SUMMARY_PATH.name: temp_dir / ENCODING_SUMMARY_PATH.name,
            MANIFEST_PATH.name: temp_dir / MANIFEST_PATH.name,
            X_TRAIN_PATH.name: temp_dir / X_TRAIN_PATH.name,
            X_TEST_PATH.name: temp_dir / X_TEST_PATH.name,
            Y_TRAIN_PATH.name: temp_dir / Y_TRAIN_PATH.name,
            Y_TEST_PATH.name: temp_dir / Y_TEST_PATH.name,
        }
        ordinal_review.to_csv(staged[REVIEW_PATH.name], index=False, encoding="utf-8-sig")
        encoding_plan.to_csv(staged[ENCODING_PLAN_PATH.name], index=False, encoding="utf-8-sig")
        unknown_proposals.to_csv(staged[UNKNOWN_PROPOSALS_PATH.name], index=False, encoding="utf-8-sig")
        ordinal_mappings.to_csv(staged[ORDINAL_MAPPINGS_PATH.name], index=False, encoding="utf-8-sig")
        onehot_vocabularies.to_csv(staged[ONEHOT_VOCABULARIES_PATH.name], index=False, encoding="utf-8-sig")
        unseen_test.to_csv(staged[UNSEEN_TEST_PATH.name], index=False, encoding="utf-8-sig")
        feature_inventory.to_csv(staged[FEATURE_INVENTORY_PATH.name], index=False, encoding="utf-8-sig")
        encoding_summary.to_csv(staged[ENCODING_SUMMARY_PATH.name], index=False, encoding="utf-8-sig")
        X_train_b8.to_pickle(staged[X_TRAIN_PATH.name])
        X_test_b8.to_pickle(staged[X_TEST_PATH.name])
        shutil.copy2(B7_Y_TRAIN_PATH, staged[Y_TRAIN_PATH.name])
        shutil.copy2(B7_Y_TEST_PATH, staged[Y_TEST_PATH.name])

        output_pickle_names = [X_TRAIN_PATH.name, X_TEST_PATH.name, Y_TRAIN_PATH.name, Y_TEST_PATH.name]
        manifest = {
            "block": "B8",
            "source_block": "B7",
            "source_output_hashes": source_hashes,
            "PATCH_REASON": PATCH_REASON,
            "PATCH_SOURCE": PATCH_SOURCE,
            "semantic_review_file": REVIEW_PATH.name,
            "previous_ordinal_audited_n": len(PREVIOUS_ORDINAL_COLS),
            "changed_ordinal_to_nominal": [],
            "changed_nominal_to_ordinal": UNKNOWN_TO_ORDINAL,
            "train_n": len(X_train_b8),
            "test_n": len(X_test_b8),
            "input_feature_n": X_train_b7.shape[1],
            "output_feature_n": X_train_b8.shape[1],
            "continuous_columns_unchanged": CONTINUOUS_COLS,
            "indicator_unchanged": INDICATOR_COL,
            "ordinal_input_columns": final_ordinal_cols,
            "onehot_input_columns": onehot_cols,
            "ordinal_missing_sentinel": ORDINAL_MISSING_SENTINEL,
            "ordinal_escape_sentinel": ORDINAL_ESCAPE_SENTINEL,
            "ordinal_unknown_test_sentinel": ORDINAL_UNKNOWN_SENTINEL,
            "sentinels_are_not_part_of_semantic_order": True,
            "onehot_unknown_test_policy": "all_zero_block_handle_unknown_ignore",
            "FIT_PARTITION": "TRAIN_ONLY",
            "TEST_USED_FOR_FIT": False,
            "TRANSFORM_PARTITIONS": ["train", "test"],
            "TEST_FROZEN_AFTER_B5": True,
            "target_used_for_encoding": False,
            "DISCRETIZATION_B8": "NONE",
            "IMC_SOURCE_FOR_B9": IMC_SOURCE_FOR_B9,
            "q50_included": True,
            "q51_included": True,
            "q50_q51_status": Q50_Q51_STATUS,
            "unknown_categorical_audit_status": AUDIT_STATUS,
            "B10_STATUS": "SMOTE_DEMONSTRATION_ONLY",
            "DOWNSTREAM_TRAIN_SOURCE": "B9_ORIGINAL_TRAIN",
            "DOWNSTREAM_SMOTE_SOURCE_ALLOWED": False,
            "output_files": {
                name: {"sha256": file_sha256(staged[name]), "bytes": staged[name].stat().st_size}
                for name in output_pickle_names
            },
        }
        staged[MANIFEST_PATH.name].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        review_previous = ordinal_review.loc[ordinal_review["review_scope"].eq("PREVIOUS_ORDINAL")]
        mapping_by_column = ordinal_mappings.groupby("column", sort=False)
        output_train = X_train_b8.to_numpy(dtype="float64")
        output_test = X_test_b8.to_numpy(dtype="float64")

        def escape_ok(column: str, codes: list[str]) -> bool:
            rows = ordinal_mappings.loc[
                ordinal_mappings["column"].eq(column)
                & ordinal_mappings["category"].astype(str).isin(codes)
            ]
            return len(rows) == len(codes) and rows["encoded_value"].eq(ORDINAL_ESCAPE_SENTINEL).all()

        checks = {
            "train_44440": len(X_train_b8) == EXPECTED_TRAIN_N,
            "test_11111": len(X_test_b8) == EXPECTED_TEST_N,
            "frontera_B7_intacta": source_hashes == expected_hashes
            and X_train_b8.index.equals(X_train_b7.index)
            and X_test_b8.index.equals(X_test_b7.index),
            "y_intacto": y_train_b7.equals(pd.read_pickle(staged[Y_TRAIN_PATH.name]))
            and y_test_b7.equals(pd.read_pickle(staged[Y_TEST_PATH.name])),
            "test_frozen": b7_manifest["TEST_FROZEN_AFTER_B5"] is True,
            "ordinales_previas_41_auditadas": len(review_previous) == 41
            and set(review_previous["column"]) == set(PREVIOUS_ORDINAL_COLS),
            "cada_ordinal_tiene_labels": review_previous["response_labels"].str.len().gt(2).all(),
            "cada_ordinal_tiene_decision": review_previous["review_decision"].isin([
                "KEEP_ORDINAL", "CHANGE_TO_NOMINAL_ONEHOT", "REQUIRES_MANUAL_REVIEW"
            ]).all(),
            "ninguna_ordinal_solo_por_codigo": review_previous["semantic_structure"].eq("ORDERED_SCALE").all()
            and review_previous["reason"].str.len().gt(20).all(),
            "q59_ordinal": "q59" in final_ordinal_cols,
            "q60_ordinal": "q60" in final_ordinal_cols,
            "q59_no_se_escape": escape_ok("q59", ["7"])
            and mapping_by_column.get_group("q59").loc[
                lambda frame: frame["category"].eq(SIN_DATO), "encoded_value"
            ].eq(ORDINAL_MISSING_SENTINEL).all(),
            "q60_no_se_escape": escape_ok("q60", ["7"])
            and mapping_by_column.get_group("q60").loc[
                lambda frame: frame["category"].eq(SIN_DATO), "encoded_value"
            ].eq(ORDINAL_MISSING_SENTINEL).all(),
            "q69_revisada": "q69" in final_ordinal_cols and escape_ok("q69", ["1", "6"]),
            "q70_revisada": "q70" in final_ordinal_cols and escape_ok("q70", ["1"]),
            "q71_revisada": "q71" in final_ordinal_cols and escape_ok("q71", ["1"]),
            "q72_revisada": "q72" in final_ordinal_cols and escape_ok("q72", ["1"]),
            "q79_revisada": "q79" in final_ordinal_cols and escape_ok("q79", ["4"]),
            "q28_revisada": "q28" in onehot_cols,
            "q34_revisada": "q34" in onehot_cols,
            "q40_revisada": "q40" in onehot_cols,
            "q45_revisada": "q45" in onehot_cols,
            "onehot_vocab_train_only": onehot_vocabularies["fit_partition"].eq("train").all()
            and all(onehot_policy_checks),
            "ordinal_mappings_sin_test": not ordinal_mappings["fit_partition"].astype(str).str.contains("test", case=False).any()
            and all(ordinal_policy_checks),
            "target_no_usado": not OPERATION_FLAGS["target_used_for_typing"],
            "test_no_usado_fit": TEST_USED_FOR_FIT is False and not OPERATION_FLAGS["test_used_for_fit"],
            "salida_numerica": all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_b8.dtypes)
            and all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_test_b8.dtypes),
            "mismas_columnas_orden": X_train_b8.columns.equals(X_test_b8.columns),
            "missing_0_0": int(X_train_b8.isna().sum().sum()) == 0
            and int(X_test_b8.isna().sum().sum()) == 0,
            "inf_0_0": np.isfinite(output_train).all() and np.isfinite(output_test).all(),
            "leakage_fuera": not any(
                source in HARD_LEAKAGE_COLS
                or output in HARD_LEAKAGE_COLS
                or any(output.startswith(f"{blocked}__") for blocked in HARD_LEAKAGE_COLS)
                for source, output in zip(feature_inventory["source_column"], feature_inventory["output_column"], strict=True)
            ),
            "metadata_fuera": not any(source in METADATA_COLS for source in feature_inventory["source_column"]),
            "textos_fuera": not any(source.startswith("texto_") for source in feature_inventory["source_column"]),
            "q50_q51_presentes": {"q50", "q51"}.issubset(X_train_b8.columns),
            "sin_IMC": "imc" not in X_train_b8.columns and not OPERATION_FLAGS["imc"],
            "sin_SMOTE": not OPERATION_FLAGS["smote"],
            "sin_feature_selection": not OPERATION_FLAGS["feature_selection"],
            "sin_PCA": not OPERATION_FLAGS["pca"],
            "sin_modelo": not OPERATION_FLAGS["model"],
            "artefactos_staging_persistidos": all(path.exists() for path in staged.values()),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"B8 patch detenido antes de promover: {failed}")

        destinations = {
            REVIEW_PATH.name: REVIEW_PATH,
            ENCODING_PLAN_PATH.name: ENCODING_PLAN_PATH,
            UNKNOWN_PROPOSALS_PATH.name: UNKNOWN_PROPOSALS_PATH,
            ORDINAL_MAPPINGS_PATH.name: ORDINAL_MAPPINGS_PATH,
            ONEHOT_VOCABULARIES_PATH.name: ONEHOT_VOCABULARIES_PATH,
            UNSEEN_TEST_PATH.name: UNSEEN_TEST_PATH,
            FEATURE_INVENTORY_PATH.name: FEATURE_INVENTORY_PATH,
            ENCODING_SUMMARY_PATH.name: ENCODING_SUMMARY_PATH,
            MANIFEST_PATH.name: MANIFEST_PATH,
            X_TRAIN_PATH.name: X_TRAIN_PATH,
            X_TEST_PATH.name: X_TEST_PATH,
            Y_TRAIN_PATH.name: Y_TRAIN_PATH,
            Y_TEST_PATH.name: Y_TEST_PATH,
        }
        for name, destination in destinations.items():
            shutil.copy2(staged[name], destination)

        promotion_checks = {
            "canonical_artifacts_exist": all(path.exists() for path in destinations.values()),
            "canonical_pickle_hashes_match_manifest": all(
                file_sha256(BASE_DIR / name) == manifest["output_files"][name]["sha256"]
                for name in output_pickle_names
            ),
            "canonical_pickles_aligned": pd.read_pickle(X_TRAIN_PATH).index.equals(pd.read_pickle(Y_TRAIN_PATH).index)
            and pd.read_pickle(X_TEST_PATH).index.equals(pd.read_pickle(Y_TEST_PATH).index),
            "canonical_train_test_same_columns": pd.read_pickle(X_TRAIN_PATH).columns.equals(pd.read_pickle(X_TEST_PATH).columns),
        }
        if not all(promotion_checks.values()):
            failed = [name for name, passed in promotion_checks.items() if not passed]
            raise AssertionError(f"Promoción B8 patch inválida: {failed}")

    print("=== REVISIÓN SEMÁNTICA B8 ===")
    print(f"Ordinales previas auditadas: {len(PREVIOUS_ORDINAL_COLS)}")
    print("Ordinal→nominal: ninguna")
    print(f"Nominal→ordinal: {UNKNOWN_TO_ORDINAL}")
    print(f"Nominal sin cambio: {UNKNOWN_KEEP_NOMINAL}")
    print(ordinal_review[["column", "semantic_structure", "review_decision", "new_encoding"]].to_string(index=False))

    print("\n=== B8 PATCH: TIPOS Y SALIDA ===")
    print("Antes: ordinales=41; one-hot inputs=105; output features=445")
    print(f"Después: ordinales={len(final_ordinal_cols)}; one-hot inputs={len(onehot_cols)}; one-hot outputs={onehot_train.shape[1]}")
    print(f"X_train_b8: {X_train_b8.shape}")
    print(f"X_test_b8: {X_test_b8.shape}")
    print(f"q59/q60: ORDINAL_WITH_SENTINEL; No sé={ORDINAL_ESCAPE_SENTINEL}; sin_dato={ORDINAL_MISSING_SENTINEL}")
    print("q69/q70/q71/q72/q79: ORDINAL_WITH_SENTINEL")
    print(f"Unseen test categories: {len(unseen_test)}; observations: {int(unseen_test['test_n'].sum())}")
    print("FIT one-hot vocabularies: TRAIN ONLY")
    print("FIT ordinal mappings: TRAIN ONLY / SEMANTIC FIXED ORDER")
    print("TRANSFORM encoders: TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")

    print("\n=== CHECKS B8 PATCH ===")
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] {name}")
    for name, passed in promotion_checks.items():
        print(f"[{'OK' if passed else 'FALLA'}] promotion_{name}")
    print(f"Resultado: {sum(checks.values())}/{len(checks)} checks B8 OK; promoción {sum(promotion_checks.values())}/{len(promotion_checks)} OK.")
    print(f"PATCH_REASON = {PATCH_REASON}")
    print(f"PATCH_SOURCE = {PATCH_SOURCE}")
    print("B8 patch promovido a artefactos canónicos; no se ejecutó B9 dentro de este script.")


if __name__ == "__main__":
    main()
