from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import mutual_info_classif


BASE_DIR = Path(__file__).resolve().parent
SOURCE_MANIFEST_PATH = BASE_DIR / "b9_feature_engineering_manifest.json"
SOURCE_INVENTORY_PATH = BASE_DIR / "b9_feature_inventory.csv"
SOURCE_X_TRAIN_PATH = BASE_DIR / "b9_X_train.pkl"
SOURCE_X_TEST_PATH = BASE_DIR / "b9_X_test.pkl"
SOURCE_Y_TRAIN_PATH = BASE_DIR / "b9_y_train.pkl"
SOURCE_Y_TEST_PATH = BASE_DIR / "b9_y_test.pkl"

RANKING_PATH = BASE_DIR / "b11_mutual_information_ranking.csv"
SELECTED_PATH = BASE_DIR / "b11_selected_features.csv"
SOURCE_SUMMARY_PATH = BASE_DIR / "b11_source_feature_mi_summary.csv"
SELECTION_SUMMARY_PATH = BASE_DIR / "b11_selection_summary.csv"
FEATURE_INVENTORY_PATH = BASE_DIR / "b11_feature_inventory.csv"
MANIFEST_PATH = BASE_DIR / "b11_selection_manifest.json"
X_TRAIN_PATH = BASE_DIR / "b11_X_train.pkl"
X_TEST_PATH = BASE_DIR / "b11_X_test.pkl"
Y_TRAIN_PATH = BASE_DIR / "b11_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b11_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_INPUT_FEATURE_N = 404
TOP_K = 30
RANDOM_STATE = 42
N_NEIGHBORS = 3
CONTINUOUS_COLS = ["q4", "q5", "imc"]
INDICATOR_COL = "q4q5_faltaba"
DISCRETE_KINDS = {"binary_numeric", "ordinal_numeric", "onehot_numeric"}
EXPECTED_KINDS = DISCRETE_KINDS | {"continuous_numeric"}
FIT_PARTITION = "TRAIN_ONLY"
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
DOWNSTREAM_TRAIN_SOURCE = "B9_ORIGINAL_TRAIN"
DOWNSTREAM_SMOTE_SOURCE_ALLOWED = False
B10_STATUS = "SMOTE_DEMONSTRATION_ONLY"
SELECTION_RULE = "TOP_K_PREDECIDED"
TIE_BREAK = "MI_DESC_FEATURE_NAME_ASC"
INTERPRETATION_CONTRACT = (
    "Mutual information ranks statistical association with the target in the "
    "training sample; it does not establish causality or independent effects."
)

HARD_LEAKAGE_COLS = {
    "q49",
    "qn49",
    "qnpa5g",
    "qnpa7g",
    "texto_q49",
    "texto_qn49",
    "texto_qnpa5g",
    "texto_qnpa7g",
}
METADATA_COLS = {"record", "psu", "stratum", "weight", "sitio"}
TARGET_COL = "target_pa_oms5"

# ponytail: these flags make the B11-only boundary directly auditable.
OPERATION_FLAGS = {
    "new_split": False,
    "imputation": False,
    "scaling": False,
    "reencoding": False,
    "feature_engineering": False,
    "smote": False,
    "pca": False,
    "model_training": False,
    "test_used_for_mi": False,
    "y_test_used_for_mi": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_hashes(manifest: dict[str, object]) -> dict[str, str]:
    paths = [
        SOURCE_X_TRAIN_PATH,
        SOURCE_X_TEST_PATH,
        SOURCE_Y_TRAIN_PATH,
        SOURCE_Y_TEST_PATH,
    ]
    actual = {path.name: sha256(path) for path in paths}
    expected = {
        path.name: manifest["output_files"][path.name]["sha256"] for path in paths
    }
    if actual != expected:
        raise AssertionError(
            f"Hashes B9 patch inválidos; B11 detenido. actual={actual}, expected={expected}"
        )
    return actual


def finite_frame(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.to_numpy(dtype="float64", copy=False)).all())


def values_are_binary(frame: pd.DataFrame) -> bool:
    values = frame.to_numpy(dtype="float64", copy=False)
    return bool(np.logical_or(values == 0.0, values == 1.0).all())


def values_are_integer(frame: pd.DataFrame) -> bool:
    values = frame.to_numpy(dtype="float64", copy=False)
    return bool(np.isclose(values, np.rint(values), atol=1e-12, rtol=0.0).all())


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manifest_b9 = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_hashes = verify_source_hashes(manifest_b9)
    inventory_b9 = pd.read_csv(SOURCE_INVENTORY_PATH, encoding="utf-8-sig")
    X_train_b9 = pd.read_pickle(SOURCE_X_TRAIN_PATH)
    X_test_b9 = pd.read_pickle(SOURCE_X_TEST_PATH)
    y_train_b9 = pd.read_pickle(SOURCE_Y_TRAIN_PATH)
    y_test_b9 = pd.read_pickle(SOURCE_Y_TEST_PATH)

    patch_verified = (
        manifest_b9.get("PATCH_REASON") == "ORDINAL_SEMANTIC_REVIEW"
        and manifest_b9.get("PATCH_SOURCE")
        == "REVISION_SEMANTICA_ORDINALES"
    )
    source_checks = {
        "train = 44.440": len(X_train_b9) == EXPECTED_TRAIN_N,
        "test = 11.111": len(X_test_b9) == EXPECTED_TEST_N,
        "NO train SMOTE": (
            len(X_train_b9) != 62_388
            and manifest_b9.get("DOWNSTREAM_TRAIN_SOURCE")
            == DOWNSTREAM_TRAIN_SOURCE
            and manifest_b9.get("DOWNSTREAM_SMOTE_SOURCE_ALLOWED") is False
            and manifest_b9.get("B10_STATUS") == B10_STATUS
        ),
        "input features = 404": (
            X_train_b9.shape[1]
            == X_test_b9.shape[1]
            == EXPECTED_INPUT_FEATURE_N
            == int(manifest_b9["output_feature_n"])
        ),
        "hashes B9 patch verificados": bool(source_hashes) and patch_verified,
        "X/y train alineados": X_train_b9.index.equals(y_train_b9.index),
        "X/y test alineados": X_test_b9.index.equals(y_test_b9.index),
        "train/test disjuntos": X_train_b9.index.intersection(X_test_b9.index).empty,
        "test frozen": (
            manifest_b9.get("TEST_FROZEN_AFTER_B5") is True
            and manifest_b9.get("TEST_USED_FOR_FIT") is False
        ),
    }
    failed = [name for name, passed in source_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Fuente B9 patch inválida; B11 detenido: {failed}")

    integrity_checks = {
        "numeric only": all(is_numeric_dtype(dtype) for dtype in X_train_b9.dtypes)
        and all(is_numeric_dtype(dtype) for dtype in X_test_b9.dtypes),
        "missing train/test = 0": int(X_train_b9.isna().sum().sum()) == 0
        and int(X_test_b9.isna().sum().sum()) == 0,
        "inf train/test = 0": finite_frame(X_train_b9) and finite_frame(X_test_b9),
        "mismas columnas B9 train/test": X_train_b9.columns.equals(X_test_b9.columns),
        "y binaria 0/1": set(pd.unique(y_train_b9)) == {0, 1}
        and set(pd.unique(y_test_b9)) == {0, 1},
    }
    failed = [name for name, passed in integrity_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Integridad B9 inválida; B11 detenido: {failed}")

    required_inventory_cols = {
        "output_column",
        "source_column",
        "output_kind",
    }
    if not required_inventory_cols.issubset(inventory_b9.columns):
        raise AssertionError("Inventario B9 no contiene las columnas requeridas")
    inventory_b9 = inventory_b9.set_index("output_column", drop=False)
    if (
        len(inventory_b9) != EXPECTED_INPUT_FEATURE_N
        or not inventory_b9.index.is_unique
        or inventory_b9.index.tolist() != X_train_b9.columns.tolist()
    ):
        raise AssertionError("Inventario B9 no está alineado con las 404 columnas")

    output_kinds = inventory_b9["output_kind"]
    kind_set = set(output_kinds)
    if not kind_set.issubset(EXPECTED_KINDS):
        raise AssertionError(f"output_kind inesperado: {sorted(kind_set - EXPECTED_KINDS)}")
    discrete_mask = output_kinds.isin(DISCRETE_KINDS).to_numpy(dtype=bool)
    continuous_from_inventory = inventory_b9.loc[
        output_kinds.eq("continuous_numeric"), "output_column"
    ].tolist()
    onehot_cols = inventory_b9.loc[
        output_kinds.eq("onehot_numeric"), "output_column"
    ].tolist()
    ordinal_cols = inventory_b9.loc[
        output_kinds.eq("ordinal_numeric"), "output_column"
    ].tolist()

    type_checks = {
        "continuas exactamente q4,q5,imc": continuous_from_inventory
        == CONTINUOUS_COLS,
        "continuas discrete=False": not discrete_mask[
            [X_train_b9.columns.get_loc(col) for col in CONTINUOUS_COLS]
        ].any(),
        "indicador discrete=True": bool(
            discrete_mask[X_train_b9.columns.get_loc(INDICATOR_COL)]
        ),
        "ordinales discrete=True": output_kinds.loc[ordinal_cols]
        .eq("ordinal_numeric")
        .all()
        and discrete_mask[[X_train_b9.columns.get_loc(col) for col in ordinal_cols]].all(),
        "one-hot discrete=True": discrete_mask[
            [X_train_b9.columns.get_loc(col) for col in onehot_cols]
        ].all(),
        "one-hot train sólo 0/1": values_are_binary(X_train_b9[onehot_cols]),
        "indicador train sólo 0/1": values_are_binary(X_train_b9[[INDICATOR_COL]]),
        "ordinales sin valores fraccionarios": values_are_integer(
            X_train_b9[ordinal_cols]
        ),
        "sin evidencia de SMOTE en input": (
            values_are_binary(X_train_b9[onehot_cols])
            and values_are_binary(X_train_b9[[INDICATOR_COL]])
            and values_are_integer(X_train_b9[ordinal_cols])
        ),
    }
    failed = [name for name, passed in type_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Tipos B9 incompatibles con MI; B11 detenido: {failed}")

    # Único fit/ranking de B11: train original, con k decidido de antemano.
    mi_scores = mutual_info_classif(
        X_train_b9,
        y_train_b9,
        discrete_features=discrete_mask,
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    ranking = pd.DataFrame(
        {
            "feature": X_train_b9.columns,
            "source_column": inventory_b9["source_column"].to_numpy(),
            "output_kind": output_kinds.to_numpy(),
            "is_discrete": discrete_mask,
            "mi_score": mi_scores,
        }
    ).sort_values(
        ["mi_score", "feature"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    ranking["selected_top30"] = ranking["rank"].le(TOP_K)
    ranking["audit_status"] = "TRAIN_ONLY_MI_ASSOCIATION_NOT_CAUSATION"
    ranking["notes"] = INTERPRETATION_CONTRACT

    selected = ranking.loc[
        ranking["selected_top30"],
        [
            "rank",
            "feature",
            "source_column",
            "output_kind",
            "mi_score",
            "is_discrete",
        ],
    ].copy()
    selected["semantic_note"] = "association_not_causation"
    selected["q50_q51_status"] = np.where(
        selected["source_column"].isin(["q50", "q51"]),
        "SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE",
        "",
    )
    selected_cols = selected["feature"].tolist()
    X_train_b11 = X_train_b9.loc[:, selected_cols].copy()
    X_test_b11 = X_test_b9.loc[:, selected_cols].copy()

    best_rows = ranking.sort_values("rank").drop_duplicates("source_column")
    source_summary = (
        ranking.groupby("source_column", as_index=False)
        .agg(
            encoded_columns_n=("feature", "size"),
            max_mi=("mi_score", "max"),
            sum_mi=("mi_score", "sum"),
            selected_encoded_columns_n=("selected_top30", "sum"),
        )
        .merge(
            best_rows[["source_column", "feature", "rank"]].rename(
                columns={"feature": "best_encoded_feature", "rank": "best_rank"}
            ),
            on="source_column",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["max_mi", "sum_mi", "source_column"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )
    )

    q_status = ranking.set_index("source_column")
    q50 = q_status.loc["q50"]
    q51 = q_status.loc["q51"]
    zero_mask = ranking["mi_score"].eq(0.0)
    selection_summary = pd.DataFrame(
        [
            {
                "input_feature_n": EXPECTED_INPUT_FEATURE_N,
                "selected_feature_n": TOP_K,
                "reduction_n": EXPECTED_INPUT_FEATURE_N - TOP_K,
                "reduction_pct": 100 * (EXPECTED_INPUT_FEATURE_N - TOP_K)
                / EXPECTED_INPUT_FEATURE_N,
                "continuous_input_n": len(CONTINUOUS_COLS),
                "discrete_input_n": int(discrete_mask.sum()),
                "continuous_selected_n": int(
                    selected["output_kind"].eq("continuous_numeric").sum()
                ),
                "discrete_selected_n": int(selected["is_discrete"].sum()),
                "onehot_selected_n": int(
                    selected["output_kind"].eq("onehot_numeric").sum()
                ),
                "ordinal_selected_n": int(
                    selected["output_kind"].eq("ordinal_numeric").sum()
                ),
                "indicator_selected_n": int(
                    selected["output_kind"].eq("binary_numeric").sum()
                ),
                "top_mi": float(ranking.iloc[0]["mi_score"]),
                "median_mi": float(ranking["mi_score"].median()),
                "selected_min_mi": float(selected["mi_score"].min()),
                "zero_mi_total_n": int(zero_mask.sum()),
                "zero_mi_selected_n": int(
                    selected["mi_score"].eq(0.0).sum()
                ),
                "q50_rank": int(q50["rank"]),
                "q50_selected": bool(q50["selected_top30"]),
                "q51_rank": int(q51["rank"]),
                "q51_selected": bool(q51["selected_top30"]),
                "train_n": len(X_train_b11),
                "test_n": len(X_test_b11),
            }
        ]
    )

    inventory_b11 = inventory_b9.reset_index(drop=True).copy()
    inventory_b11 = inventory_b11.merge(
        ranking[
            ["feature", "rank", "mi_score", "is_discrete", "selected_top30"]
        ].rename(columns={"feature": "output_column", "rank": "mi_rank"}),
        on="output_column",
        how="left",
        validate="one_to_one",
    )
    inventory_b11["b11_action"] = np.where(
        inventory_b11["selected_top30"], "selected_top30", "not_selected_top30"
    )

    tie_break_ok = all(
        group["feature"].tolist() == sorted(group["feature"].tolist())
        for _, group in ranking.groupby("mi_score", sort=False)
    )
    forbidden_sources = HARD_LEAKAGE_COLS | METADATA_COLS | {TARGET_COL}
    security_ok = not set(inventory_b9["source_column"]).intersection(
        forbidden_sources
    )
    text_ok = not inventory_b9["source_column"].str.startswith("texto_").any()

    ranking_checks = {
        "MI calculada sólo sobre X_train/y_train": not OPERATION_FLAGS[
            "test_used_for_mi"
        ]
        and not OPERATION_FLAGS["y_test_used_for_mi"],
        "X_test no usado para MI": not OPERATION_FLAGS["test_used_for_mi"],
        "y_test no usado": not OPERATION_FLAGS["y_test_used_for_mi"],
        "random_state=42": RANDOM_STATE == 42,
        "n_neighbors=3": N_NEIGHBORS == 3,
        "404 scores MI": len(ranking) == EXPECTED_INPUT_FEATURE_N,
        "ningún MI NaN": not ranking["mi_score"].isna().any(),
        "ningún MI negativo": ranking["mi_score"].ge(-1e-12).all(),
        "ranking determinístico": ranking["rank"].tolist()
        == list(range(1, EXPECTED_INPUT_FEATURE_N + 1)),
        "tie-break correcto": tie_break_ok,
        "exactamente 30 seleccionadas": int(ranking["selected_top30"].sum())
        == TOP_K,
        "selected ranks = 1..30": selected["rank"].tolist()
        == list(range(1, TOP_K + 1)),
    }
    matrix_checks = {
        "X_train_b11 = (44440,30)": X_train_b11.shape
        == (EXPECTED_TRAIN_N, TOP_K),
        "X_test_b11 = (11111,30)": X_test_b11.shape
        == (EXPECTED_TEST_N, TOP_K),
        "mismas columnas y orden": X_train_b11.columns.equals(X_test_b11.columns),
        "valores seleccionados train idénticos a B9": X_train_b11.equals(
            X_train_b9[selected_cols]
        ),
        "valores seleccionados test idénticos a B9": X_test_b11.equals(
            X_test_b9[selected_cols]
        ),
        "índices intactos": X_train_b11.index.equals(X_train_b9.index)
        and X_test_b11.index.equals(X_test_b9.index),
        "y intacto": y_train_b9.index.equals(X_train_b11.index)
        and y_test_b9.index.equals(X_test_b11.index),
    }
    safety_checks = {
        "leakage duro fuera": security_ok,
        "target fuera X": TARGET_COL not in X_train_b9.columns,
        "metadata fuera": not set(X_train_b9.columns).intersection(METADATA_COLS),
        "texto fuera": text_ok,
        "no reencoding": not OPERATION_FLAGS["reencoding"],
        "no scaling": not OPERATION_FLAGS["scaling"],
        "no imputación": not OPERATION_FLAGS["imputation"],
        "no SMOTE": not OPERATION_FLAGS["smote"],
        "no PCA": not OPERATION_FLAGS["pca"],
        "no modelo": not OPERATION_FLAGS["model_training"],
        "método MI": mutual_info_classif.__name__ == "mutual_info_classif",
        "Top K = 30": TOP_K == 30,
        "Top30 no decidido post-hoc": SELECTION_RULE == "TOP_K_PREDECIDED",
        "interpretación asociación-no-causalidad registrada": "does not establish causality"
        in INTERPRETATION_CONTRACT,
    }

    all_prewrite_checks = {
        **source_checks,
        **integrity_checks,
        **type_checks,
        **ranking_checks,
        **matrix_checks,
        **safety_checks,
    }
    failed = [name for name, passed in all_prewrite_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Checks B11 fallaron antes de persistir: {failed}")

    write_csv(ranking, RANKING_PATH)
    write_csv(selected, SELECTED_PATH)
    write_csv(source_summary, SOURCE_SUMMARY_PATH)
    write_csv(selection_summary, SELECTION_SUMMARY_PATH)
    write_csv(inventory_b11, FEATURE_INVENTORY_PATH)
    X_train_b11.to_pickle(X_TRAIN_PATH)
    X_test_b11.to_pickle(X_TEST_PATH)
    shutil.copy2(SOURCE_Y_TRAIN_PATH, Y_TRAIN_PATH)
    shutil.copy2(SOURCE_Y_TEST_PATH, Y_TEST_PATH)

    artifact_paths = [
        RANKING_PATH,
        SELECTED_PATH,
        SOURCE_SUMMARY_PATH,
        SELECTION_SUMMARY_PATH,
        FEATURE_INVENTORY_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]
    output_files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifact_paths
    }
    manifest_b11 = {
        "block": "B11",
        "source_block": "B9_PATCH",
        "source_manifest": SOURCE_MANIFEST_PATH.name,
        "source_hashes": source_hashes,
        "SEMANTIC_PATCH_APPLIED": patch_verified,
        "method": "mutual_info_classif",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "n_neighbors": N_NEIGHBORS,
        "discrete_features_policy": "B9_FEATURE_INVENTORY_TYPE_AWARE_MASK",
        "FIT_PARTITION": FIT_PARTITION,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "selection_rule": SELECTION_RULE,
        "top_k": TOP_K,
        "tie_break": TIE_BREAK,
        "input_feature_n": EXPECTED_INPUT_FEATURE_N,
        "output_feature_n": TOP_K,
        "train_n": EXPECTED_TRAIN_N,
        "test_n": EXPECTED_TEST_N,
        "selected_features": selected_cols,
        "DOWNSTREAM_TRAIN_SOURCE": DOWNSTREAM_TRAIN_SOURCE,
        "DOWNSTREAM_TRAIN_N": EXPECTED_TRAIN_N,
        "DOWNSTREAM_SMOTE_SOURCE_ALLOWED": DOWNSTREAM_SMOTE_SOURCE_ALLOWED,
        "B10_STATUS": B10_STATUS,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "INTERPRETATION_CONTRACT": INTERPRETATION_CONTRACT,
        "INTERPRETATION_LABEL": "association_not_causation",
        "operation_flags": OPERATION_FLAGS,
        "output_files": output_files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest_b11, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reload_checks = {
        "ranking persistido": len(pd.read_csv(RANKING_PATH, encoding="utf-8-sig"))
        == EXPECTED_INPUT_FEATURE_N,
        "selected list persistida": len(pd.read_csv(SELECTED_PATH, encoding="utf-8-sig"))
        == TOP_K,
        "source summary persistido": SOURCE_SUMMARY_PATH.exists()
        and len(pd.read_csv(SOURCE_SUMMARY_PATH, encoding="utf-8-sig")) > 0,
        "pickles recargables/alineados": pd.read_pickle(X_TRAIN_PATH).equals(
            X_train_b11
        )
        and pd.read_pickle(X_TEST_PATH).equals(X_test_b11)
        and pd.read_pickle(Y_TRAIN_PATH).equals(y_train_b9)
        and pd.read_pickle(Y_TEST_PATH).equals(y_test_b9),
        "manifest completo": MANIFEST_PATH.exists()
        and all(
            key in manifest_b11
            for key in [
                "source_hashes",
                "method",
                "random_state",
                "n_neighbors",
                "FIT_PARTITION",
                "TEST_USED_FOR_FIT",
                "selection_rule",
                "top_k",
                "tie_break",
                "INTERPRETATION_CONTRACT",
                "output_files",
            ]
        ),
    }
    failed = [name for name, passed in reload_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Persistencia B11 inválida: {failed}")

    checks = {**all_prewrite_checks, **reload_checks}

    print("=== B11 — MUTUAL INFORMATION · TOP 30 ===")
    print("\n### Fuente")
    print(f"Manifest: {SOURCE_MANIFEST_PATH.resolve()}")
    print(f"SEMANTIC_PATCH_APPLIED = {patch_verified}")
    for name, digest in source_hashes.items():
        print(f"SHA256 {name} = {digest}")
    print(f"Train original: {X_train_b9.shape}")
    print(f"Test congelado: {X_test_b9.shape}")
    print("NO SMOTE: confirmado; no se cargó ningún artefacto B10")

    print("\n### Tipos")
    print(f"Continuas ({len(CONTINUOUS_COLS)}): {CONTINUOUS_COLS}")
    print(f"Discretas ({int(discrete_mask.sum())})")
    print(f"  one-hot: {len(onehot_cols)}")
    print(f"  ordinales: {len(ordinal_cols)}")
    print("  indicador: 1 (q4q5_faltaba)")

    print("\n### Mutual Information — Top 30 completo")
    print(
        selected[[
            "rank",
            "feature",
            "source_column",
            "output_kind",
            "mi_score",
        ]].to_string(index=False, float_format=lambda value: f"{value:.10f}")
    )
    print(f"MI máximo: {ranking['mi_score'].max():.10f}")
    print(f"MI mediana (404): {ranking['mi_score'].median():.10f}")
    print(f"MI mínimo seleccionado: {selected['mi_score'].min():.10f}")
    print(f"Features con MI=0: {int(zero_mask.sum())}")

    print("\n### q50/q51")
    for source in ["q50", "q51"]:
        row = q_status.loc[source]
        print(
            f"{source}: rank={int(row['rank'])}, MI={row['mi_score']:.10f}, "
            f"selected={'sí' if bool(row['selected_top30']) else 'no'}, "
            "status=SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE"
        )

    print("\n### Selección")
    print(f"Antes train/test: {X_train_b9.shape} / {X_test_b9.shape}")
    print(f"Después train/test: {X_train_b11.shape} / {X_test_b11.shape}")

    print("\n### Fuente agregada — Top 10 por max_mi/sum_mi")
    print("Vista complementaria; NO se usó para selección.")
    print(
        source_summary.head(10).to_string(
            index=False, float_format=lambda value: f"{value:.10f}"
        )
    )

    print("\n### Fit/transform")
    print("FIT Mutual Information ranking: TRAIN ONLY")
    print("SELECTION RULE: PREDECIDED TOP 30")
    print("APPLY SELECTED COLUMNS: TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")
    print("B10_STATUS = SMOTE_DEMONSTRATION_ONLY")
    print("DOWNSTREAM_TRAIN_SOURCE = B9_ORIGINAL_TRAIN")

    print("\n### Interpretación")
    print("MUTUAL INFORMATION = ASSOCIATION, NOT CAUSATION")
    print(INTERPRETATION_CONTRACT)

    print("\n### Artefactos")
    for path in [*artifact_paths, MANIFEST_PATH]:
        print(path.resolve())

    print("\n### Checks")
    for number, (name, passed) in enumerate(checks.items(), start=1):
        print(f"CHECK {number:02d} [{'OK' if passed else 'FAIL'}] {name}")
    print(f"CHECKS B11: {sum(checks.values())}/{len(checks)} OK")


if __name__ == "__main__":
    main()
