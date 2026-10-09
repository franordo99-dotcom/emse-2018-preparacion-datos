from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
SOURCE_MANIFEST_PATH = BASE_DIR / "b11_selection_manifest.json"
SOURCE_SELECTED_PATH = BASE_DIR / "b11_selected_features.csv"
SOURCE_INVENTORY_PATH = BASE_DIR / "b11_feature_inventory.csv"
SOURCE_X_TRAIN_PATH = BASE_DIR / "b11_X_train.pkl"
SOURCE_X_TEST_PATH = BASE_DIR / "b11_X_test.pkl"
SOURCE_Y_TRAIN_PATH = BASE_DIR / "b11_y_train.pkl"
SOURCE_Y_TEST_PATH = BASE_DIR / "b11_y_test.pkl"

SCALER_PARAMS_PATH = BASE_DIR / "b12_pca_scaler_params.csv"
EXPLAINED_VARIANCE_PATH = BASE_DIR / "b12_pca_explained_variance.csv"
THRESHOLDS_PATH = BASE_DIR / "b12_pca_thresholds.csv"
LOADINGS_PATH = BASE_DIR / "b12_pca_loadings.csv"
TOP_CONTRIBUTORS_PATH = BASE_DIR / "b12_pca_top_contributors.csv"
TRADEOFFS_PATH = BASE_DIR / "b12_pca_tradeoffs.csv"
SUMMARY_PATH = BASE_DIR / "b12_pca_summary.csv"
MANIFEST_PATH = BASE_DIR / "b12_pca_manifest.json"
SCREE_PATH = BASE_DIR / "b12_scree_plot.png"
CUMULATIVE_PATH = BASE_DIR / "b12_cumulative_variance.png"
X_TRAIN_PATH = BASE_DIR / "b12_X_train_pca.pkl"
X_TEST_PATH = BASE_DIR / "b12_X_test_pca.pkl"
Y_TRAIN_PATH = BASE_DIR / "b12_y_train.pkl"
Y_TEST_PATH = BASE_DIR / "b12_y_test.pkl"

EXPECTED_TRAIN_N = 44_440
EXPECTED_TEST_N = 11_111
EXPECTED_INPUT_FEATURE_N = 30
PCA_VARIANCE_THRESHOLD = 0.90
PCA_COMPONENT_RULE = "MIN_COMPONENTS_FOR_90_PERCENT_TRAIN_VARIANCE"
PCA_LIMITATION = (
    "PCA is a linear Euclidean method designed for quantitative variables. "
    "Here it is applied to a mixed encoded space containing continuous, "
    "ordinal and one-hot features. This is a didactic approximation; MCA "
    "would be more natural for predominantly categorical data but is outside "
    "the scope of this TP/manual."
)
B11_SELECTION_USED_TARGET = True
B12_PCA_DIRECT_TARGET_USE = False
PCA_INPUT_IS_TARGET_INFORMED = True
PCA_SIGN_INDETERMINACY = True
TEST_USED_FOR_FIT = False
TEST_FROZEN_AFTER_B5 = True
B10_STATUS = "SMOTE_DEMONSTRATION_ONLY"
DOWNSTREAM_TRAIN_SOURCE = "B9_ORIGINAL_TRAIN"
DOWNSTREAM_SMOTE_SOURCE_ALLOWED = False
PCA_SCALER_FIT = "TRAIN_ONLY"
PCA_FIT_PARTITION = "TRAIN_ONLY"
PCA_SVD_SOLVER = "full"
TOL = 1e-10

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

# Banderas de estado: documentan qué operaciones hace B12 y cuáles no.
OPERATION_FLAGS = {
    "new_split": False,
    "new_imputation": False,
    "reencoding": False,
    "smote": False,
    "new_feature_selection": False,
    "model_training": False,
    "target_passed_to_pca": False,
    "test_used_for_scaler_fit": False,
    "test_used_for_pca_fit": False,
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
        SOURCE_SELECTED_PATH,
        SOURCE_INVENTORY_PATH,
    ]
    actual = {path.name: sha256(path) for path in paths}
    expected = {
        path.name: manifest["output_files"][path.name]["sha256"] for path in paths
    }
    if actual != expected:
        raise AssertionError(
            f"Hashes B11 inválidos; B12 detenido. actual={actual}, expected={expected}"
        )
    return actual


def finite_frame(frame: pd.DataFrame) -> bool:
    return bool(np.isfinite(frame.to_numpy(dtype="float64", copy=False)).all())


def minimum_components(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def threshold_is_minimal(cumulative: np.ndarray, k: int, threshold: float) -> bool:
    reached = cumulative[k - 1] >= threshold
    previous_below = k == 1 or cumulative[k - 2] < threshold
    return bool(reached and previous_below)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def create_plots(
    explained_ratio: np.ndarray,
    cumulative: np.ndarray,
    k80: int,
    k90: int,
    k95: int,
) -> None:
    components = np.arange(1, len(explained_ratio) + 1)
    ink = "#202124"
    blue = "#35618f"
    blue_dark = "#23415f"
    gold = "#c58b2a"
    grid = "#d9dee3"

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    ax.bar(
        components,
        explained_ratio,
        color=blue,
        edgecolor=blue_dark,
        linewidth=0.55,
    )
    ax.set_title(
        "PCA — varianza explicada por componente",
        loc="left",
        color=ink,
        pad=28,
    )
    ax.text(
        0,
        1.01,
        "Top 30 B11 · PCA ajustado sólo con train (n=44.440)",
        transform=ax.transAxes,
        color="#5f6368",
        fontsize=10,
    )
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Proporción de varianza explicada")
    ax.set_xticks(components, [f"PC{i}" for i in components])
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SCREE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    ax.plot(
        components,
        cumulative,
        color=blue,
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="Varianza acumulada",
    )
    for threshold, style in [(0.80, ":"), (0.90, "--"), (0.95, "-.")]:
        ax.axhline(
            threshold,
            color="#697177",
            linewidth=1.0,
            linestyle=style,
            label=f"{threshold:.0%}",
        )
    for k, threshold in [(k80, 0.80), (k90, 0.90), (k95, 0.95)]:
        ax.scatter(
            [k],
            [cumulative[k - 1]],
            color=gold,
            edgecolor="#7a5418",
            linewidth=0.7,
            s=55,
            zorder=4,
        )
        ax.annotate(
            f"k={k}",
            (k, cumulative[k - 1]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
            color=ink,
        )
    ax.set_title(
        "PCA — varianza explicada acumulada",
        loc="left",
        color=ink,
        pad=28,
    )
    ax.text(
        0,
        1.01,
        "Umbrales definidos sobre train; regla oficial: mínimo k para 90%",
        transform=ax.transAxes,
        color="#5f6368",
        fontsize=10,
    )
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Proporción acumulada")
    ax.set_xlim(1, len(components))
    ax.set_ylim(0, 1.03)
    ax.set_xticks(components)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.grid(axis="both", color=grid, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(CUMULATIVE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manifest_b11 = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_hashes = verify_source_hashes(manifest_b11)
    selected_b11 = pd.read_csv(SOURCE_SELECTED_PATH, encoding="utf-8-sig")
    inventory_b11 = pd.read_csv(SOURCE_INVENTORY_PATH, encoding="utf-8-sig")
    X_train_b11 = pd.read_pickle(SOURCE_X_TRAIN_PATH)
    X_test_b11 = pd.read_pickle(SOURCE_X_TEST_PATH)
    y_train_b11 = pd.read_pickle(SOURCE_Y_TRAIN_PATH)
    y_test_b11 = pd.read_pickle(SOURCE_Y_TEST_PATH)

    selected_b11 = selected_b11.sort_values("rank", kind="mergesort")
    selected_cols = selected_b11["feature"].tolist()
    selected_inventory = inventory_b11.loc[
        inventory_b11["output_column"].isin(selected_cols)
    ].set_index("output_column").loc[selected_cols]

    source_checks = {
        "train=44.440": len(X_train_b11) == EXPECTED_TRAIN_N,
        "test=11.111": len(X_test_b11) == EXPECTED_TEST_N,
        "input=30": X_train_b11.shape[1]
        == X_test_b11.shape[1]
        == EXPECTED_INPUT_FEATURE_N,
        "B10 no es fuente / no SMOTE input": (
            len(X_train_b11) != 62_388
            and manifest_b11.get("DOWNSTREAM_TRAIN_N") == EXPECTED_TRAIN_N
            and manifest_b11.get("B10_STATUS") == B10_STATUS
            and manifest_b11.get("DOWNSTREAM_TRAIN_SOURCE")
            == DOWNSTREAM_TRAIN_SOURCE
            and manifest_b11.get("DOWNSTREAM_SMOTE_SOURCE_ALLOWED") is False
        ),
        "hashes B11 verificados": bool(source_hashes),
        "columnas coinciden selected B11": (
            len(selected_b11) == EXPECTED_INPUT_FEATURE_N
            and selected_b11["rank"].tolist() == list(range(1, 31))
            and X_train_b11.columns.tolist() == selected_cols
            and X_test_b11.columns.tolist() == selected_cols
            and selected_inventory.index.tolist() == selected_cols
        ),
        "X/y alineados": X_train_b11.index.equals(y_train_b11.index)
        and X_test_b11.index.equals(y_test_b11.index),
        "train/test disjuntos": X_train_b11.index.intersection(
            X_test_b11.index
        ).empty,
        "test frozen": manifest_b11.get("TEST_FROZEN_AFTER_B5") is True
        and manifest_b11.get("TEST_USED_FOR_FIT") is False,
    }
    failed = [name for name, passed in source_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Fuente B11 inválida; B12 detenido: {failed}")

    train_values = X_train_b11.to_numpy(dtype="float64", copy=True)
    test_values = X_test_b11.to_numpy(dtype="float64", copy=True)
    train_variance = train_values.var(axis=0, ddof=0)
    integrity_checks = {
        "numeric": all(is_numeric_dtype(dtype) for dtype in X_train_b11.dtypes)
        and all(is_numeric_dtype(dtype) for dtype in X_test_b11.dtypes),
        "NaN 0/0": int(X_train_b11.isna().sum().sum()) == 0
        and int(X_test_b11.isna().sum().sum()) == 0,
        "inf 0/0": finite_frame(X_train_b11) and finite_frame(X_test_b11),
        "ninguna feature train constante": bool((train_variance > 0).all()),
    }
    failed = [name for name, passed in integrity_checks.items() if not passed]
    if failed:
        constant_cols = X_train_b11.columns[train_variance == 0].tolist()
        raise AssertionError(
            f"Integridad B11 inválida; B12 detenido: {failed}; constantes={constant_cols}"
        )

    scaler_pca = StandardScaler()
    scaler_pca.fit(train_values)
    train_scaled = scaler_pca.transform(train_values)
    test_scaled = scaler_pca.transform(test_values)
    train_scaled_mean = train_scaled.mean(axis=0)
    train_scaled_std = train_scaled.std(axis=0, ddof=0)

    scaler_params = pd.DataFrame(
        {
            "feature": selected_cols,
            "mean_train": scaler_pca.mean_,
            "variance_train": scaler_pca.var_,
            "scale_train": scaler_pca.scale_,
        }
    )
    scaler_checks = {
        "scaler fit train-only": PCA_SCALER_FIT == "TRAIN_ONLY",
        "scaler test no fit": not OPERATION_FLAGS["test_used_for_scaler_fit"],
        "30 medias train scaled≈0": bool(
            np.all(np.abs(train_scaled_mean) < TOL)
        ),
        "30 std train scaled≈1": bool(
            np.all(np.abs(train_scaled_std - 1.0) < TOL)
        ),
    }
    failed = [name for name, passed in scaler_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Scaler PCA inválido; B12 detenido: {failed}")

    pca = PCA(n_components=None, svd_solver=PCA_SVD_SOLVER)
    pca.fit(train_scaled)
    train_pc_full = pca.transform(train_scaled)
    test_pc_full = pca.transform(test_scaled)

    explained_variance = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)
    k80 = minimum_components(cumulative, 0.80)
    k90 = minimum_components(cumulative, PCA_VARIANCE_THRESHOLD)
    k95 = minimum_components(cumulative, 0.95)
    variance_at_k90 = float(cumulative[k90 - 1])

    all_pc_cols = [f"PC{i}" for i in range(1, EXPECTED_INPUT_FEATURE_N + 1)]
    output_pc_cols = all_pc_cols[:k90]
    X_train_b12_pca = pd.DataFrame(
        train_pc_full[:, :k90], index=X_train_b11.index, columns=output_pc_cols
    )
    X_test_b12_pca = pd.DataFrame(
        test_pc_full[:, :k90], index=X_test_b11.index, columns=output_pc_cols
    )

    explained_variance_df = pd.DataFrame(
        {
            "component": all_pc_cols,
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance": cumulative,
        }
    )
    thresholds_df = pd.DataFrame(
        [
            {"threshold": 0.80, "k": k80, "variance_at_k": cumulative[k80 - 1]},
            {"threshold": 0.90, "k": k90, "variance_at_k": cumulative[k90 - 1]},
            {"threshold": 0.95, "k": k95, "variance_at_k": cumulative[k95 - 1]},
        ]
    )

    loadings = pd.DataFrame(pca.components_.T, columns=all_pc_cols)
    loadings.insert(0, "output_kind", selected_b11["output_kind"].to_numpy())
    loadings.insert(0, "source_column", selected_b11["source_column"].to_numpy())
    loadings.insert(0, "input_feature", selected_cols)

    contributor_rows: list[dict[str, object]] = []
    for component_number in range(1, min(k90, 10) + 1):
        component = f"PC{component_number}"
        component_loadings = loadings[
            ["input_feature", "source_column", "output_kind", component]
        ].copy()
        component_loadings["abs_loading"] = component_loadings[component].abs()
        component_loadings = component_loadings.sort_values(
            ["abs_loading", "input_feature"],
            ascending=[False, True],
            kind="mergesort",
        ).head(5)
        for rank, row in enumerate(component_loadings.itertuples(index=False), 1):
            contributor_rows.append(
                {
                    "component": component,
                    "rank_within_component": rank,
                    "feature": row.input_feature,
                    "source_column": row.source_column,
                    "output_kind": row.output_kind,
                    "loading": getattr(row, component),
                    "abs_loading": row.abs_loading,
                }
            )
    top_contributors = pd.DataFrame(contributor_rows)

    tradeoffs = pd.DataFrame(
        [
            ("VENTAJA", 1, "Reduce dimensionalidad."),
            ("VENTAJA", 2, "Elimina correlación lineal entre componentes."),
            ("VENTAJA", 3, "Concentra varianza en menos ejes."),
            ("VENTAJA", 4, "Puede facilitar visualización y síntesis."),
            (
                "DESVENTAJA",
                1,
                "Pierde interpretación directa de las variables originales.",
            ),
            ("DESVENTAJA", 2, "Es un método lineal."),
            (
                "DESVENTAJA",
                3,
                "La varianza explicada no equivale a relevancia para el target.",
            ),
            (
                "DESVENTAJA",
                4,
                "Sobre one-hot y ordinales la geometría PCA es una aproximación.",
            ),
            (
                "DESVENTAJA",
                5,
                "La selección previa B11 hace que la entrada esté informada por el target.",
            ),
        ],
        columns=["type", "order", "description"],
    )

    kind_counts = selected_b11["output_kind"].value_counts()
    summary = pd.DataFrame(
        [
            {
                "input_feature_n": EXPECTED_INPUT_FEATURE_N,
                "train_n": EXPECTED_TRAIN_N,
                "test_n": EXPECTED_TEST_N,
                "pca_scaler": "StandardScaler",
                "pca_scaler_fit_partition": "train",
                "full_components_n": EXPECTED_INPUT_FEATURE_N,
                "k80": k80,
                "k90": k90,
                "k95": k95,
                "official_variance_threshold": PCA_VARIANCE_THRESHOLD,
                "official_components_n": k90,
                "variance_at_k90": variance_at_k90,
                "reduction_n": EXPECTED_INPUT_FEATURE_N - k90,
                "reduction_pct": 100
                * (EXPECTED_INPUT_FEATURE_N - k90)
                / EXPECTED_INPUT_FEATURE_N,
                "selected_input_continuous_n": int(
                    kind_counts.get("continuous_numeric", 0)
                ),
                "selected_input_ordinal_n": int(
                    kind_counts.get("ordinal_numeric", 0)
                ),
                "selected_input_onehot_n": int(
                    kind_counts.get("onehot_numeric", 0)
                ),
                "selected_input_indicator_n": int(
                    kind_counts.get("binary_numeric", 0)
                ),
                "test_used_for_fit": TEST_USED_FOR_FIT,
                "pca_input_target_informed": PCA_INPUT_IS_TARGET_INFORMED,
            }
        ]
    )

    identity = np.eye(EXPECTED_INPUT_FEATURE_N)
    pca_checks = {
        "PCA fit train-only": PCA_FIT_PARTITION == "TRAIN_ONLY",
        "y no usado": not OPERATION_FLAGS["target_passed_to_pca"],
        "PCA test no fit": not OPERATION_FLAGS["test_used_for_pca_fit"],
        "30 componentes completas": pca.components_.shape == (30, 30),
        "explained variance ratios no negativas": bool((explained_ratio >= 0).all()),
        "suma ratios≈1": bool(np.isclose(explained_ratio.sum(), 1.0, atol=TOL)),
        "acumulada monótona": bool(np.all(np.diff(cumulative) >= -TOL)),
        "acumulada final≈1": bool(np.isclose(cumulative[-1], 1.0, atol=TOL)),
        "componentes ortonormales": bool(
            np.allclose(pca.components_ @ pca.components_.T, identity, atol=TOL)
        ),
        "scores train centrados≈0": bool(
            np.all(np.abs(train_pc_full.mean(axis=0)) < TOL)
        ),
    }
    k_checks = {
        "threshold fijo=0.90": PCA_VARIANCE_THRESHOLD == 0.90,
        "k80 mínimo correcto": threshold_is_minimal(cumulative, k80, 0.80),
        "k90 mínimo correcto": threshold_is_minimal(cumulative, k90, 0.90),
        "k95 mínimo correcto": threshold_is_minimal(cumulative, k95, 0.95),
        "k90 decidido sólo con train": not OPERATION_FLAGS["test_used_for_pca_fit"],
        "output features=k90": X_train_b12_pca.shape[1] == k90,
    }
    matrix_checks = {
        "train PCA=(44440,k90)": X_train_b12_pca.shape
        == (EXPECTED_TRAIN_N, k90),
        "test PCA=(11111,k90)": X_test_b12_pca.shape
        == (EXPECTED_TEST_N, k90),
        "mismas columnas/orden": X_train_b12_pca.columns.equals(
            X_test_b12_pca.columns
        ),
        "índices intactos": X_train_b12_pca.index.equals(X_train_b11.index)
        and X_test_b12_pca.index.equals(X_test_b11.index),
        "y intacto": y_train_b11.index.equals(X_train_b12_pca.index)
        and y_test_b11.index.equals(X_test_b12_pca.index),
        "missing PCA 0/0": int(X_train_b12_pca.isna().sum().sum()) == 0
        and int(X_test_b12_pca.isna().sum().sum()) == 0,
        "inf PCA 0/0": finite_frame(X_train_b12_pca)
        and finite_frame(X_test_b12_pca),
    }

    forbidden_sources = HARD_LEAKAGE_COLS | METADATA_COLS | {TARGET_COL}
    selected_sources = set(selected_b11["source_column"])
    interpretation_checks = {
        "limitación mixed encoded registrada": "mixed encoded space"
        in PCA_LIMITATION,
        "asociación/target previo B11 reconocido": B11_SELECTION_USED_TARGET
        and PCA_INPUT_IS_TARGET_INFORMED,
        "PCA direct target use=false": B12_PCA_DIRECT_TARGET_USE is False,
        "sign indeterminacy registrada": PCA_SIGN_INDETERMINACY,
        "no nombres causales/componentes inventados": output_pc_cols
        == [f"PC{i}" for i in range(1, k90 + 1)],
    }
    safety_checks = {
        "leakage fuera": not selected_sources.intersection(forbidden_sources),
        "no nueva imputación": not OPERATION_FLAGS["new_imputation"],
        "no reencoding": not OPERATION_FLAGS["reencoding"],
        "no SMOTE ejecutado": not OPERATION_FLAGS["smote"],
        "no feature selection nueva": not OPERATION_FLAGS["new_feature_selection"],
        "no modelo": not OPERATION_FLAGS["model_training"],
    }

    all_prewrite_checks = {
        **source_checks,
        **integrity_checks,
        **scaler_checks,
        **pca_checks,
        **k_checks,
        **matrix_checks,
        **interpretation_checks,
        **safety_checks,
    }
    failed = [name for name, passed in all_prewrite_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Checks B12 fallaron antes de persistir: {failed}")

    write_csv(scaler_params, SCALER_PARAMS_PATH)
    write_csv(explained_variance_df, EXPLAINED_VARIANCE_PATH)
    write_csv(thresholds_df, THRESHOLDS_PATH)
    write_csv(loadings, LOADINGS_PATH)
    write_csv(top_contributors, TOP_CONTRIBUTORS_PATH)
    write_csv(tradeoffs, TRADEOFFS_PATH)
    write_csv(summary, SUMMARY_PATH)
    X_train_b12_pca.to_pickle(X_TRAIN_PATH)
    X_test_b12_pca.to_pickle(X_TEST_PATH)
    shutil.copy2(SOURCE_Y_TRAIN_PATH, Y_TRAIN_PATH)
    shutil.copy2(SOURCE_Y_TEST_PATH, Y_TEST_PATH)
    create_plots(explained_ratio, cumulative, k80, k90, k95)

    artifact_paths = [
        SCALER_PARAMS_PATH,
        EXPLAINED_VARIANCE_PATH,
        THRESHOLDS_PATH,
        LOADINGS_PATH,
        TOP_CONTRIBUTORS_PATH,
        TRADEOFFS_PATH,
        SUMMARY_PATH,
        SCREE_PATH,
        CUMULATIVE_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]
    output_files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifact_paths
    }
    manifest_b12 = {
        "block": "B12",
        "source_block": "B11",
        "source_manifest": SOURCE_MANIFEST_PATH.name,
        "source_hashes": source_hashes,
        "train_n": EXPECTED_TRAIN_N,
        "test_n": EXPECTED_TEST_N,
        "input_feature_n": EXPECTED_INPUT_FEATURE_N,
        "output_feature_n": k90,
        "PCA_SCALER": "StandardScaler",
        "PCA_SCALER_FIT": PCA_SCALER_FIT,
        "PCA_METHOD": "sklearn.decomposition.PCA",
        "sklearn_version": sklearn.__version__,
        "PCA_SVD_SOLVER": PCA_SVD_SOLVER,
        "PCA_FIT_PARTITION": PCA_FIT_PARTITION,
        "TEST_USED_FOR_FIT": TEST_USED_FOR_FIT,
        "PCA_VARIANCE_THRESHOLD": PCA_VARIANCE_THRESHOLD,
        "PCA_COMPONENT_RULE": PCA_COMPONENT_RULE,
        "k80": k80,
        "k90": k90,
        "k95": k95,
        "B11_SELECTION_USED_TARGET": B11_SELECTION_USED_TARGET,
        "B12_PCA_DIRECT_TARGET_USE": B12_PCA_DIRECT_TARGET_USE,
        "PCA_INPUT_IS_TARGET_INFORMED": PCA_INPUT_IS_TARGET_INFORMED,
        "B10_STATUS": B10_STATUS,
        "DOWNSTREAM_TRAIN_SOURCE": DOWNSTREAM_TRAIN_SOURCE,
        "DOWNSTREAM_SMOTE_SOURCE_ALLOWED": DOWNSTREAM_SMOTE_SOURCE_ALLOWED,
        "TEST_FROZEN_AFTER_B5": TEST_FROZEN_AFTER_B5,
        "PCA_LIMITATION": PCA_LIMITATION,
        "PCA_SIGN_INDETERMINACY": PCA_SIGN_INDETERMINACY,
        "operation_flags": OPERATION_FLAGS,
        "output_files": output_files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest_b12, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reload_checks = {
        "scaler params persistidos": len(
            pd.read_csv(SCALER_PARAMS_PATH, encoding="utf-8-sig")
        )
        == EXPECTED_INPUT_FEATURE_N,
        "explained variance persistida": len(
            pd.read_csv(EXPLAINED_VARIANCE_PATH, encoding="utf-8-sig")
        )
        == EXPECTED_INPUT_FEATURE_N,
        "loadings persistidos": pd.read_csv(
            LOADINGS_PATH, encoding="utf-8-sig"
        ).shape
        == (EXPECTED_INPUT_FEATURE_N, EXPECTED_INPUT_FEATURE_N + 3),
        "thresholds persistidos": len(
            pd.read_csv(THRESHOLDS_PATH, encoding="utf-8-sig")
        )
        == 3,
        "tradeoffs persistidos": len(
            pd.read_csv(TRADEOFFS_PATH, encoding="utf-8-sig")
        )
        == 9,
        "scree generado": SCREE_PATH.exists() and SCREE_PATH.stat().st_size > 0,
        "acumulada generada": CUMULATIVE_PATH.exists()
        and CUMULATIVE_PATH.stat().st_size > 0,
        "pickles recargables/alineados": pd.read_pickle(X_TRAIN_PATH).equals(
            X_train_b12_pca
        )
        and pd.read_pickle(X_TEST_PATH).equals(X_test_b12_pca)
        and pd.read_pickle(Y_TRAIN_PATH).equals(y_train_b11)
        and pd.read_pickle(Y_TEST_PATH).equals(y_test_b11),
        "manifest completo": MANIFEST_PATH.exists()
        and all(
            key in manifest_b12
            for key in [
                "source_hashes",
                "PCA_SCALER_FIT",
                "PCA_METHOD",
                "PCA_FIT_PARTITION",
                "TEST_USED_FOR_FIT",
                "PCA_VARIANCE_THRESHOLD",
                "PCA_COMPONENT_RULE",
                "k80",
                "k90",
                "k95",
                "PCA_LIMITATION",
                "PCA_SIGN_INDETERMINACY",
                "output_files",
            ]
        ),
    }
    failed = [name for name, passed in reload_checks.items() if not passed]
    if failed:
        raise AssertionError(f"Persistencia B12 inválida: {failed}")

    checks = {**all_prewrite_checks, **reload_checks}

    print("=== B12 — PCA TRAIN-ONLY SOBRE TOP 30 B11 ===")
    print("\n### Fuente")
    print(f"Manifest B11: {SOURCE_MANIFEST_PATH.resolve()}")
    for name, digest in source_hashes.items():
        print(f"SHA256 {name} = {digest}")
    print(f"Train B11: {X_train_b11.shape}")
    print(f"Test B11: {X_test_b11.shape}")
    print("NO SMOTE: confirmado; no se cargó ningún artefacto B10")

    print("\n### PCA preprocessing")
    print("Scaler: sklearn.preprocessing.StandardScaler")
    print(f"Max |mean train scaled|: {np.abs(train_scaled_mean).max():.3e}")
    print(f"Max |std(ddof=0)-1| train scaled: {np.abs(train_scaled_std - 1).max():.3e}")
    print(f"Features con varianza cero: {int((train_variance == 0).sum())}")

    print("\n### Varianza explicada — 30 componentes")
    print(
        explained_variance_df[
            [
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.10f}")
    )
    print(f"k80 = {k80} (acumulada={cumulative[k80 - 1]:.10f})")
    print(f"k90 = {k90} (acumulada={variance_at_k90:.10f})")
    print(f"k95 = {k95} (acumulada={cumulative[k95 - 1]:.10f})")

    print("\n### Reducción oficial")
    print(f"Antes: {EXPECTED_INPUT_FEATURE_N}")
    print(f"Después: {k90}")
    print(
        f"Reducción: {EXPECTED_INPUT_FEATURE_N - k90} "
        f"({100 * (EXPECTED_INPUT_FEATURE_N - k90) / EXPECTED_INPUT_FEATURE_N:.4f}%)"
    )
    print(f"Train PCA: {X_train_b12_pca.shape}")
    print(f"Test PCA: {X_test_b12_pca.shape}")

    print("\n### Loadings — top 5 contributors PC1..PC5")
    print(
        top_contributors.loc[
            top_contributors["component"].isin(["PC1", "PC2", "PC3", "PC4", "PC5"])
        ].to_string(index=False, float_format=lambda value: f"{value:.10f}")
    )
    print("El signo es arbitrario; no implica valoración positiva/negativa.")

    print("\n### Fit/transform")
    print("PCA INPUT = B11 TOP 30 FROM ORIGINAL TRAIN")
    print("B10 STATUS = SMOTE_DEMONSTRATION_ONLY")
    print("FIT PCA-SCALER: TRAIN ONLY")
    print("TRANSFORM PCA-SCALER: TRAIN + TEST")
    print("FIT PCA: TRAIN ONLY")
    print("TRANSFORM PCA: TRAIN + TEST")
    print("TEST_USED_FOR_FIT = False")
    print("PCA VARIANCE THRESHOLD = 90% TRAIN CUMULATIVE VARIANCE")
    print("PCA INPUT IS TARGET-INFORMED BY B11 = True")
    print("PCA DIRECT TARGET USE = False")
    print("PCA ON MIXED ENCODED FEATURES = DIDACTIC APPROXIMATION")

    print("\n### Ventajas/desventajas")
    print(tradeoffs.to_string(index=False))

    print("\n### Limitación")
    print(f"PCA_LIMITATION = {PCA_LIMITATION}")
    print(
        "PCA se ajusta sin y, pero sobre un subconjunto previamente seleccionado "
        "en train según asociación con el target."
    )

    print("\n### Artefactos")
    for path in [*artifact_paths, MANIFEST_PATH]:
        print(path.resolve())

    print("\n### Checks")
    for number, (name, passed) in enumerate(checks.items(), start=1):
        print(f"CHECK {number:02d} [{'OK' if passed else 'FAIL'}] {name}")
    print(f"CHECKS B12: {sum(checks.values())}/{len(checks)} OK")
    print("B12 finalizado.")


if __name__ == "__main__":
    main()
