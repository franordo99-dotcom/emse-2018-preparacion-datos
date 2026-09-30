# Glosario del repositorio — TP Análisis de Datos · EMSE 2018

Este documento surge de una inspección estática del contenido existente en disco. No se ejecutó ningún script del pipeline ni se recalculó ningún resultado. Las descripciones se basan en imports, constantes, funciones, lecturas, escrituras, validaciones y manifests presentes en el repositorio.

## Estructura de carpetas

```text
Análisis de Datos/
├── README.md, GLOSARIO_REPO.md, .gitignore
├── b0_setup_emse.py ... b12_pca_emse.py
│   └── scripts del pipeline B0–B12; B8 tiene script base y patch semántico
├── b2_*.csv ... b13_*.{csv,md}
│   └── artefactos tabulares, manifests, matrices persistidas y documentación
├── b12_*.png
│   └── gráficos PCA guardados en la raíz
├── outputs/
│   ├── b2_figures/                 # 14 figuras de EDA
│   ├── b3_figures/                 # 8 figuras de faltantes
│   └── b4_figures/                 # 9 figuras de outliers
├── EMSE_DatosAbiertos.csv          # fuente de datos del pipeline
├── cuestionario-emse-2018.pdf      # material de referencia; no es autoridad automática de mapeo
├── d1_mi_top10_*                   # artefactos de entrega posteriores a B13, sin script visible
└── _no_entrega/                    # material ajeno o auxiliar, preservado e ignorado
    ├── CEIA_Analisis_de_datos/, heart-failure-eda/
    ├── wholesale_customers.csv, wholesale_customers_eda.ipynb
    ├── inspect_wholesale_customers.py, __pycache__/
    └── consigna PDF y manual DOCX
```

El repositorio Git de esta carpeta tiene actualmente cero archivos versionados: los archivos de entrega aparecen como no seguidos por Git. `_no_entrega/` está ignorado. Los dos proyectos anidados preservados allí poseen sus propios metadatos `.git` y no forman parte del pipeline EMSE B0–B13 documentado aquí.

## Bloques B0–B13

### B0 — Setup, carga y auditoría estructural

- **ETAPA:** B0 — Setup y carga del dataset.
- **SCRIPT(S):** `b0_setup_emse.py`.
- **QUÉ HACE:** localiza `EMSE_DatosAbiertos.csv`, lo carga con pandas y clasifica columnas mediante reglas nominales (`qN`, `texto_qN`, `qn*`, metadata y residuales). Resume memoria, tipos, `q4`/`q5`, faltantes conjuntos y posibles derivadas determinísticas de `q49`; sólo imprime el resultado.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`, localizado desde el directorio del script o el directorio de trabajo.
- **SALIDAS:** no persiste archivos; produce salida estándar.
- **CHECKS:** shape esperado; existencia de `q4`, `q5`, `q49`, `qnpa5g`; capas estructurales disjuntas y cobertura total; `qnpa5g`/`qnpa7g` dentro del mapa de leakage; `q50`/`q51` fuera del leakage duro; conversión numérica válida de `q4`/`q5`.

### B1 — Target y mapa de leakage

- **ETAPA:** B1 — Construcción del target y consolidación de leakage.
- **SCRIPT(S):** `b1_target_emse.py`.
- **QUÉ HACE:** reutiliza B0, reconstruye la semántica de `q49` mediante `texto_q49`, crea en memoria `target_pa_oms5` y contrasta su consistencia con `qnpa5g`. Caracteriza `qn49`, `qnpa5g`, `qnpa7g` y sus contrapartes textuales para excluir fuente y proxies del futuro conjunto de features.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`; funciones y constantes importadas desde `b0_setup_emse.py`.
- **SALIDAS:** no persiste archivos; produce salida estándar.
- **CHECKS:** shape y columnas requeridas; mapping `q49`→texto consistente; target limitado a 0/1/NaN y faltante exactamente cuando falta `q49`; `qnpa5g` no usado para construir el target; concordancia reportada; fuente/proxies excluidos; `q50`/`q51` no clasificados como leakage duro.

### B2 — EDA y diccionario de datos

- **ETAPA:** B2 — EDA descriptivo y data dictionary.
- **SCRIPT(S):** `b2_eda_dictionary_emse.py`.
- **QUÉ HACE:** reconstruye el universo supervisado desde el CSV, asigna roles y tipos estadísticos, y genera resúmenes numéricos, categóricos, de faltantes, target y PSU. Produce gráficos descriptivos sin imputar, seleccionar, transformar ni entrenar.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`; funciones/constantes de `b0_setup_emse.py` y `b1_target_emse.py`.
- **SALIDAS:** `b2_data_dictionary.csv`, `b2_numeric_summary.csv`, `b2_categorical_summary.csv`, `b2_missing_summary.csv`, `b2_psu_profile.csv`, `b2_numeric_by_target.csv`, `b2_categorical_by_target.csv`; 14 PNG en `outputs/b2_figures/`.
- **CHECKS:** shape crudo y universo supervisado; target 0/1; leakage y textos fuera de candidatas; `q4`/`q5` continuas; metadata separada; unicidad de `record`; constancia de `sitio`; diccionario con 309 columnas únicas; tipos ambiguos identificados; ausencia de split, imputación, scaling, encoding, SMOTE, PCA y selección; existencia de artefactos.

### B3 — Diagnóstico de faltantes

- **ETAPA:** B3 — Diagnóstico de mecanismos y patrones de ausencia.
- **SCRIPT(S):** `b3_missing_diagnosis_emse.py`.
- **QUÉ HACE:** parte de los resúmenes B2, reconstruye el universo supervisado y estudia la co-ausencia `q4`/`q5` por target, PSU, estrato y variables demográficas identificables. Evalúa candidatos de branching con criterios conservadores y calcula co-ocurrencia de faltantes sin imputar.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`, `b2_data_dictionary.csv`, `b2_missing_summary.csv`; funciones de B0/B1.
- **SALIDAS:** `b3_missing_diagnosis.csv`, `b3_q4q5_missing_by_target.csv`, `b3_q4q5_missing_by_psu.csv`, `b3_q4q5_missing_by_stratum.csv`, `b3_q4q5_missing_by_sex.csv`, `b3_q4q5_missing_by_age.csv`, `b3_q4q5_missing_by_grade.csv`, `b3_q4q5_evidence_summary.csv`, `b3_demographic_identification.csv`, `b3_branching_candidates.csv`, `b3_missing_cooccurrence.csv`; 8 PNG en `outputs/b3_figures/`.
- **CHECKS:** universo y valores originales preservados; máscaras `q4`/`q5` idénticas y conteo contractual; `qn40` totalmente faltante y marcado como vacío informativo; ninguna baja de filas/columnas ni transformación; `ESTRUCTURAL_DEMOSTRADO` sólo con match exacto; ausencia de `MCAR_CONFIRMADO`/`MNAR_CONFIRMADO`; evidencia informada para cada diagnóstico; contradicciones con MAR marcadas; artefactos B2 sin modificación y salidas existentes.

### B4 — Detección y política de outliers

- **ETAPA:** B4 — Diagnóstico descriptivo de outliers.
- **SCRIPT(S):** `b4_outlier_diagnosis_emse.py`.
- **QUÉ HACE:** calcula un IMC exclusivamente diagnóstico a partir de `q4`/`q5`, banderas IQR, ±3 desvíos y reglas de revisión de dominio. Compara métodos, inspecciona casos, tasas por target y coherencia numérica; conserva los valores sin modificarlos.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`; funciones de B0/B1.
- **SALIDAS:** `b4_outliers_iqr.csv`, `b4_outliers_z3.csv`, `b4_domain_review_flags.csv`, `b4_outlier_method_overlap.csv`, `b4_outlier_case_review.csv`, `b4_outlier_summary.csv`, `b4_outlier_flags_by_target.csv`, `b4_coherence_checks.csv`; 9 PNG en `outputs/b4_figures/`.
- **CHECKS:** shape/universo; `q4`/`q5` y faltantes preservados; IMC sólo en pares completos y cálculo coherente; ninguna imputación, eliminación, modificación, partición, transformación, selección o modelo; política `RETENER_Y_MARCAR_SIN_MODIFICAR`; errores demostrados con evidencia o conteo cero; artefactos existentes.

### B5 — Split y frontera de evaluación

- **ETAPA:** B5 — Split train/test estratificado y congelamiento del test.
- **SCRIPT(S):** `b5_split_emse.py`.
- **QUÉ HACE:** reconstruye target y features candidatas desde el CSV y el diccionario B2. Implementa un split estratificado individual con `numpy.random.default_rng`, semilla 42 y 20% de test; guarda membresía, inventario, resúmenes y manifest, pero no matrices X/y completas.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`, `b2_data_dictionary.csv`; funciones de B0/B1.
- **SALIDAS:** `b5_split_summary.csv`, `b5_target_distribution.csv`, `b5_psu_partition_profile.csv`, `b5_missing_partition_summary.csv`, `b5_feature_inventory.csv`, `b5_split_manifest.json`, `b5_train_records.csv`, `b5_test_records.csv`.
- **CHECKS:** universo, tamaños, disjunción/unión y ausencia de pérdidas; target binario y prevalencias; exclusión de leakage, textos, metadata y `qn40`; conservación de `q50`, `q51` y variables ambiguas; 148 candidatas (2 continuas/146 categóricas); alineación de X/y/sidecars y records; un único split no grupal; test congelado; hashes/membresía/semilla y artefactos.

### B6 — Imputación train-only

- **ETAPA:** B6 — Imputación y creación del indicador de ausencia.
- **SCRIPT(S):** `b6_imputation_emse.py`.
- **QUÉ HACE:** reconstruye exactamente la frontera B5 mediante los índices persistidos. Aprende medianas de `q4`/`q5` sólo en train, aplica esas medianas a train/test, reemplaza NaN categóricos por `sin_dato` y agrega `q4q5_faltaba` sin modificar los objetos raw.
- **ENTRADAS:** `EMSE_DatosAbiertos.csv`, `b5_split_manifest.json`, `b5_feature_inventory.csv`, `b5_train_records.csv`, `b5_test_records.csv`; funciones de B0/B1.
- **SALIDAS:** `b6_continuous_imputation_params.csv`, `b6_missing_before_after.csv`, `b6_indicator_summary.csv`, `b6_categorical_missing_fill_summary.csv`, `b6_feature_inventory.csv`, `b6_imputation_manifest.json`, `b6_X_train.pkl`, `b6_X_test.pkl`, `b6_y_train.pkl`, `b6_y_test.pkl`.
- **CHECKS:** hashes y membresía B5; test congelado y ausencia de nuevo split; 148→149 features con un único agregado; exclusiones y candidatas preservadas; co-ausencia exacta; medianas train-only; NaN final cero; valores observados sin cambios; indicador exacto; regla `sin_dato`; ausencia de transformaciones posteriores/modelo; persistencia y recarga alineada.

### B7 — StandardScaler train-only

- **ETAPA:** B7 — Escalado de `q4` y `q5`.
- **SCRIPT(S):** `b7_scaling_emse.py`.
- **QUÉ HACE:** carga las matrices B6 y estandariza sólo `q4`/`q5`, con `StandardScaler` de sklearn si está disponible o su equivalente matemático con `ddof=0`. Ajusta parámetros en train y transforma train/test; deja intactas las otras 147 columnas.
- **ENTRADAS:** `b5_split_manifest.json`, `b6_imputation_manifest.json`, `b6_feature_inventory.csv`, `b6_X_train.pkl`, `b6_X_test.pkl`, `b6_y_train.pkl`, `b6_y_test.pkl`.
- **SALIDAS:** `b7_scaler_params.csv`, `b7_scaling_summary.csv`, `b7_feature_inventory.csv`, `b7_scaling_manifest.json`, `b7_X_train.pkl`, `b7_X_test.pkl`, `b7_y_train.pkl`, `b7_y_test.pkl`.
- **CHECKS:** manifests/hashes, tamaños, índices, targets y faltantes B6; columnas escaladas exactamente `q4`/`q5`; fit train-only y `ddof=0`; media≈0/std≈1 en train; indicador, categóricas y `sin_dato` intactos; exclusiones vigentes; ninguna operación posterior; artefactos recargables.

### B8 — Encoding categórico y patch semántico

- **ETAPA:** B8 — Encoding train-only; revisión semántica posterior de ordinales.
- **SCRIPT(S):** `b8_encoding_emse.py`; `b8_patch_ordinal_semantic_review_emse.py`.
- **QUÉ HACE:** el script base construye mappings ordinales y vocabularios one-hot sólo con train, mantiene `q4`, `q5` y `q4q5_faltaba`, y maneja categorías de test no vistas sin refit. El patch relee B7, el diccionario B2 y labels del CSV, audita las 41 ordinales previas, modifica el tratamiento de variables señaladas y promueve una nueva versión canónica B8 después de validar un staging temporal.
- **ENTRADAS:** base: `b7_scaling_manifest.json`, `b7_feature_inventory.csv`, `b7_X_train.pkl`, `b7_X_test.pkl`, `b7_y_train.pkl`, `b7_y_test.pkl`, `b6_feature_inventory.csv`, `b2_data_dictionary.csv`. Patch: esos artefactos B7/B2, `EMSE_DatosAbiertos.csv` y el `b8_encoding_plan.csv` previo.
- **SALIDAS:** `b8_encoding_plan.csv`, `b8_unknown_categorical_proposals.csv`, `b8_ordinal_mappings.csv`, `b8_onehot_vocabularies.csv`, `b8_unseen_test_categories.csv`, `b8_feature_inventory.csv`, `b8_encoding_summary.csv`, `b8_encoding_manifest.json`, `b8_X_train.pkl`, `b8_X_test.pkl`, `b8_y_train.pkl`, `b8_y_test.pkl`; el patch agrega `b8_patch_ordinal_review.csv` y reemplaza los artefactos canónicos anteriores tras los checks.
- **CHECKS:** frontera/target B7 intactos; input de 149 features sin NaN; continuas e indicador sin recodificar; `q50`/`q51` y casos ambiguos documentados; vocabularios/mappings train-only y test sin fit; salida numérica/finita con columnas alineadas; leakage, metadata y textos fuera; ninguna discretización, IMC, SMOTE, selección, PCA o modelo. El patch añade auditoría completa de las 41 ordinales, labels/decisión semántica por variable y verificaciones específicas de `q59`, `q60`, `q69`–`q72`, `q79`, `q28`, `q34`, `q40` y `q45`.

### B9 — Feature engineering de IMC

- **ETAPA:** B9 — IMC físico y escalado train-only.
- **SCRIPT(S):** `b9_feature_engineering_emse.py`.
- **QUÉ HACE:** usa B8 como matriz predictiva y B6 sólo como fuente física imputada no escalada de altura/peso. Calcula `q5 / q4²`, aprende media/escala del IMC en train con `ddof=0`, transforma train/test y agrega únicamente la versión estandarizada `imc`.
- **ENTRADAS:** `b5_split_manifest.json`, `b6_imputation_manifest.json`, `b8_encoding_manifest.json`, `b8_feature_inventory.csv`, cuatro pickles B6 y cuatro pickles B8.
- **SALIDAS:** `b9_imc_raw_summary.csv`, `b9_imc_by_missing_indicator.csv`, `b9_imc_scaler_params.csv`, `b9_feature_inventory.csv`, `b9_feature_engineering_summary.csv`, `b9_feature_engineering_manifest.json`, `b9_X_train.pkl`, `b9_X_test.pkl`, `b9_y_train.pkl`, `b9_y_test.pkl`.
- **CHECKS:** hashes/índices/targets B6↔B8 y test congelado; `q4`/`q5` físicos positivos y completos; fuente y fórmula exactas; scaler del IMC train-only; salida finita; 403→404 con un único agregado; 403 columnas B8 intactas; exclusiones; ninguna nueva imputación, recodificación, discretización, tratamiento de outliers, SMOTE, PCA, selección o modelo; persistencia y recarga.

### B10 — SMOTE de demostración

- **ETAPA:** B10 — Remuestreo SMOTE train-only, rama demostrativa.
- **SCRIPT(S):** `b10_smote_emse.py`.
- **QUÉ HACE:** carga B9 y aplica `imblearn.over_sampling.SMOTE` sólo al train con estrategia `auto`, semilla 42 y 5 vecinos. Copia test byte por byte, distingue filas originales/sintéticas y documenta coordenadas fraccionarias que SMOTE puede introducir en el espacio one-hot/ordinal.
- **ENTRADAS:** `b9_feature_engineering_manifest.json`, `b9_feature_inventory.csv`, `b9_X_train.pkl`, `b9_X_test.pkl`, `b9_y_train.pkl`, `b9_y_test.pkl`.
- **SALIDAS:** `b10_class_distribution.csv`, `b10_sample_origin.csv`, `b10_smote_summary.csv`, `b10_smote_categorical_interpolation_diagnostic.csv`, `b10_feature_inventory.csv`, `b10_smote_manifest.json`, `b10_X_train_smote.pkl`, `b10_y_train_smote.pkl`, `b10_X_test.pkl`, `b10_y_test.pkl`.
- **CHECKS:** hashes/alineación y espacio de 446 columnas esperado por el script; SMOTE/configuración/train-only y clases finales balanceadas; test idéntico por hash, datos, índices y prevalencia; observaciones originales sin alteración y sintéticas identificables; ausencia de nuevas features y de operaciones posteriores; diagnóstico one-hot/ordinal/indicador; persistencia. **Estos artefactos son `SMOTE_DEMONSTRATION_ONLY` y no alimentan B11/B12.**

### B11 — Selección por Mutual Information

- **ETAPA:** B11 — Feature selection MI Top 30.
- **SCRIPT(S):** `b11_mutual_information_selection_emse.py`.
- **QUÉ HACE:** carga el B9 parcheado y calcula `mutual_info_classif` sólo en el train original, con máscara discreta derivada del inventario, semilla 42 y 3 vecinos. Ordena por MI descendente/nombre ascendente, retiene las 30 columnas prefijadas y aplica esa selección a test; MI mide asociación, no causalidad.
- **ENTRADAS:** `b9_feature_engineering_manifest.json`, `b9_feature_inventory.csv`, `b9_X_train.pkl`, `b9_X_test.pkl`, `b9_y_train.pkl`, `b9_y_test.pkl`.
- **SALIDAS:** `b11_mutual_information_ranking.csv`, `b11_selected_features.csv`, `b11_source_feature_mi_summary.csv`, `b11_selection_summary.csv`, `b11_feature_inventory.csv`, `b11_selection_manifest.json`, `b11_X_train.pkl`, `b11_X_test.pkl`, `b11_y_train.pkl`, `b11_y_test.pkl`.
- **CHECKS:** train original de 44.440 y no B10; hashes B9; matrices numéricas/finitas/alineadas; tipos discretos válidos sin fracciones de SMOTE; MI sólo en train con configuración contractual; 404 scores, ranking/tie-break y exactamente Top 30; matrices seleccionadas idénticas a columnas B9; exclusiones; sin reencoding/scaling/imputación/SMOTE/PCA/modelo; contrato asociación-no-causalidad; artefactos recargables.

### B12 — PCA train-only

- **ETAPA:** B12 — PCA sobre Top 30 B11.
- **SCRIPT(S):** `b12_pca_emse.py`.
- **QUÉ HACE:** estandariza las 30 columnas seleccionadas con un scaler específico ajustado sólo en train y ajusta un PCA completo de sklearn también sólo en train. Calcula curvas de varianza, k80/k90/k95, conserva oficialmente k90 componentes y genera loadings, contribuyentes y gráficos; el espacio mixto codificado se presenta como aproximación didáctica.
- **ENTRADAS:** `b11_selection_manifest.json`, `b11_selected_features.csv`, `b11_feature_inventory.csv`, `b11_X_train.pkl`, `b11_X_test.pkl`, `b11_y_train.pkl`, `b11_y_test.pkl`.
- **SALIDAS:** `b12_pca_scaler_params.csv`, `b12_pca_explained_variance.csv`, `b12_pca_thresholds.csv`, `b12_pca_loadings.csv`, `b12_pca_top_contributors.csv`, `b12_pca_tradeoffs.csv`, `b12_pca_summary.csv`, `b12_pca_manifest.json`, `b12_scree_plot.png`, `b12_cumulative_variance.png`, `b12_X_train_pca.pkl`, `b12_X_test_pca.pkl`, `b12_y_train.pkl`, `b12_y_test.pkl`.
- **CHECKS:** fuente/hashes/Top 30 B11 y ausencia de SMOTE; matrices numéricas/finitas sin columnas constantes; scaler train-only con medias≈0/std≈1; PCA train-only sin `y` ni test; 30 componentes, ratios/cumulada/ortonormalidad y scores centrados; mínimos k80/k90/k95 y umbral 90% decidido en train; shapes/índices/targets; limitación mixta, entrada informada por target en B11 e indeterminación de signo; artefactos y gráficos existentes.

### B13 — Consolidación documental

- **ETAPA:** B13 — Consolidación final de narrativa, cifras y trazabilidad metodológica.
- **SCRIPT(S):** **no existe ningún `b13_*.py` en disco**.
- **QUÉ HACE:** no puede reconstruirse una implementación desde código porque no hay script. Los artefactos existentes consolidan la narrativa, las cifras clave, la frontera de test y el pipeline textual, sin agregar fit ni transformaciones.
- **ENTRADAS:** no determinables desde un script. La documentación referencia artefactos de B5–B12, incluidos split, transformaciones, ranking MI, selección y resultados PCA.
- **SALIDAS:** `b13_narrative_master.md`, `b13_key_numbers.csv`, `b13_pipeline_final.md`, `b13_test_boundary_ledger.csv`.
- **CHECKS:** no hay asserts ejecutables visibles. Los documentos registran B10 demostrativo/no downstream, train downstream desde B9 original, test congelado, B11 Top 30 y B12 con 13 componentes.

## Inventario de artefactos

Se incluyen los artefactos técnicos del pipeline y los productos de entrega encontrados en disco. No se incluyen scripts `.py`, fuentes de datos/manuales, bytecode, material de preparación interna ni archivos de proyectos ajenos al TP EMSE.

| Bloque | Archivo | Tipo |
|---|---|---|
| B2 | `b2_categorical_by_target.csv` | csv |
| B2 | `b2_categorical_summary.csv` | csv |
| B2 | `b2_data_dictionary.csv` | csv |
| B2 | `b2_missing_summary.csv` | csv |
| B2 | `b2_numeric_by_target.csv` | csv |
| B2 | `b2_numeric_summary.csv` | csv |
| B2 | `b2_psu_profile.csv` | csv |
| B2 | `outputs/b2_figures/01_target_barras.png` | png |
| B2 | `outputs/b2_figures/02_q4_histograma.png` | png |
| B2 | `outputs/b2_figures/03_q5_histograma.png` | png |
| B2 | `outputs/b2_figures/04_q4_boxplot.png` | png |
| B2 | `outputs/b2_figures/05_q5_boxplot.png` | png |
| B2 | `outputs/b2_figures/06_q4_vs_target_boxplot.png` | png |
| B2 | `outputs/b2_figures/07_q4_vs_target_histograma.png` | png |
| B2 | `outputs/b2_figures/08_q5_vs_target_boxplot.png` | png |
| B2 | `outputs/b2_figures/09_q5_vs_target_histograma.png` | png |
| B2 | `outputs/b2_figures/10_q2_vs_target.png` | png |
| B2 | `outputs/b2_figures/11_q6_vs_target.png` | png |
| B2 | `outputs/b2_figures/12_q10_vs_target.png` | png |
| B2 | `outputs/b2_figures/13_q50_vs_target.png` | png |
| B2 | `outputs/b2_figures/14_q51_vs_target.png` | png |
| B3 | `b3_branching_candidates.csv` | csv |
| B3 | `b3_demographic_identification.csv` | csv |
| B3 | `b3_missing_cooccurrence.csv` | csv |
| B3 | `b3_missing_diagnosis.csv` | csv |
| B3 | `b3_q4q5_evidence_summary.csv` | csv |
| B3 | `b3_q4q5_missing_by_age.csv` | csv |
| B3 | `b3_q4q5_missing_by_grade.csv` | csv |
| B3 | `b3_q4q5_missing_by_psu.csv` | csv |
| B3 | `b3_q4q5_missing_by_sex.csv` | csv |
| B3 | `b3_q4q5_missing_by_stratum.csv` | csv |
| B3 | `b3_q4q5_missing_by_target.csv` | csv |
| B3 | `outputs/b3_figures/01_missing_general_top.png` | png |
| B3 | `outputs/b3_figures/02_missing_priority_heatmap.png` | png |
| B3 | `outputs/b3_figures/03_q4q5_missing_by_target.png` | png |
| B3 | `outputs/b3_figures/04_q4q5_missing_by_psu.png` | png |
| B3 | `outputs/b3_figures/05_q4q5_missing_by_stratum.png` | png |
| B3 | `outputs/b3_figures/06_q4q5_missing_by_sexo.png` | png |
| B3 | `outputs/b3_figures/07_q4q5_missing_by_edad.png` | png |
| B3 | `outputs/b3_figures/08_q4q5_missing_by_grado.png` | png |
| B4 | `b4_coherence_checks.csv` | csv |
| B4 | `b4_domain_review_flags.csv` | csv |
| B4 | `b4_outlier_case_review.csv` | csv |
| B4 | `b4_outlier_flags_by_target.csv` | csv |
| B4 | `b4_outlier_method_overlap.csv` | csv |
| B4 | `b4_outlier_summary.csv` | csv |
| B4 | `b4_outliers_iqr.csv` | csv |
| B4 | `b4_outliers_z3.csv` | csv |
| B4 | `outputs/b4_figures/01_q4_hist_iqr.png` | png |
| B4 | `outputs/b4_figures/02_q4_boxplot.png` | png |
| B4 | `outputs/b4_figures/03_q5_hist_iqr.png` | png |
| B4 | `outputs/b4_figures/04_q5_boxplot.png` | png |
| B4 | `outputs/b4_figures/05_imc_diag_hist_iqr.png` | png |
| B4 | `outputs/b4_figures/06_imc_diag_boxplot.png` | png |
| B4 | `outputs/b4_figures/07_q4_vs_q5_domain_flags.png` | png |
| B4 | `outputs/b4_figures/08_imc_diag_by_target_boxplot.png` | png |
| B4 | `outputs/b4_figures/09_outlier_method_counts.png` | png |
| B5 | `b5_feature_inventory.csv` | csv |
| B5 | `b5_missing_partition_summary.csv` | csv |
| B5 | `b5_psu_partition_profile.csv` | csv |
| B5 | `b5_split_manifest.json` | json |
| B5 | `b5_split_summary.csv` | csv |
| B5 | `b5_target_distribution.csv` | csv |
| B5 | `b5_test_records.csv` | csv |
| B5 | `b5_train_records.csv` | csv |
| B6 | `b6_categorical_missing_fill_summary.csv` | csv |
| B6 | `b6_continuous_imputation_params.csv` | csv |
| B6 | `b6_feature_inventory.csv` | csv |
| B6 | `b6_imputation_manifest.json` | json |
| B6 | `b6_indicator_summary.csv` | csv |
| B6 | `b6_missing_before_after.csv` | csv |
| B6 | `b6_X_test.pkl` | pickle |
| B6 | `b6_X_train.pkl` | pickle |
| B6 | `b6_y_test.pkl` | pickle |
| B6 | `b6_y_train.pkl` | pickle |
| B7 | `b7_feature_inventory.csv` | csv |
| B7 | `b7_scaler_params.csv` | csv |
| B7 | `b7_scaling_manifest.json` | json |
| B7 | `b7_scaling_summary.csv` | csv |
| B7 | `b7_X_test.pkl` | pickle |
| B7 | `b7_X_train.pkl` | pickle |
| B7 | `b7_y_test.pkl` | pickle |
| B7 | `b7_y_train.pkl` | pickle |
| B8 | `b8_encoding_manifest.json` | json |
| B8 | `b8_encoding_plan.csv` | csv |
| B8 | `b8_encoding_summary.csv` | csv |
| B8 | `b8_feature_inventory.csv` | csv |
| B8 | `b8_onehot_vocabularies.csv` | csv |
| B8 | `b8_ordinal_mappings.csv` | csv |
| B8 | `b8_patch_ordinal_review.csv` | csv |
| B8 | `b8_unknown_categorical_proposals.csv` | csv |
| B8 | `b8_unseen_test_categories.csv` | csv |
| B8 | `b8_X_test.pkl` | pickle |
| B8 | `b8_X_train.pkl` | pickle |
| B8 | `b8_y_test.pkl` | pickle |
| B8 | `b8_y_train.pkl` | pickle |
| B9 | `b9_feature_engineering_manifest.json` | json |
| B9 | `b9_feature_engineering_summary.csv` | csv |
| B9 | `b9_feature_inventory.csv` | csv |
| B9 | `b9_imc_by_missing_indicator.csv` | csv |
| B9 | `b9_imc_raw_summary.csv` | csv |
| B9 | `b9_imc_scaler_params.csv` | csv |
| B9 | `b9_X_test.pkl` | pickle |
| B9 | `b9_X_train.pkl` | pickle |
| B9 | `b9_y_test.pkl` | pickle |
| B9 | `b9_y_train.pkl` | pickle |
| B10 | `b10_class_distribution.csv` | csv |
| B10 | `b10_feature_inventory.csv` | csv |
| B10 | `b10_sample_origin.csv` | csv |
| B10 | `b10_smote_categorical_interpolation_diagnostic.csv` | csv |
| B10 | `b10_smote_manifest.json` | json |
| B10 | `b10_smote_summary.csv` | csv |
| B10 | `b10_X_test.pkl` | pickle |
| B10 | `b10_X_train_smote.pkl` | pickle |
| B10 | `b10_y_test.pkl` | pickle |
| B10 | `b10_y_train_smote.pkl` | pickle |
| B11 | `b11_feature_inventory.csv` | csv |
| B11 | `b11_mutual_information_ranking.csv` | csv |
| B11 | `b11_selected_features.csv` | csv |
| B11 | `b11_selection_manifest.json` | json |
| B11 | `b11_selection_summary.csv` | csv |
| B11 | `b11_source_feature_mi_summary.csv` | csv |
| B11 | `b11_X_test.pkl` | pickle |
| B11 | `b11_X_train.pkl` | pickle |
| B11 | `b11_y_test.pkl` | pickle |
| B11 | `b11_y_train.pkl` | pickle |
| B12 | `b12_cumulative_variance.png` | png |
| B12 | `b12_pca_explained_variance.csv` | csv |
| B12 | `b12_pca_loadings.csv` | csv |
| B12 | `b12_pca_manifest.json` | json |
| B12 | `b12_pca_scaler_params.csv` | csv |
| B12 | `b12_pca_summary.csv` | csv |
| B12 | `b12_pca_thresholds.csv` | csv |
| B12 | `b12_pca_top_contributors.csv` | csv |
| B12 | `b12_pca_tradeoffs.csv` | csv |
| B12 | `b12_scree_plot.png` | png |
| B12 | `b12_X_test_pca.pkl` | pickle |
| B12 | `b12_X_train_pca.pkl` | pickle |
| B12 | `b12_y_test.pkl` | pickle |
| B12 | `b12_y_train.pkl` | pickle |
| B13 | `b13_key_numbers.csv` | csv |
| B13 | `b13_narrative_master.md` | md |
| B13 | `b13_pipeline_final.md` | md |
| B13 | `b13_test_boundary_ledger.csv` | csv |
| D1 | `d1_mi_top10_bar.png` | png |
| D1 | `d1_mi_top10_bar.svg` | svg |
| D1 | `d1_mi_top10_labels.csv` | csv |
| D1 | `d1_mi_top10_manifest.json` | json |

No hay artefactos persistidos de B0 ni B1: ambos scripts imprimen su auditoría y trabajan en memoria.

## Cadena de dependencias

Orden reproducible inferido de las lecturas reales del código:

```text
EMSE_DatosAbiertos.csv
├── B0 b0_setup_emse.py
│   └── funciones de carga/inventario importadas por B1–B4/B5/B6
├── B1 b1_target_emse.py
│   └── funciones de target importadas por B2–B6
└── B2 b2_eda_dictionary_emse.py
    ├── b2_data_dictionary.csv ───────────────► B3 y B5; apoyo de B8
    └── b2_missing_summary.csv ───────────────► B3

B3 b3_missing_diagnosis_emse.py               # diagnóstico; no produce matriz downstream
B4 b4_outlier_diagnosis_emse.py                # diagnóstico; no modifica matriz downstream

B5 b5_split_emse.py
├── b5_train_records.csv
├── b5_test_records.csv
├── b5_feature_inventory.csv
└── b5_split_manifest.json
    └── FRONTERA TRAIN/TEST: se crea y congela aquí
        ▼
B6 b6_imputation_emse.py
└── b6_X/y_{train,test}.pkl + manifest/inventario
    ▼
B7 b7_scaling_emse.py
└── b7_X/y_{train,test}.pkl + manifest/inventario
    ▼
B8 b8_encoding_emse.py
└── B8 patch b8_patch_ordinal_semantic_review_emse.py
    └── b8_X/y_{train,test}.pkl canónicos + mappings/vocabularios
        ▼
B9 b9_feature_engineering_emse.py
├── matriz predictiva desde B8
├── q4/q5 físicos desde B6
└── b9_X/y_{train,test}.pkl canónicos (train original)
    ├──► B11 b11_mutual_information_selection_emse.py
    │    └── b11_X/y_{train,test}.pkl + ranking Top 30
    │         ▼
    │       B12 b12_pca_emse.py
    │         └── b12_X/y_* + varianza/loadings/gráficos
    │              ▼
    │            B13 artefactos documentales (sin script visible)
    │
    └──► B10 b10_smote_emse.py
         └── RAMA LATERAL: SMOTE_DEMONSTRATION_ONLY
             Sus matrices NO son usadas por B11, B12 ni B13.
```

La frontera de evaluación ocurre en B5. B6–B9 transforman train/test preservando esa membresía y aprendiendo parámetros sólo desde train según sus manifests y checks. B11 aprende el ranking MI en el train original y sólo aplica las columnas elegidas a test. B12 ajusta scaler/PCA en train y transforma test. No existe entrenamiento de un modelo en B0–B13.

## Huecos y observaciones

- **B13 no tiene script:** existen cuatro artefactos documentales `b13_*`, pero ningún `b13_*.py`. Por eso no puede verificarse desde código qué comando exacto los generó ni reproducirse B13 automáticamente con lo que hay en disco.
- **B10 conserva un contrato anterior al patch B8→B9:** `b10_smote_emse.py` y `b10_smote_manifest.json` esperan/registran 446 features, mientras los manifests canónicos actuales registran B8 con 403, B9 con 404 y B11 con 404 entradas. Los artefactos B10 corresponden a la rama demostrativa previa y no son compatibles como salida derivada del B9 canónico actual sin una revisión; esto no afecta el downstream porque B11/B12 leen B9 original, no B10.
- **Script de entrega ausente:** `d1_mi_top10_*` existe como conjunto de artefactos, pero no hay script generador con ese prefijo. Su procedencia sólo puede rastrearse por su manifest.
- **Archivos mencionados por B0–B12:** durante la inspección estática no se encontró ningún input u output explícito de esos scripts ausente en disco. Esto no prueba que una reejecución sea exitosa; sólo confirma presencia nominal.
- **B0/B1 sin artefactos persistidos:** sus resultados se emiten por consola. La reproducibilidad de sus outputs depende de volver a ejecutar los scripts, acción deliberadamente no realizada en este relevamiento.
- **B8 tiene dos generadores para el mismo conjunto canónico:** el script base y el patch escriben los mismos nombres `b8_*`; el patch reemplaza esos artefactos después de validar su staging. El contenido actual corresponde al contrato parcheado (manifest: 403 features de salida).
- **Estado de Git:** el repositorio raíz no tiene archivos seguidos; los archivos de entrega aparecen como untracked y `_no_entrega/` queda ignorado. Esto impide usar historial Git para atribuir qué versión creó cada artefacto.
- **Material ajeno o auxiliar:** `CEIA_Analisis_de_datos/`, `heart-failure-eda/`, `wholesale_customers.csv`, `wholesale_customers_eda.ipynb`, `inspect_wholesale_customers.py`, el bytecode, la consigna PDF y el manual DOCX se preservan en `_no_entrega/`. Ninguno es leído por los scripts EMSE B0–B13 inspeccionados. El cuestionario EMSE permanece en la raíz como referencia específica de la fuente, no como input de script.
- **Alcance semántico:** los scripts usan los pares código/texto del CSV y no una equivalencia automática con la numeración del PDF. En este glosario MI se describe como asociación estadística; no se atribuyen causalidad, influencia ni performance predictiva.
