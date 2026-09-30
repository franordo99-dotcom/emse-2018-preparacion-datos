# TP Integrador — Análisis de Datos (CEIA/FIUBA) · EMSE 2018

Este trabajo prepara datos de la Encuesta Mundial de Salud Escolar (EMSE) 2018 para una clasificación binaria supervisada: cumplimiento de la recomendación de actividad física de al menos cinco días semanales con ≥60 minutos diarios, derivado de `q49`. Recorre carga, diagnóstico, partición, transformaciones, selección por Mutual Information y PCA. **No se entrena ningún modelo:** la consigna no lo requiere.

## Variables clave

- q49 = días de actividad física de ≥60 min en la última semana (fuente del target).
- q50 = transporte activo: días que fue caminando o en bici a la escuela (feature, no target).
- q61 = días de clase de educación física en la escuela (feature).

El target `target_pa_oms5` se construye desde el código **y la etiqueta de respuesta** de `q49`, no desde su número de código aislado. `q49` y sus proxies directos quedan fuera de las features. `q50` y `q51` son del mismo dominio, pero no forman parte de la definición del target.

## Pipeline

| Bloque | Etapa | Qué hace |
|---|---|---|
| B0 | Setup y carga | Carga el CSV y audita su estructura inicial. |
| B1 | Target y leakage | Construye el target desde `q49` y registra fuente/proxies excluidos. |
| B2 | EDA y diccionario | Describe variables, faltantes, distribuciones y metadata muestral. |
| B3 | Faltantes | Diagnostica patrones de ausencia sin imputar. |
| B4 | Outliers | Marca extremos de altura, peso e IMC diagnóstico sin modificarlos. |
| B5 | Split | Crea y congela la frontera train/test individual estratificada. |
| B6 | Imputación | Aprende medianas de `q4`/`q5` en train; agrega indicador y `sin_dato`. |
| B7 | Scaling | Estandariza sólo `q4`/`q5` con parámetros de train. |
| B8 | Encoding y patch | Codifica categóricas con vocabularios de train y corrige ordinales según etiquetas. |
| B9 | IMC | Calcula IMC con `q4`/`q5` físicos de B6 y lo escala desde train. |
| B10 | SMOTE | Demostración lateral de remuestreo sólo en train; no alimenta B11/B12. |
| B11 | Selección MI | Ordena asociación individual con el target en train y retiene Top 30. |
| B12 | PCA | Ajusta scaler/PCA sobre Top 30 en train y conserva 13 componentes. |
| B13 | Consolidación | Reúne narrativa, cifras, trazabilidad de la frontera y el pipeline final; no tiene script. |

Para entradas, salidas y checks de cada bloque, consultar [GLOSARIO_REPO.md](GLOSARIO_REPO.md). Los gráficos descriptivos están en `outputs/`; las matrices, resúmenes y manifests están en la raíz.

## Cómo reproducir la rama principal

Colocar `EMSE_DatosAbiertos.csv` en la raíz. Se requieren Python y `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` y `matplotlib`; el repositorio no fija versiones. Una instalación posible es:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

Desde la raíz, en una copia de trabajo si se desea conservar intactos los artefactos publicados, ejecutar en este orden. Los scripts B2–B12 escriben archivos con los nombres ya presentes.

```bash
python b0_setup_emse.py
python b1_target_emse.py
python b2_eda_dictionary_emse.py
python b3_missing_diagnosis_emse.py
python b4_outlier_diagnosis_emse.py
python b5_split_emse.py
python b6_imputation_emse.py
python b7_scaling_emse.py
python b8_encoding_emse.py
python b8_patch_ordinal_semantic_review_emse.py
python b9_feature_engineering_emse.py
python b11_mutual_information_selection_emse.py
python b12_pca_emse.py
```

## Datos

El dataset es **EMSE 2018 (datos abiertos)**. Descargarlo desde la fuente oficial y colocarlo en la raíz del repositorio con el nombre exacto `EMSE_DatosAbiertos.csv`. Por su tamaño y por la política de publicación de esta entrega, el CSV no se versiona en GitHub: `.gitignore` lo excluye.

Los archivos `.pkl` tampoco se versionan. Se regeneran ejecutando el pipeline en orden (`B0 → B12`) después de contar con el CSV en la raíz; los pickles aparecen a partir de B6. B0–B4 producen diagnósticos y artefactos tabulares/gráficos; B5 crea la frontera train/test que consumen los bloques siguientes.

**B5 es la frontera train/test.** El patch B8 reemplaza la salida canónica del B8 base; B9 lee la matriz B8 parcheada y las unidades físicas imputadas de B6. B11/B12 leen el **train original de B9**, no el train remuestreado. B13 consiste en entregables documentales existentes y no tiene comando de ejecución en este repositorio.

**B10 no se ejecuta en la rama principal.** Los artefactos B10 existentes corresponden a una versión anterior al patch semántico B8→B9: registran 446 features, mientras el B9 canónico actual registra 404. B10 es una demostración de SMOTE, no se usa aguas abajo y esta discrepancia no cambia B11/B12. Ejecutar el script B10 actual contra el B9 parcheado exigiría revisar ese contrato; aquí no se hizo.

## Lectura responsable de los resultados

- Todas las cifras son crudas, sin ponderar por el diseño muestral de la encuesta.
- SMOTE (B10) es una rama demostrativa, no usada aguas abajo.
- Las asociaciones individuales medidas son débiles; ninguna variable predice el target por sí sola.

Mutual Information mide **asociación, no causalidad**. Las 13 componentes PCA resumen varianza del espacio codificado seleccionado; no son factores causales ni prueban rendimiento predictivo. No se informan métricas de performance porque no se entrenó un modelo.

## Artefactos sin script generador visible

- `d1_mi_top10_*`: gráfico, labels y manifest del Top 10 MI para la entrega; no hay script generador en disco.
- `b13_narrative_master.md`, `b13_pipeline_final.md`, `b13_key_numbers.csv` y `b13_test_boundary_ledger.csv`: consolidación documental; B13 no es un script ejecutable.

## Publicación en GitHub

La carpeta `_no_entrega/` contiene material ajeno preservado localmente y está excluida por `.gitignore`. El CSV, el cuestionario PDF y todos los pickles también quedan fuera de la publicación. Antes de publicar, corresponde verificar el permiso de redistribución del dataset y del cuestionario; ese permiso no se determina a partir del código del TP.
