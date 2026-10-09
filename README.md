# TP Integrador — Análisis de Datos (CEIA/FIUBA) · EMSE 2018

**Autor:** Francisco Ordóñez · Especialización en Inteligencia Artificial (CEIA), FIUBA · 2026

Este trabajo prepara datos de la Encuesta Mundial de Salud Escolar (EMSE) 2018 para una clasificación binaria supervisada: cumplimiento de la recomendación de actividad física de al menos cinco días semanales con ≥60 minutos diarios, derivado de `q49`. Recorre carga, diagnóstico, partición, transformaciones, selección por Información Mutua y PCA. **No se entrena ningún modelo:** la consigna no lo requiere.

## Por dónde empezar

1. [Este README](README.md): problema, pipeline y cómo reproducir.
2. [b13_narrative_master.md](b13_narrative_master.md): el análisis completo y sus conclusiones.
3. [outputs/](outputs/): gráficos del EDA (b2), de faltantes (b3) y de outliers (b4).
4. [b11_mutual_information_ranking.csv](b11_mutual_information_ranking.csv): ranking de asociación con el target.
5. [b12_pca_explained_variance.csv](b12_pca_explained_variance.csv) y [b12_scree_plot.png](b12_scree_plot.png): resultado del PCA.
6. [GLOSARIO_REPO.md](GLOSARIO_REPO.md): entradas, salidas y controles de cada bloque.

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
| B8 | Encoding | Codifica categóricas con vocabularios de train (ordinal u one-hot según las etiquetas) |
| B9 | IMC | Calcula IMC con `q4`/`q5` físicos de B6 y lo escala desde train. |
| B10 | SMOTE | Demostración lateral de remuestreo sólo en train; no alimenta B11/B12. |
| B11 | Selección MI | Ordena asociación individual con el target en train y retiene Top 30. |
| B12 | PCA | Ajusta scaler/PCA sobre Top 30 en train y conserva 13 componentes. |
| B13 | Consolidación | Reúne narrativa, cifras, trazabilidad de la frontera y el pipeline final. |

Para entradas, salidas y checks de cada bloque, consultar [GLOSARIO_REPO.md](GLOSARIO_REPO.md). Los gráficos descriptivos están en `outputs/`; las matrices, resúmenes y manifests están en la raíz.

## Cómo reproducir la rama principal

Colocar `EMSE_DatosAbiertos.csv` en la raíz. Se requieren Python y `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` y `matplotlib`, con las versiones registradas en [requirements.txt](requirements.txt). Para instalarlas:

```bash
python -m pip install -r requirements.txt
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

El dataset es **EMSE 2018 (datos abiertos)**. [Descargar el dataset desde la fuente oficial](https://datos.salud.gob.ar/dataset/base-de-datos-de-la-3-encuesta-mundial-de-salud-escolar-emse-con-resultados-nacionales-argentina/archivo/509979de-3a24-4f86-8859-15e177eccb20) y colocarlo en la raíz del repositorio con el nombre exacto `EMSE_DatosAbiertos.csv`. Por su tamaño, el CSV no se versiona en GitHub: `.gitignore` lo excluye.

Los archivos `.pkl` tampoco se versionan. Se regeneran ejecutando el pipeline en orden (`B0 → B12`) después de contar con el CSV en la raíz; los pickles aparecen a partir de B6. B0–B4 producen diagnósticos y artefactos tabulares/gráficos; B5 crea la frontera train/test que consumen los bloques siguientes.

**B5 separa train y test.** B8 incluye una revisión semántica de las variables ordinales según sus etiquetas. B9 usa esa codificación y los valores físicos de altura y peso imputados en B6. B11 y B12 usan el **train original de B9**, no el remuestreado. B13 reúne la documentación de resultados.

**B10 es una demostración de SMOTE y sus resultados no se usan en los pasos siguientes.** Sus archivos corresponden a una versión anterior de la codificación, con 446 columnas en lugar de las 404 de B9. Esto no afecta B11 ni B12, que trabajan con el train original.

## Resultados principales

- 55.551 estudiantes con target; cumple la recomendación el 29,8%.
- Un 36,4% no informó peso ni altura, siempre juntos; falta el 39,0% entre quienes no cumplen y el 30,2% entre quienes cumplen. Se trabajó bajo hipótesis MAR: mediana de train más un indicador de faltante.
- El encoding llevó el espacio de 149 a 404 columnas; la selección retuvo 30 y el PCA 13 componentes (90,77% de la varianza de train).
- Las asociaciones son débiles: IMC 0,0098, transporte activo a la escuela (q50) 0,0095 y clases de educación física (q61) 0,0087, sobre un máximo posible de 0,61 nats.

## Lectura responsable de los resultados

- Todas las cifras son crudas, sin ponderar por el diseño muestral de la encuesta.
- SMOTE (B10) es una rama demostrativa, no usada aguas abajo.
- Las asociaciones individuales medidas son débiles; ninguna variable predice el target por sí sola.

Información Mutua mide **asociación, no causalidad**. Las 13 componentes PCA resumen varianza del espacio codificado seleccionado; no son factores causales ni prueban rendimiento predictivo. No se informan métricas de performance porque no se entrenó un modelo.

## Material de entrega

- `d1_mi_top10_*`: gráfico del Top 10 de Información Mutua usado en la infografía, con etiquetas y manifest.
- `b13_*`: narrativa, cifras clave, pipeline final y registro de la frontera train/test.
