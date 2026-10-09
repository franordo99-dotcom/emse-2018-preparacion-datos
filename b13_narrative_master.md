# EMSE 2018 — narrativa maestra del análisis

## 1. Problema

El trabajo plantea una **clasificación binaria supervisada**: anticipar si un estudiante cumple el proxy operativo de la recomendación OMS de actividad física, definido como realizar al menos 60 minutos diarios durante **5 o más días por semana**. El target `target_pa_oms5` se construyó exclusivamente desde la semántica observada de `q49`: 5, 6 o 7 días se codificaron como 1; 0 a 4 días como 0; los faltantes permanecieron sin target.

No se entrenó ningún modelo porque la consigna desarrollada hasta B13 se concentra en comprensión, preparación, reducción y síntesis. Por lo tanto, no se informa performance predictiva.

## 2. Universo

El CSV crudo contiene **56.981 estudiantes y 309 columnas**. El universo supervisado excluye únicamente las 1.430 filas sin target observado y conserva **55.551 estudiantes**: **16.558 positivos (29,8068%)** y **38.993 negativos (70,1932%)**, sin ponderar.

El split B5 congeló 44.440 casos para train y 11.111 para test. La frontera fue individual, aleatoria, estratificada por target y reproducible con `random_state=42`.

## 3. Preparación

### Faltantes

B3 confirmó co-ausencia exacta de `q4` y `q5`: 20.209 casos del universo supervisado carecen de ambas mediciones. Las tasas variaron materialmente según target, PSU, estrato, edad y grado; según el criterio metodológico adoptado, el diagnóstico de trabajo quedó como `MAR_HIPOTESIS_RESPALDADA`, con reserva `MNAR_POSIBLE_NO_CONFIRMABLE`. Esto no demuestra causalidad ni convierte los casos completos en una muestra aleatoria.

### Imputación

B6 aprendió exclusivamente en train las medianas de altura y peso (`q4=1,64 m`, `q5=58 kg`), las aplicó sin refit a train y test y agregó el indicador conjunto `q4q5_faltaba`. Los faltantes categóricos recibieron la categoría fija `sin_dato`; no se usó moda ni información de test.

### Outliers

B4 comparó IQR, ±3 desvíos y banderas descriptivas de dominio. Como esos criterios identifican candidatos pero no errores demostrados, la política fue `RETENER_Y_MARCAR_SIN_MODIFICAR`: no se eliminaron, recortaron, winsorizaron ni reemplazaron valores extremos.

### Frontera y transformaciones

El split ocurrió antes de todo aprendizaje de parámetros. B7 ajustó `StandardScaler` de `q4/q5` sólo en train. B8 aprendió vocabularios one-hot y mappings ordinales sólo en train y luego aplicó la revisión semántica que dejó 403 features. B9 calculó IMC mediante una regla fija usando `q4/q5` imputados pero no escalados de B6, ajustó su scaler sólo en train y produjo 404 features.

### SMOTE

B10 ejecutó SMOTE sólo sobre train como demostración técnica y generó 17.948 observaciones sintéticas. La interpolación produjo la limitación esperada sobre dimensiones one-hot y ordinales. Esta rama quedó archivada como `SMOTE_DEMONSTRATION_ONLY` y **no fue utilizada por B11 ni B12**.

## 4. Selección B11

B11 calculó Mutual Information exclusivamente sobre las 44.440 observaciones originales de train, con una máscara que distinguió 3 continuas de 401 dimensiones discretas. La regla de retención fue fijada antes de ver el ranking: **Top 30 columnas**, reduciendo 404 a 30 dimensiones, una reducción de 92,5743%.

Las primeras posiciones fueron `imc`, `q50`, `q61`, `q2__1` y `q2__2`. `q50` quedó rank 2 y `q51` rank 17; ambas permanecen como `SAME_DOMAIN_REVIEW_NOT_HARD_LEAKAGE`. MI mide **asociación individual con el target en train**, no causalidad, efecto independiente ni importancia causal. Varias dummies de una misma pregunta pueden aparecer simultáneamente porque la selección oficial opera sobre columnas codificadas individuales.

## 5. PCA B12

B12 reestandarizó las 30 columnas sólo para la geometría PCA, con `StandardScaler` ajustado en train. Después ajustó `PCA(n_components=None, svd_solver="full")` una sola vez sobre train y sin recibir `y`. La regla predefinida retuvo el mínimo número de componentes que alcanza 90% de la varianza acumulada de train: **13 componentes explican 90,7732%**. Como referencias, `k80=10` y `k95=16`.

La reducción 30→13 equivale a 56,6667%. Sus ventajas son sintetizar dimensiones, producir ejes linealmente no correlacionados y concentrar varianza; sus desventajas son perder interpretación directa, asumir linealidad y no garantizar relevancia predictiva. PCA se aplicó a un espacio mixto con continuas, ordinales y one-hot: es una aproximación didáctica; MCA sería más natural para datos predominantemente categóricos, pero está fuera del alcance. Además, la entrada ya estaba informada por el target debido a B11, aunque PCA no usó `y` directamente.

`PC30` presentó varianza explicada aproximadamente cero. Esto es compatible con dependencia lineal del espacio codificado, no con una feature individual constante ni con un error de PCA. Estado: `NON_BLOCKING_OBSERVATION`.

## 6. Resultado metodológico

El flujo pasó de un dataset amplio, mixto y con faltantes a una representación trazable y reducida, respetando la frontera train/test y separando con claridad descripción, diagnóstico, asociación y reducción dimensional. La línea principal conserva el train original: B9→B11→B12; B10 permanece como una rama demostrativa sin downstream.

El resultado no permite afirmar performance predictiva ni causalidad. Sí permite defender que las decisiones fueron reproducibles, que test permaneció fuera de todos los fits posteriores a B5 y que la síntesis final conserva 13 componentes que explican 90,7732% de la varianza del espacio Top 30 en train.
