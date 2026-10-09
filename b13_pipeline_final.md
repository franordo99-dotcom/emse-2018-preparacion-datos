# Pipeline final consolidado

```text
EMSE_DatosAbiertos.csv
RAW: 56.981 × 309
        ↓
TARGET desde semántica q49
55.551 casos observados · 29,8068% positivos
        ↓
EDA / MISSING / OUTLIERS
B2 descripción · B3 hipótesis MAR respaldada · B4 retener y marcar
        ↓
B5 SPLIT INDIVIDUAL ESTRATIFICADO
TRAIN ORIGINAL 44.440          TEST CONGELADO 11.111
        ↓                              │
B6 IMPUTACIÓN TRAIN-ONLY               │ transform únicamente
medianas q4/q5 + indicador             │
sin_dato fijo                          │
        ↓                              │
B7 SCALING TRAIN-ONLY                  │
q4/q5                                  │
        ↓                              │
B8 ENCODING TRAIN-ONLY                 │
ordinal semántico + one-hot            │
403 features                           │
        ↓                              │
B9 IMC FÍSICO + SCALER TRAIN-ONLY      │
404 features                           │
        ├─────────────────────────────────────────────┐
        │                                             │
        │ LÍNEA PRINCIPAL                             │ RAMA LATERAL
        ↓                                             └── B10 SMOTE
B11 Información Mutua Top 30                            DEMONSTRATION ONLY
404 → 30                                                 17.948 sintéticos
        ↓                                                 NO DOWNSTREAM
B12 PCA TRAIN-ONLY
30 → 13 PCs · 90,7732% varianza train
        ↓
B13 CONSOLIDACIÓN
sin fit · sin transformación · sin modelo
```

## Frontera train/test

- B10: solo demostración.
- `DOWNSTREAM_TRAIN_SOURCE = B9_ORIGINAL_TRAIN`
- `DOWNSTREAM_TRAIN_N = 44440`
- `DOWNSTREAM_SMOTE_SOURCE_ALLOWED = False`
- `TEST_FROZEN_AFTER_B5 = True`
- `TEST_USED_FOR_FIT_ANYWHERE_AFTER_B5 = False`
