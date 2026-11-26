# Paquete `sm_cfhr` — Structured Monotone Capacity-Factor Huber Regressor

Paquete Python reutilizable, compatible con `scikit-learn` (`BaseEstimator`, `RegressorMixin`, `clone`, `Pipeline`, `GridSearchCV`), cuyo núcleo matemático de aprendizaje está implementado en NumPy/SciPy puro sin delegar el estimador en bibliotecas externas.

## 1. Instalación editable

```bash
pip install -e .
```

## 2. Uso rápido (API `scikit-learn`)

```python
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sm_cfhr import SMCFHRRegressor

model = SMCFHRRegressor(
    delta=1.35,
    alpha_lin=0.01,
    alpha_meteo=0.10,
    use_mono=True,
    monotone_pos_idx=(4, 7),
    meteo_idx=tuple(range(7, 17)),
    random_state=2026,
)

# Entrenamiento y predicción en factor de capacidad adimensional [0, 1]
model.fit(X_train, cf_train)
cf_pred = model.predict(X_test)

# Reconstrucción física en energía (kWh) usando capacidad instalada (MW)
kwh_pred = model.predict_energy(X_test, cap_mw=cap_mw_test, horizon_direct=1)

# Integración con Pipeline, clone y serialización
pipe = Pipeline([("scaler", StandardScaler()), ("reg", clone(model))])
pipe.fit(X_train, cf_train)
joblib.dump(pipe, "sm_cfhr_pipe.joblib")
loaded_pipe = joblib.load("sm_cfhr_pipe.joblib")
```

> **Nota sobre unidades:** `SMCFHRRegressor.fit` y `predict` operan en factor de capacidad adimensional $c \in [0, 1]$ para garantizar equivariancia entre plantas de distinta escala. Para obtener energía física en `kWh`, utilice `predict_energy(X, cap_mw, horizon_direct=1)`.

## 3. Correspondencia entre Formulación Matemática y Código

| Símbolo / Ecuación | Significado | Implementación en código |
|---|---|---|
| $c_i = \operatorname{clip}(y_i / (B_i H), 0, 1)$ | Factor de capacidad acotado ($B_i = 24{,}000 \cdot \mathrm{cap}_i$) | `ashr_oof_pipeline.oof_runner.extract_cf_features` |
| $m_c, \hat{\sigma}_c = \max(1.4826\,\mathrm{MAD}(c), 10^{-6})$ | Normalización robusta de la respuesta | `src.modelo_aditivo_huber._robust_target_stats` |
| $\rho_\delta(r)$ y $\nabla\rho_\delta(r)$ | Pérdida simétrica de Huber y gradiente analítico | `src.modelo_aditivo_huber._huber_val_and_grad` |
| $\lambda_{\text{lin}}, \lambda_{\text{meteo}}$ | Regularización $L_2$ diferenciada por familia | `src.adaptive_structured_huber._fit_single` |
| $\theta_j \ge 0,\; j \in \mathcal{M}_{\text{pos}}$ | Restricciones físicas monótonas (L-BFGS-B bounds) | `scipy.optimize.minimize(..., method="L-BFGS-B", bounds=bounds)` |

## 4. Compatibilidad Retroactiva

- `src/adaptive_structured_huber.py` conserva exactamente el mismo algoritmo y firma de `__init__`; todos los scripts en `scripts/evaluacion/` y `src/ashr_oof_pipeline/` siguen funcionando sin cambios.
- `sm_cfhr` actúa como fachada modular limpia sobre `src/adaptive_structured_huber.py` y `src/ashr_oof_pipeline/`.

## 5. Comprobaciones Pendientes para Fase de Ejecución

- Ejecutar `pip install -e .` en el entorno `solarpv-eda` y verificar `from sm_cfhr import SMCFHRRegressor` sin `sys.path`.
- Verificar `pytest` completo tras la incorporación de `n_features_in_`, `feature_names_in_` y `NotFittedError`.
- Ejecutar `--ejecutar-oof-definitivo` y re-ejecutar el notebook científico en Estado B.
