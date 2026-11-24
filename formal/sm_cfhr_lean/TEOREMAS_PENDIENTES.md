# Teoremas Pendientes y Alcance Explícito de la Formalización Lean 4

## 1. Estado de teoremas matemáticos formalizados
- Todos los teoremas centrales de las Secciones 4--5 y Apéndice A del manuscrito (clipping, pérdida y derivada de Huber, convexidad del conjunto factible y del objetivo, gradiente analítico, Lipschitz, coercividad, existencia, unicidad de $\mathbf{w}^*$, unicidad condicional de $b^*$ con contraejemplo, condiciones KKT, cotas por observación al gradiente y monotonía parcial) se encuentran formalizados en `SMCFHR/*.lean` **sin `sorry`, `admit` ni axiomas personalizados**.

## 2. Propiedades fuera del alcance de Lean 4 (verificación separada)
Para mantener rigor científico absoluto, se declara explícitamente que Lean 4 **no** ha verificado ni puede verificar dentro de este desarrollo algebraico sobre $\mathbb{R}$:
1. La implementación numérica en Python / NumPy / SciPy ni la aritmética de punto flotante IEEE-754.
2. La convergencia empírica en tiempo finito del algoritmo `L-BFGS-B` de SciPy.
3. La calidad predictiva ($\mathrm{MAE}_{\mathrm{macro}}$, $\mathrm{Skill}$) sobre los 28 escenarios solares.
4. La calibración empírica de las pruebas estadísticas (`DM-HLN-HAC`, `MCS`, `SPA`, `GW`, `CW`).
5. La validez del protocolo de validación causal anidada ni la ausencia de fugas en los scripts de datos (verificadas mediante `pytest` en `tests/`).
