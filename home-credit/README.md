# Home Credit Default Risk

Predecir si un cliente tendrá problemas para devolver un préstamo (`TARGET = 1`, ~8 % de los
casos), con datos de solicitudes reales: 307.511 en el conjunto de entrenamiento. Es el problema
típico de riesgo de impago: ordenar a los clientes por riesgo y decidir sobre quién actuar.

## Datos

Hay que tener cuenta de Kaggle, **aceptar las reglas de la competición** en su página y dejar
el token de la API en `~/.kaggle/kaggle.json` (Kaggle › Settings › API › Create New Token).
Después:

```bash
.venv/bin/kaggle competitions download -c home-credit-default-risk -p datos/home-credit
```

```bash
unzip -o datos/home-credit/home-credit-default-risk.zip application_train.csv -d datos/home-credit
```

Con `application_train.csv` basta para las cuatro sesiones. Las demás tablas (`bureau`,
`installments_payments`…) son para ampliar.

## Sesiones

Cuatro sesiones de 1–2 h, cada una sobre lo que deja la anterior.

| # | Sesión | Script | Qué hay que llevarse |
|---|---|---|---|
| 1 | **Primer vistazo**: tasa global, nulos, tasa de impago por variable con su intervalo de confianza (Wilson) | `src/vistazo.py` | Leer una tasa con su incertidumbre; no fiarse de grupos pequeños |
| 2 | **Modelo base**: regresión logística con un pipeline (imputar, codificar, escalar) | `src/base.py` | AUC, *lift* por decil y calibración: qué dice cada una y cuál importa para actuar |
| 3 | **Boosting**: HistGradientBoosting contra la logística | `src/base.py --modelo boosting` | Cuándo compensa un modelo más complejo y cuándo no |
| 4 | **Fugas de información**: meter a propósito una variable que no se conocería al conceder el préstamo y ver qué pasa | `src/base.py --fuga` | Reconocer una fuga: un AUC «demasiado bueno» es una alarma, no un éxito |

**Una limitación:** Home Credit no trae fecha de solicitud, así que aquí se valida con una
partición aleatoria estratificada. Con datos que tienen fecha, lo honesto es validar por
cohortes (entrenar con las antiguas, probar con las nuevas), porque el negocio cambia con el
tiempo y una partición aleatoria lo esconde.

## Conclusiones

Se rellenan al terminar cada sesión.
