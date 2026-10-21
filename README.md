# ml-case-studies

Casos de estudio de *machine learning* con datos públicos. Cada caso parte de una pregunta de
negocio y la recorre entera: entender los datos, medir con su incertidumbre, modelar, validar
sin engañarse y explicar qué haría el negocio con el resultado.

## Casos

| Caso | Datos | Pregunta | Estado |
|---|---|---|---|
| [Riesgo de impago](home-credit/) | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle) | ¿Qué clientes tienen más probabilidad de no devolver un préstamo, y cómo se lee ese riesgo para actuar? | En curso |

## Principios

- **Una tasa sin su intervalo no dice nada.** Las tasas por grupo llevan su intervalo de
  confianza (Wilson), y los grupos pequeños se descartan.
- **El AUC no basta.** Cada modelo se lee también con el *lift* por decil (cuánto riesgo
  concentra el grupo sobre el que se actuaría) y con la calibración (si la probabilidad que da
  es creíble).
- **Un resultado demasiado bueno es una alarma.** Se buscan fugas de información: variables que
  no se conocerían en el momento de decidir.
- **El modelo más simple que funcione.** Un modelo complejo solo se queda si mejora de verdad al
  sencillo con datos que no ha visto.

## Cómo ejecutarlo

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Los datos no están en el repo: cada caso explica cómo descargarlos en `datos/`.

## Estructura

```
<caso>/
  README.md     pregunta, datos, sesiones y conclusiones
  src/          scripts reproducibles
  notebooks/    exploración (sin salidas)
datos/          datos descargados (fuera de git)
```
