"""metricas.py · medidas para leer un modelo de riesgo (o una tasa) sin engañarse."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def wilson(exitos, n, confianza=0.95):
    """Intervalo de Wilson para una proporción: se porta bien con n pequeño y tasas extremas."""
    exitos, n = np.asarray(exitos, float), np.asarray(n, float)
    z = norm.ppf(1 - (1 - confianza) / 2)
    p = exitos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - margen, centro + margen


def tasa_por_grupo(y, grupo, minimo=30):
    """Tasa de y por grupo con su intervalo, ordenada de mayor a menor. Grupos con < minimo casos fuera."""
    df = pd.DataFrame({"y": y, "g": grupo.fillna("(nulo)")})
    t = df.groupby("g").y.agg(casos="size", positivos="sum")
    t = t[t.casos >= minimo]
    t["tasa"] = t.positivos / t.casos
    t["ic_bajo"], t["ic_alto"] = wilson(t.positivos, t.casos)
    return t.sort_values("tasa", ascending=False)


def lift_por_decil(y, prob):
    """Deciles de riesgo (10 = más riesgo): casos, tasa real, *lift* sobre la media y % de positivos capturados."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(prob)})
    df["decil"] = pd.qcut(df.p.rank(method="first"), 10, labels=range(1, 11))
    t = df.groupby("decil", observed=True).agg(casos=("y", "size"), positivos=("y", "sum"),
                                               prob_media=("p", "mean"))
    t["tasa"] = t.positivos / t.casos
    t["lift"] = t.tasa / df.y.mean()
    t["capturado_acum"] = t.positivos[::-1].cumsum()[::-1] / df.y.sum()
    return t.sort_index(ascending=False)


def calibracion(y, prob, bins=10):
    """Probabilidad predicha frente a tasa real por tramo. Bien calibrado: las dos columnas se parecen."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(prob)})
    df["tramo"] = pd.qcut(df.p, bins, duplicates="drop")
    t = df.groupby("tramo", observed=True).agg(casos=("y", "size"), prevista=("p", "mean"),
                                               real=("y", "mean"))
    t["diferencia_pp"] = 100 * (t.real - t.prevista)
    return t
