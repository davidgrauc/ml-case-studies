#!/usr/bin/env python3
"""vistazo.py · sesión 1: tasa global, nulos y tasa de impago por variable con su intervalo.

    .venv/bin/python home-credit/src/vistazo.py
"""
from pathlib import Path

import pandas as pd

from metricas import tasa_por_grupo

DATOS = Path(__file__).resolve().parents[2] / "datos" / "home-credit" / "application_train.csv"
CATEGORICAS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
               "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]


def main():
    df = pd.read_csv(DATOS)
    print(f"{len(df):,} solicitudes · tasa de impago {df.TARGET.mean():.1%}\n")

    nulos = df.isna().mean().sort_values(ascending=False)
    print(f"Columnas con más del 50 % de nulos: {(nulos > 0.5).sum()} de {df.shape[1]}\n")

    for col in CATEGORICAS:
        t = tasa_por_grupo(df.TARGET, df[col])
        print(f"── {col}")
        print(t.assign(tasa=t.tasa.map("{:.1%}".format), ic_bajo=t.ic_bajo.map("{:.1%}".format),
                       ic_alto=t.ic_alto.map("{:.1%}".format)).to_string(), "\n")

    # Una numérica en tramos: la edad (DAYS_BIRTH es negativa, en días)
    edad = pd.cut(-df.DAYS_BIRTH / 365.25, [18, 25, 35, 45, 55, 70])
    print("── Edad")
    print(tasa_por_grupo(df.TARGET, edad.astype(str))[["casos", "tasa", "ic_bajo", "ic_alto"]].round(3))


if __name__ == "__main__":
    main()
