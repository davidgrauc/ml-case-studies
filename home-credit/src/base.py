#!/usr/bin/env python3
"""base.py · sesiones 2–4: modelo de riesgo con AUC, *lift* por decil y calibración.

    .venv/bin/python home-credit/src/base.py                     # logística (sesión 2)
    .venv/bin/python home-credit/src/base.py --modelo boosting   # boosting (sesión 3)
    .venv/bin/python home-credit/src/base.py --fuga              # con una variable con fuga (sesión 4)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metricas import calibracion, lift_por_decil

DATOS = Path(__file__).resolve().parents[2] / "datos" / "home-credit" / "application_train.csv"
NUMERICAS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
             "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
             "REGION_RATING_CLIENT", "CNT_CHILDREN"]
CATEGORICAS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "NAME_INCOME_TYPE",
               "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "OCCUPATION_TYPE"]


def preparar(df, fuga):
    X = df[NUMERICAS + CATEGORICAS].copy()
    # DAYS_EMPLOYED = 365243 significa «sin empleo»: es un código, no un dato
    X["SIN_EMPLEO"] = (X.DAYS_EMPLOYED == 365243).astype(int)
    X.loc[X.SIN_EMPLEO == 1, "DAYS_EMPLOYED"] = np.nan
    # Ratios que usa cualquier analista de crédito
    X["CUOTA_SOBRE_INGRESOS"] = X.AMT_ANNUITY / X.AMT_INCOME_TOTAL
    X["CREDITO_SOBRE_BIEN"] = X.AMT_CREDIT / X.AMT_GOODS_PRICE
    if fuga:
        # Simula un dato que solo se conoce DESPUÉS: «nº de recordatorios de pago enviados».
        # Depende del resultado, así que en producción no existiría al conceder el préstamo.
        rng = np.random.default_rng(0)
        X["RECORDATORIOS_ENVIADOS"] = rng.poisson(0.3 + 2.5 * df.TARGET)
    return X


def modelo(tipo, X):
    num = [c for c in X.columns if c not in CATEGORICAS]
    if tipo == "boosting":
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100,
                                                        sparse_output=False), CATEGORICAS)],
                                 remainder="passthrough")
        return make_pipeline(prep, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                  random_state=0))
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="(nulo)"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=100)), CATEGORICAS),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["logistica", "boosting"], default="logistica")
    ap.add_argument("--fuga", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATOS, usecols=["TARGET"] + NUMERICAS + CATEGORICAS)
    X, y = preparar(df, args.fuga), df.TARGET
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    m = modelo(args.modelo, X).fit(X_tr, y_tr)
    p_tr, p_te = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_te)[:, 1]

    print(f"Modelo: {args.modelo}{' · CON FUGA' if args.fuga else ''}")
    print(f"AUC entrenamiento {roc_auc_score(y_tr, p_tr):.3f} · prueba {roc_auc_score(y_te, p_te):.3f}\n")
    print("Lift por decil (prueba, 10 = más riesgo)")
    print(lift_por_decil(y_te, p_te).round(3).to_string(), "\n")
    print("Calibración (prueba)")
    print(calibracion(y_te, p_te).round(3).to_string())


if __name__ == "__main__":
    main()
