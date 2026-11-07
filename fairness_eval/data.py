from functools import cache
import numpy as np
import pandas as pd
from . import config as cfg

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
# attribute names from the UCI documentation (german.doc)
GERMAN_COLUMNS = [
    "checking", "duration", "credit_history", "purpose", "amount", "savings", "employment",
    "installment_rate", "personal_status", "other_debtors", "residence_since", "property", "age",
    "other_plans", "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "credit",
]
COMPAS_FEATURES = [
    "sex", "age", "age_cat", "race", "c_charge_degree", "priors_count",
    "juv_fel_count", "juv_misd_count", "juv_other_count",
]


def adult():
    train, test = cfg.URLS["adult"]
    raw = pd.concat([
        pd.read_csv(train, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?"),
        pd.read_csv(test, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?", skiprows=1),
    ]).dropna()
    # fnlwgt is a census sampling weight, not a property of the person; education duplicates education_num
    df = raw.drop(columns=["fnlwgt", "education", "income"])
    y = raw.income.str.rstrip(".").eq(">50K").to_numpy(int)
    z = raw.sex.eq("Male").to_numpy(int)
    return df.assign(sex=z), y, z


def german():
    raw = pd.read_csv(cfg.URLS["german"][0], sep=" ", names=GERMAN_COLUMNS)
    y = raw.credit.eq(1).to_numpy(int)  # 1 = good credit
    z = (raw.age > 25).to_numpy(int)
    return raw.drop(columns="credit"), y, z


def compas():
    raw = pd.read_csv(cfg.URLS["compas"][0])
    # the filters of the ProPublica analysis
    raw = raw[
        raw.days_b_screening_arrest.between(-30, 30)
        & raw.is_recid.ne(-1)
        & raw.c_charge_degree.ne("O")
        & raw.score_text.ne("N/A")
        & raw.race.isin(["African-American", "Caucasian"])
    ]
    # the COMPAS score itself is left out, the model predicts from the defendant's record
    df = raw[COMPAS_FEATURES].copy()
    y = raw.two_year_recid.eq(0).to_numpy(int)  # favorable outcome: no recidivism within two years
    z = raw.race.eq("Caucasian").to_numpy(int)
    return df.assign(race=z), y, z


@cache
def load(dataset):
    """Features with the sensitive attribute one hot encoded as numbers, labels y in {0, 1} and groups z in {0, 1}.
    z stays in the features, as in the authors' code (xz_train)."""
    df, y, z = {"adult": adult, "german": german, "compas": compas}[dataset]()
    x = pd.get_dummies(df, drop_first=True, dtype=float)
    return x.reset_index(drop=True), y, z


def summary():
    rows = []
    for name in cfg.DATASETS:
        x, y, z = load(name)
        rows.append({
            "dataset": name, "rows": len(y), "features": x.shape[1], "sensitive": cfg.SENSITIVE[name],
            "share_privileged": z.mean(), "base_rate": y.mean(),
            "base_rate_z0": y[z == 0].mean(), "base_rate_z1": y[z == 1].mean(),
        })
    return pd.DataFrame(rows)
