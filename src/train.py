"""Генерация учебной базы клиентов и обучение модели."""

from __future__ import annotations

import math
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample_clients.csv"
MODEL = ROOT / "models" / "churn_risk.joblib"

FEATURES_CAT = ["segment", "channel"]
FEATURES_NUM = [
    "months_as_client",
    "recency_days",
    "frequency_90d",
    "avg_check",
    "support_tickets_90d",
]
TARGET = "target_churn"


def generate_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    segments = ["mass", "regular", "vip"]
    channels = ["ads", "organic", "referral", "email"]
    rows = []
    for i in range(1, n + 1):
        segment = random.choices(segments, weights=[50, 35, 15])[0]
        channel = random.choice(channels)
        months = random.randint(1, 48)
        recency = max(1, int(random.gauss(40, 25)))
        freq = max(0, int(random.gauss(4, 3)))
        check = round(max(300, random.gauss(2500, 1200)), 0)
        tickets = max(0, int(random.gauss(0.6, 1.2)))
        phone = f"+7 343 {random.randint(200, 399)}-{random.randint(10, 99)}-{random.randint(10, 99)}"
        logit = -2.2
        logit += 0.018 * recency
        logit -= 0.22 * min(freq, 12)
        logit += 0.28 * min(tickets, 6)
        logit -= 0.015 * min(months, 36)
        logit += {"mass": 0.25, "regular": 0.0, "vip": -0.35}[segment]
        logit += {"ads": 0.10, "organic": -0.05, "referral": -0.20, "email": 0.05}[channel]
        p = 1 / (1 + math.exp(-logit))
        target = int(random.random() < p)
        rows.append(
            {
                "client_id": f"C{i:04d}",
                "phone": phone,
                "segment": segment,
                "channel": channel,
                "months_as_client": months,
                "recency_days": recency,
                "frequency_90d": freq,
                "avg_check": check,
                "support_tickets_90d": tickets,
                "target_churn": target,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample()
    df.to_csv(DATA, index=False)
    x = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
                        ("num", "passthrough", FEATURES_NUM),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    auc = roc_auc_score(y_valid, pipe.predict_proba(x_valid)[:, 1])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features_cat": FEATURES_CAT,
            "features_num": FEATURES_NUM,
            "valid_roc_auc": float(auc),
        },
        MODEL,
    )
    full = pipe.predict_proba(x)[:, 1]
    groups = pd.cut(full, bins=[-0.01, 0.35, 0.55, 1.01], labels=["низкий", "средний", "высокий"])
    print(f"saved {MODEL}")
    print(f"rows={len(df)} churn_share={y.mean():.3f} valid ROC-AUC={auc:.3f}")
    print("groups:", groups.value_counts().to_dict())


if __name__ == "__main__":
    main()
