"""Оценка риска ухода по сохранённой модели."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_risk.joblib"
REQUIRED = [
    "segment",
    "channel",
    "months_as_client",
    "recency_days",
    "frequency_90d",
    "avg_check",
    "support_tickets_90d",
]


def load_bundle():
    if not MODEL.exists():
        raise FileNotFoundError("Сначала запустите: python src/train.py")
    return joblib.load(MODEL)


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_bundle()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("В таблице нет колонок: " + ", ".join(missing))
    x = df[bundle["features_cat"] + bundle["features_num"]]
    out = df.copy()
    out["churn_score"] = bundle["pipeline"].predict_proba(x)[:, 1]
    out["risk_group"] = pd.cut(
        out["churn_score"],
        bins=[-0.01, 0.35, 0.55, 1.01],
        labels=["низкий", "средний", "высокий"],
    )
    return out.sort_values("churn_score", ascending=False).reset_index(drop=True)
