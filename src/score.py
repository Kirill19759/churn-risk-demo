"""Оценка риска ухода и короткие пояснения."""

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


def reason_text(row: pd.Series) -> str:
    parts = []
    if row["recency_days"] >= 45:
        parts.append(f"не был {int(row['recency_days'])} дней")
    if row["frequency_90d"] <= 1:
        parts.append("мало покупок за 90 дней")
    if row["support_tickets_90d"] >= 2:
        parts.append("есть обращения в поддержку")
    if row["segment"] == "vip" and row["churn_score"] >= 0.55:
        parts.append("VIP, потеря дороже")
    if not parts:
        parts.append("сочетание признаков выше фона")
    return "; ".join(parts)


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
    out["reason"] = out.apply(reason_text, axis=1)
    out["action"] = out["risk_group"].map(
        {
            "высокий": "позвонить на этой неделе",
            "средний": "письмо или предложение",
            "низкий": "не трогать сейчас",
        }
    )
    return out.sort_values("churn_score", ascending=False).reset_index(drop=True)
