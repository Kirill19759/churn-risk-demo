"""Локальное демо: база клиентов → риск ухода → выгрузка."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.score import REQUIRED, load_bundle, score_frame

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_clients.csv"

st.set_page_config(page_title="tagiltsev_ml · риск ухода", layout="wide")
st.title("Риск ухода клиента")
st.caption("Демонстрационный сервис tagiltsev_ml. Не промышленное внедрение у конкретного клиента.")

bundle = load_bundle()
st.write(
    f"Модель обучена на учебной базе. "
    f"Качество на отложенной части: ROC-AUC {bundle['valid_roc_auc']:.2f}. "
    f"На вашей CRM цифра будет другой."
)

st.subheader("Что нужно в файле")
st.code(", ".join(REQUIRED))

uploaded = st.file_uploader("Загрузите CSV с клиентами", type=["csv"])
use_sample = st.checkbox("Показать учебный пример", value=uploaded is None)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(SAMPLE)
else:
    st.stop()

st.write(f"Строк во входе: {len(df)}")

try:
    ranked = score_frame(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Высокий риск", int((ranked["risk_group"] == "высокий").sum()))
c2.metric("Средний", int((ranked["risk_group"] == "средний").sum()))
c3.metric("Низкий", int((ranked["risk_group"] == "низкий").sum()))
c4.metric("Средняя оценка", f"{ranked['churn_score'].mean():.2f}")

st.dataframe(ranked.head(30), use_container_width=True)
st.download_button(
    "Скачать список риска",
    ranked.to_csv(index=False).encode("utf-8-sig"),
    file_name="clients_churn_risk.csv",
    mime="text/csv",
)
