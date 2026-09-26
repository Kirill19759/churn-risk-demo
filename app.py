"""Кабинет удержания: вход по паролю и карточки контакта."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.score import load_bundle, score_frame
from src.train import MODEL, main as train_main

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_clients.csv"
DEMO_USER = "demo"
DEMO_PASSWORD = "tagiltsev-ml"

st.set_page_config(page_title="tagiltsev_ml · кабинет удержания", layout="wide")


def ensure_model() -> None:
    if not MODEL.exists() or not SAMPLE.exists():
        train_main()


def login_view() -> None:
    st.markdown("### tagiltsev_ml")
    st.title("Кабинет удержания клиентов")
    st.write("Учебное демо для показа пилота. Не промышленное внедрение.")
    with st.form("login"):
        user = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
    if submitted:
        if user == DEMO_USER and password == DEMO_PASSWORD:
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Неверный логин или пароль.")


def cabinet_view() -> None:
    ensure_model()
    bundle = load_bundle()
    ranked = score_frame(pd.read_csv(SAMPLE))
    high = ranked[ranked["risk_group"] == "высокий"]

    top = st.columns([4, 1])
    top[0].markdown("**tagiltsev_ml** · кабинет удержания")
    if top[1].button("Выйти"):
        st.session_state["auth"] = False
        st.rerun()

    st.title("План контактов на неделю")
    st.caption(
        f"Учебная база, {len(ranked)} клиентов. "
        f"ROC-AUC на отложенной части этой выборки: {bundle['valid_roc_auc']:.2f}. "
        "На чужой CRM цифра будет другой."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Высокий риск", int(len(high)))
    c2.metric("Позвонить в приоритете", int(min(10, len(high))))
    c3.metric("Всего в базе", int(len(ranked)))

    st.subheader("С кем связаться сначала")
    cards = high.head(9)
    rows = list(cards.iterrows())
    for start in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, rows[start : start + 3]):
            phone = row["phone"] if "phone" in row.index else ""
            with col:
                st.markdown(
                    f"**{row['client_id']}** · {row['segment']}  \n"
                    f"{phone}  \n"
                    f"Риск {row['churn_score']:.2f}  \n"
                    f"{row['reason']}  \n"
                    f"_{row['action']}_"
                )

    st.download_button(
        "Скачать список для звонков",
        high.to_csv(index=False).encode("utf-8-sig"),
        file_name="week_contact_list.csv",
        mime="text/csv",
    )
    show_cols = [
        "client_id",
        "phone",
        "segment",
        "recency_days",
        "frequency_90d",
        "churn_score",
        "risk_group",
        "reason",
        "action",
    ]
    show_cols = [c for c in show_cols if c in ranked.columns]
    with st.expander("Вся база"):
        st.dataframe(ranked[show_cols], use_container_width=True)

    st.write("Кирилл Тагильцев · [@tagiltsev_ml](https://t.me/tagiltsev_ml) · tagiltsev.ml@mail.ru")


if "auth" not in st.session_state:
    st.session_state["auth"] = False

if st.session_state["auth"]:
    cabinet_view()
else:
    login_view()
