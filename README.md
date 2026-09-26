# Кабинет удержания клиентов

Демонстрационный кабинет **tagiltsev_ml**.

Живой адрес: https://churn-risk-demo-ejkdaowl5d3hqsm2xyccx8.streamlit.app/

Вход: логин `demo`, пароль `tagiltsev-ml`.

На экране — план контактов на неделю, почему связаться и учебный телефон. Это учебные данные, не база клиента.

Пароль только закрывает страницу от случайного просмотра. Это не защита корпоративных данных.

После простоя бесплатный адрес может открываться 20–30 секунд.

## Как запустить локально

```text
git clone https://github.com/Kirill19759/churn-risk-demo.git
cd churn-risk-demo
python -m pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Контакты

Кирилл Тагильцев · https://t.me/tagiltsev_ml · tagiltsev.ml@mail.ru
