# Кабинет удержания клиентов

Демонстрационный кабинет **tagiltsev_ml**.

Вход: логин `demo`, пароль `tagiltsev-ml`.

На экране — кого трогать на этой неделе, почему и какое действие. Это учебные данные, не база клиента.

Пароль только закрывает страницу от случайного просмотра. Это не защита корпоративных данных.

## Локально на Windows

```text
cd C:\Users\MLNW\churn-risk-demo
git pull
python src\train.py
streamlit run app.py
```

Откройте http://localhost:8501

## Как выложить на бесплатный адрес

1. Войдите на https://share.streamlit.io через GitHub.
2. Create app.
3. Repository: `Kirill19759/churn-risk-demo`.
4. Main file: `app.py`.
5. Deploy.
6. Ссылку вставьте в Telegram и README.

Первый запуск может занять минуту: приложение само соберёт учебную базу, если модели ещё нет.

## Контакты

Кирилл Тагильцев · https://t.me/tagiltsev_ml · tagiltsev.ml@mail.ru
