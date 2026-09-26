# Риск ухода клиента

Демонстрационный сервис **tagiltsev_ml**.

Задача: из базы клиентов получить список, кого трогать первым, пока человек не ушёл.

Это открытое демо, не внедрение у клиента. Выборка синтетическая. На чужой CRM качество будет другим.

## Что на выходе

- оценка вероятности ухода (`churn_score`);
- группа: низкий / средний / высокий риск;
- CSV, отсортированный по убыванию риска.

## Запуск на Windows

```text
cd C:\Users\MLNW\churn-risk-demo
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\train.py
streamlit run app.py
```

Если активация окружения запрещена, пакеты уже стоят глобально — достаточно `python src\train.py` и `streamlit run app.py`.

## Колонки

`segment`, `months_as_client`, `recency_days`, `frequency_90d`, `avg_check`, `support_tickets_90d`, `channel`

`target_churn` нужен только для обучения.

## Ограничения

Сервис не обещает, что клиенты останутся. Он ставит порядок контакта.

## Контакты

Кирилл Тагильцев · tagiltsev_ml  
Telegram: https://t.me/tagiltsev_ml  
Почта: tagiltsev.ml@mail.ru
