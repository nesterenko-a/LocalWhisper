# Разработка

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe app.py
```

Проверки без скачивания модели:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Для ручной проверки загрузите короткую запись с речью, проверьте TXT/SRT/VTT, затем повторите с микрофоном и с включённым офлайн-режимом. GPU проверяется отдельно на совместимом оборудовании.

Не добавляйте в Git модели, виртуальное окружение, личные записи, результаты транскрипции и секреты. Зависимости верхнего уровня закреплены в `requirements.txt`, полный набор текущего окружения — в `requirements-lock.txt`.
