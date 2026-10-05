# kai-data-analytics

Лабораторні роботи з дисципліни «Дата аналітика», КАІ, група Б-121-23-1-ПІ, Макоїд Костянтин Вікторович.

## lab1: Data Wrangling

Очищення набору даних UCI Automobile (imports-85.data): заміна "?" на NaN, заповнення пропусків середнім і модою, приведення типів, нормалізація height, бінінг horsepower, індикаторні змінні для aspiration.

Дані в репозиторій не додаються: скрипт завантажує їх з https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data у теку lab1/data під час першого запуску.

Запуск:

```
pip install -r requirements.txt
cd lab1
jupyter notebook data_wrangling.ipynb
```

або

```
cd lab1
python data_wrangling.py
```

Графіки зберігаються у lab1/figures, очищений набір у lab1/data/auto_clean.csv.
