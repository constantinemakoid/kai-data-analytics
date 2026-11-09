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

## lab2: Аналіз даних за допомогою Python

Описова статистика, regplot і boxplot для ознак набору UCI Automobile, value_counts, groupby, зведена таблиця і теплова карта drive-wheels × body-style, кореляція bore, stroke, compression-ratio, horsepower, коефіцієнт Пірсона і P-value для wheel-base і horsepower відносно price, однофакторний дисперсійний аналіз (F-test) для категоріальних ознак.

Перед аналізом рядки без price видаляються, решта пропусків заповнюється середнім (числові) і модою (num-of-doors).

```
cd lab2
jupyter notebook exploratory_analysis.ipynb
```

або `python exploratory_analysis.py`. Графіки зберігаються у lab2/figures.

## lab3: Model Development

Проста лінійна регресія engine-size → price, множинна регресія price ~ normalized-losses + highway-mpg, поліноміальна модель 11 порядку для highway-mpg (numpy.polyfit і Pipeline зі StandardScaler і PolynomialFeatures), порівняння SLR, MLR, поліноміальних моделей і Pipeline за R² і MSE на всій вибірці, на тестовій вибірці та з 5-кратною крос-валідацією.

```
cd lab3
jupyter notebook model_development.ipynb
```

або `python model_development.py`. Графіки зберігаються у lab3/figures.
