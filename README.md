# logistic-regression-classification
An example of data classification using logistic regression. Visualization of the class boundary. 

# Logistic Regression Classification

Пример классификации данных с использованием логистической регрессии.  
В этом проекте я использовал **scikit-learn** для построения модели и **matplotlib** для визуализации границ классов.

## 📂 Содержимое проекта
- `Logistic_Regression.py` — код для генерации данных, обучения модели и построения графика.
- Визуализация границы классов (см. ниже пример кода).

## 🚀 Используемые библиотеки
- numpy
- matplotlib
- scikit-learn

## 🔑 Основной код
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

model = LogisticRegression(max_iter=1000, random_state=42)
model = model.fit(X, y)
