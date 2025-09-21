# Logistic Regression Classification

An example of data classification using logistic regression.  
In this project, I used **scikit-learn** to build the model and **matplotlib** to visualize the class boundaries.

## 📂 Project contents
- `Logistic_Regression.py` — code for generating data, training the model, and plotting the graph.
- Visualization of class boundaries (see code example below).

## 🚀 Libraries used
- numpy
- matplotlib
- scikit-learn

## 🔑 Main code
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
    random_state=42)


model = LogisticRegression(max_iter=1000, random_state=42)
model = model.fit(X, y)
