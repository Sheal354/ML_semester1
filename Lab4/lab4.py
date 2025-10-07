from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Загрузка данных
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets

# 2. Преобразование целевой переменной в 3 класса по биологическим границам
# Классы: 0 = молодой (Rings <= 8), 1 = средний (9 <= Rings <= 10), 2 = старый (Rings >= 11)
y_rings = y.values.flatten()
y_class = np.where(y_rings <= 8, 0, np.where(y_rings <= 10, 1, 2))

# 3. Разделение на train (60%), val (20%), test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Размеры выборок: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Распределение классов в общей выборке: {np.bincount(y_class)}")

# 4. Кодирование 'Sex'
X_train = pd.get_dummies(X_train, columns=['Sex'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['Sex'], drop_first=False)
X_test = pd.get_dummies(X_test, columns=['Sex'], drop_first=False)

X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 5. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Объединение train + val для GridSearch
X_train_val = np.vstack([X_train_scaled, X_val_scaled])
y_train_val = np.hstack([y_train, y_val])
split_index = [-1] * len(X_train_scaled) + [0] * len(X_val_scaled)
pds = PredefinedSplit(test_fold=split_index)

#  ОБУЧЕНИЕ PERCEPTRON
print("\n" + "="*50)
print("ОБУЧЕНИЕ PERCEPTRON С ПАРАМЕТРАМИ ПО УМОЛЧАНИЮ...")
perceptron_base = Perceptron(random_state=42)
perceptron_base.fit(X_train_scaled, y_train)
acc_base_perceptron = accuracy_score(y_test, perceptron_base.predict(X_test_scaled))
print(f"Точность Perceptron по умолчанию: {acc_base_perceptron:.4f}")

print("\nGridSearch для Perceptron...")
param_grid_perceptron = {
    'penalty': [None, 'l1', 'l2'],
    'alpha': [1e-4, 1e-3, 1e-2],
    'max_iter': [1000]
}
grid_perceptron = GridSearchCV(
    Perceptron(random_state=42),
    param_grid_perceptron,
    cv=pds,
    scoring='accuracy',
    n_jobs=-1
)
grid_perceptron.fit(X_train_val, y_train_val)

best_acc_perceptron = grid_perceptron.best_score_
best_params_perceptron = grid_perceptron.best_params_

print(f"Лучшая точность Perceptron: {best_acc_perceptron:.4f}")
print(f"Лучшие параметры: {best_params_perceptron}")

# ОБУЧЕНИЕ MLPCLASSIFIER
print("\n" + "="*50)
print("ОБУЧЕНИЕ MLPCLASSIFIER С ПАРАМЕТРАМИ ПО УМОЛЧАНИЮ...")
mlp_base = MLPClassifier(random_state=42, max_iter=1000)
mlp_base.fit(X_train_scaled, y_train)
acc_base_mlp = accuracy_score(y_test, mlp_base.predict(X_test_scaled))
print(f"Точность MLPClassifier по умолчанию: {acc_base_mlp:.4f}")

print("\nGridSearch для MLPClassifier...")
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [1e-4, 1e-3, 1e-2],
    'max_iter': [1000]
}
grid_mlp = GridSearchCV(
    MLPClassifier(random_state=42),
    param_grid_mlp,
    cv=pds,
    scoring='accuracy',
    n_jobs=-1
)
grid_mlp.fit(X_train_val, y_train_val)

best_acc_mlp = grid_mlp.best_score_
best_params_mlp = grid_mlp.best_params_

print(f"Лучшая точность MLPClassifier: {best_acc_mlp:.4f}")
print(f"Лучшие параметры: {best_params_mlp}")

# ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ
final_acc_perceptron = accuracy_score(y_test, grid_perceptron.predict(X_test_scaled))
final_acc_mlp = accuracy_score(y_test, grid_mlp.predict(X_test_scaled))

print("\n" + "="*50)
print("ФИНАЛЬНАЯ ТОЧНОСТЬ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("="*50)
print(f"Perceptron (настроенный): {final_acc_perceptron:.4f}")
print(f"MLP (настроенный):        {final_acc_mlp:.4f}")

# ГРАФИКИ
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(grid_perceptron.cv_results_['mean_test_score'], marker='o', color='steelblue')
plt.title('GridSearch: точность Perceptron')
plt.xlabel('Индекс комбинации гиперпараметров')
plt.ylabel('Точность на валидационной выборке')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(grid_mlp.cv_results_['mean_test_score'], marker='o', color='seagreen')
plt.title('GridSearch: точность MLP')
plt.xlabel('Индекс комбинации гиперпараметров')
plt.ylabel('Точность на валидационной выборке')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()