import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Шаг 1: Загрузка данных из файла
data = []
with open('data.txt', 'r') as file:
    for line in file:
        # Удаляем пробелы в начале и конце, разбиваем строку на элементы
        values = line.strip().split()
        # Преобразуем значения в числа с плавающей точкой
        if values:  # Проверка на пустую строку
            data.append([float(value) for value in values])

# Проверяем количество строк
print(f"Всего строк в датасете: {len(data)}")

# Шаг 2: Создаем список индексов и перемешиваем его
random.seed(42)  # Для воспроизводимости
indices = list(range(len(data)))
random.shuffle(indices)

# Шаг 3: Определяем индекс разделения (70% на обучение, 30% на тест)
train_size = int(0.7 * len(data))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Шаг 4: Формируем обучающую и тестовую выборки
X_train = [data[i] for i in train_indices]
X_test = [data[i] for i in test_indices]

# Выводим размеры выборок для проверки
print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Разделяем на признаки и целевые переменные
# В датасете 18 признаков:
# - 1-16: различные параметры системы
# - 17-й признак (индекс 16): GT Compressor decay state coefficient
# - 18-й признак (индекс 17): GT Turbine decay state coefficient
y1_train = [row[16] for row in X_train]  # 17-й признак (индекс 16)
y1_test = [row[16] for row in X_test]
y2_train = [row[17] for row in X_train]  # 18-й признак (индекс 17)
y2_test = [row[17] for row in X_test]

X_train_features = [row[:16] for row in X_train]  # 16 признаков
X_test_features = [row[:16] for row in X_test]  # 16 признаков

print(f"Формат обучающей выборки (признаки): {len(X_train_features)} наблюдений")
print(f"Формат обучающей выборки (целевые переменные): {len(y1_train)} значений")

# 1. Обучаем модель линейной регрессии
print("\nМодель для коэффициента деградации компрессора (17-й признак):")
regressor1 = LinearRegression().fit(X_train_features, y1_train)
y1_pred = regressor1.predict(X_test_features)

# Оцениваем качество модели
mse1 = mean_squared_error(y1_test, y1_pred)
r2_1 = r2_score(y1_test, y1_pred)

print(f"Mean squared error: {mse1:.6f}")
print(f"Coefficient of determination (R²): {r2_1:.4f}")

# График истинных значений vs предсказанных для компрессора
plt.figure(figsize=(10, 6))
plt.scatter(y1_test, y1_pred, alpha=0.7)
plt.plot([min(y1_test), max(y1_test)], [min(y1_test), max(y1_test)], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Compressor Decay Coefficient)')
plt.show()

# Гистограмма ошибок для компрессора
plt.figure(figsize=(10, 6))
plt.hist(np.array(y1_test) - np.array(y1_pred), bins=30, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution (Compressor Decay Coefficient)')
plt.show()

# 2. Обучаем модель линейной регрессии для коэффициента деградации турбины
print("\nМодель для коэффициента деградации турбины (18-й признак):")
regressor2 = LinearRegression().fit(X_train_features, y2_train)
y2_pred = regressor2.predict(X_test_features)

# Оцениваем качество модели
mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"Mean squared error: {mse2:.6f}")
print(f"Coefficient of determination (R²): {r2_2:.4f}")

# График истинных значений vs предсказанных для турбины
plt.figure(figsize=(10, 6))
plt.scatter(y2_test, y2_pred, alpha=0.7)
plt.plot([min(y2_test), max(y2_test)], [min(y2_test), max(y2_test)], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Turbine Decay Coefficient)')
plt.show()

# Гистограмма ошибок для турбины
plt.figure(figsize=(10, 6))
plt.hist(np.array(y2_test) - np.array(y2_pred), bins=30, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution (Turbine Decay Coefficient)')
plt.show()

# Часть 2: Полиномиальная регрессия
print("\nПолиномиальная регрессия с различными степенями:")

# Нормализуем данные ПЕРЕД генерацией полиномиальных признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Ограничимся только первыми тремя степенями
degrees = [1, 2, 3]
train_mae1 = []  # MAE для коэффициента деградации компрессора (обучающая)
test_mae1 = []  # MAE для коэффициента деградации компрессора (тестовая)
train_mae2 = []  # MAE для коэффициента деградации турбины (обучающая)
test_mae2 = []  # MAE для коэффициента деградации турбины (тестовая)

for degree in degrees:
    # Модель 1: для коэффициента деградации компрессора
    model1 = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model1.fit(X_train_scaled, y1_train)

    # Оцениваем на обучающей выборке
    y1_train_pred = model1.predict(X_train_scaled)
    train_mae_1 = mean_absolute_error(y1_train, y1_train_pred)
    train_mae1.append(train_mae_1)

    # Оцениваем на тестовой выборке
    y1_test_pred = model1.predict(X_test_scaled)
    test_mae_1 = mean_absolute_error(y1_test, y1_test_pred)
    test_mae1.append(test_mae_1)

    # Модель 2: для коэффициента деградации турбины
    model2 = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model2.fit(X_train_scaled, y2_train)

    # Оцениваем на обучающей выборке
    y2_train_pred = model2.predict(X_train_scaled)
    train_mae_2 = mean_absolute_error(y2_train, y2_train_pred)
    train_mae2.append(train_mae_2)

    # Оцениваем на тестовой выборке
    y2_test_pred = model2.predict(X_test_scaled)
    test_mae_2 = mean_absolute_error(y2_test, y2_test_pred)
    test_mae2.append(test_mae_2)

    # Выводим результаты
    print(f"Степень {degree}:")
    print(f"  Коэффициент деградации компрессора (17-й признак):")
    print(f"    MAE обучающая = {train_mae_1:.6f}, MAE тестовая = {test_mae_1:.6f}")
    print(f"  Коэффициент деградации турбины (18-й признак):")
    print(f"    MAE обучающая = {train_mae_2:.6f}, MAE тестовая = {test_mae_2:.6f}")

# График зависимости MAE от степени полинома для коэффициента деградации компрессора
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mae1, 'o-', label='Обучающая выборка (компрессор)', linewidth=2)
plt.plot(degrees, test_mae1, 's-', label='Тестовая выборка (компрессор)', linewidth=2)
plt.xlabel('Степень полинома')
plt.ylabel('MAE (средняя абсолютная ошибка)')
plt.title('Зависимость точности модели от степени полиномиальной функции (Compressor)')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости MAE от степени полинома для коэффициента деградации турбины
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mae2, 'o-', label='Обучающая выборка (турбина)', linewidth=2)
plt.plot(degrees, test_mae2, 's-', label='Тестовая выборка (турбина)', linewidth=2)
plt.xlabel('Степень полинома')
plt.ylabel('MAE (средняя абсолютная ошибка)')
plt.title('Зависимость точности модели от степени полиномиальной функции (Turbine)')
plt.legend()
plt.grid(True)
plt.show()

# График разницы между обучающей и тестовой точностью для компрессора
plt.figure(figsize=(10, 6))
gap1 = [test_mae1[i] - train_mae1[i] for i in range(len(degrees))]  # положительные значения = переобучение
plt.plot(degrees, gap1, 'd-', linewidth=2, color='purple')
plt.xlabel('Степень полинома')
plt.ylabel('Разница MAE (тестовая - обучающая)')
plt.title('Разница в точности между обучающей и тестовой выборками (Compressor)')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# График разницы между обучающей и тестовой точностью для турбины
plt.figure(figsize=(10, 6))
gap2 = [test_mae2[i] - train_mae2[i] for i in range(len(degrees))]  # положительные значения = переобучение
plt.plot(degrees, gap2, 'd-', linewidth=2, color='purple')
plt.xlabel('Степень полинома')
plt.ylabel('Разница MAE (тестовая - обучающая)')
plt.title('Разница в точности между обучающей и тестовой выборками (Turbine)')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Часть 3: Регуляризованная модель (Ridge)
print("\nРегуляризованная модель (Ridge) с различными коэффициентами регуляризации:")

# Набор значений alpha (коэффициентов регуляризации)
alphas = [0.01, 0.1, 1, 10, 100, 1000]
train_ridge_scores1 = []  # Для компрессора на обучающей выборке
test_ridge_scores1 = []  # Для компрессора на тестовой выборке
train_ridge_scores2 = []  # Для турбины на обучающей выборке
test_ridge_scores2 = []  # Для турбины на тестовой выборке
train_ridge_mae1 = []  # MAE для компрессора на обучающей
test_ridge_mae1 = []  # MAE для компрессора на тестовой
train_ridge_mae2 = []  # MAE для турбины на обучающей
test_ridge_mae2 = []  # MAE для турбины на тестовой

for alpha in alphas:
    # Модель 1: для коэффициента деградации компрессора
    model1 = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model1.fit(X_train_features, y1_train)

    # Оцениваем на обучающей выборке
    y1_train_pred = model1.predict(X_train_features)
    train_r2_1 = r2_score(y1_train, y1_train_pred)
    train_mae_1 = mean_absolute_error(y1_train, y1_train_pred)
    train_ridge_scores1.append(train_r2_1)
    train_ridge_mae1.append(train_mae_1)

    # Оцениваем на тестовой выборке
    y1_test_pred = model1.predict(X_test_features)
    test_r2_1 = r2_score(y1_test, y1_test_pred)
    test_mae_1 = mean_absolute_error(y1_test, y1_test_pred)
    test_ridge_scores1.append(test_r2_1)
    test_ridge_mae1.append(test_mae_1)

    # Модель 2: для коэффициента деградации турбины
    model2 = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model2.fit(X_train_features, y2_train)

    # Оцениваем на обучающей выборке
    y2_train_pred = model2.predict(X_train_features)
    train_r2_2 = r2_score(y2_train, y2_train_pred)
    train_mae_2 = mean_absolute_error(y2_train, y2_train_pred)
    train_ridge_scores2.append(train_r2_2)
    train_ridge_mae2.append(train_mae_2)

    # Оцениваем на тестовой выборке
    y2_test_pred = model2.predict(X_test_features)
    test_r2_2 = r2_score(y2_test, y2_test_pred)
    test_mae_2 = mean_absolute_error(y2_test, y2_test_pred)
    test_ridge_scores2.append(test_r2_2)
    test_ridge_mae2.append(test_mae_2)

    # Выводим результаты
    print(f"Alpha = {alpha:.5f}:")
    print(f"  Коэффициент деградации компрессора (17-й признак):")
    print(f"    R² обучающая = {train_r2_1:.4f}, R² тестовая = {test_r2_1:.4f}")
    print(f"    MAE обучающая = {train_mae_1:.6f}, MAE тестовая = {test_mae_1:.6f}")
    print(f"  Коэффициент деградации турбины (18-й признак):")
    print(f"    R² обучающая = {train_r2_2:.4f}, R² тестовая = {test_r2_2:.4f}")
    print(f"    MAE обучающая = {train_mae_2:.6f}, MAE тестовая = {test_mae_2:.6f}")

# График зависимости R² от коэффициента регуляризации для компрессора
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, train_ridge_scores1, 'o-', label='Обучающая выборка (компрессор)', linewidth=2)
plt.semilogx(alphas, test_ridge_scores1, 's-', label='Тестовая выборка (компрессор)', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('R² (коэффициент детерминации)')
plt.title('Зависимость точности модели от коэффициента регуляризации (Compressor)')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости R² от коэффициента регуляризации для турбины
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, train_ridge_scores2, 'o-', label='Обучающая выборка (турбина)', linewidth=2)
plt.semilogx(alphas, test_ridge_scores2, 's-', label='Тестовая выборка (турбина)', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('R² (коэффициент детерминации)')
plt.title('Зависимость точности модели от коэффициента регуляризации (Turbine)')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости MAE от коэффициента регуляризации для компрессора
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, train_ridge_mae1, 'o-', label='Обучающая выборка (компрессор)', linewidth=2)
plt.semilogx(alphas, test_ridge_mae1, 's-', label='Тестовая выборка (компрессор)', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('MAE (средняя абсолютная ошибка)')
plt.title('Зависимость ошибки модели от коэффициента регуляризации (Compressor)')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости MAE от коэффициента регуляризации для турбины
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, train_ridge_mae2, 'o-', label='Обучающая выборка (турбина)', linewidth=2)
plt.semilogx(alphas, test_ridge_mae2, 's-', label='Тестовая выборка (турбина)', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('MAE (средняя абсолютная ошибка)')
plt.title('Зависимость ошибки модели от коэффициента регуляризации (Turbine)')
plt.legend()
plt.grid(True)
plt.show()

# График разницы между обучающей и тестовой точностью для компрессора
plt.figure(figsize=(10, 6))
gap1_ridge = [train_ridge_scores1[i] - test_ridge_scores1[i] for i in range(len(alphas))]
plt.semilogx(alphas, gap1_ridge, 'd-', linewidth=2, color='purple')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('Разница R² (обучающая - тестовая)')
plt.title('Разница в точности между обучающей и тестовой выборками (Compressor)')
plt.grid(True)
plt.show()

# График разницы между обучающей и тестовой точностью для турбины
plt.figure(figsize=(10, 6))
gap2_ridge = [train_ridge_scores2[i] - test_ridge_scores2[i] for i in range(len(alphas))]
plt.semilogx(alphas, gap2_ridge, 'd-', linewidth=2, color='purple')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('Разница R² (обучающая - тестовая)')
plt.title('Разница в точности между обучающей и тестовой выборками (Turbine)')
plt.grid(True)
plt.show()