import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
num_points = 21
x_range = np.linspace(-2 * np.pi, 2 * np.pi, num_points)

# Генерируем сетку
x1, x2 = np.meshgrid(x_range, x_range)

# Выравниваем в 1D массивы
x1_flat = x1.flatten()
x2_flat = x2.flatten()

# Вычисляем y
y = np.sin(x1_flat + x2_flat)

# Создаем DataFrame
df = pd.DataFrame({
    'x1': x1_flat,
    'x2': x2_flat,
    'y': y
})

# Сохраняем в CSV
df.to_csv('data.csv', index=False)

# Вычисляем средние значения
mean_x1 = df['x1'].mean()
mean_x2 = df['x2'].mean()

print(f"Среднее x1: {mean_x1:.4f}")
print(f"Среднее x2: {mean_x2:.4f}")

# Фильтрация строк: x1 < mean_x1 или x2 < mean_x2
filtered_df = df[(df['x1'] < mean_x1) | (df['x2'] < mean_x2)]

# Сохраняем отфильтрованные строки в новый CSV
filtered_df.to_csv('filtered_data.csv', index=False)

print(f"Количество строк в исходном файле: {len(df)}")
print(f"Количество строк в отфильтрованном файле: {len(filtered_df)}")

# Вывод статистики
print("\n--- Статистика по столбцам ---")
for col in ['x1', 'x2', 'y']:
    mean_val = df[col].mean()
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"{col}:")
    print(f"  Среднее: {mean_val:.4f}")
    print(f"  Минимум: {min_val:.4f}")
    print(f"  Максимум: {max_val:.4f}")

# Построение 2D-графиков
def find_nearest_value(series, value):
    idx = (series - value).abs().idxmin()
    return series[idx]

x1_const_input = -1
x2_const_input = 0.5

x2_const = find_nearest_value(df['x2'], x2_const_input)
x1_const = find_nearest_value(df['x1'], x1_const_input)

print(f"Ближайшее значение x1: {x1_const}")
print(f"\nБлижайшее значение x2: {x2_const}")

subset_x1 = df[df['x2'] == x2_const]
subset_x2 = df[df['x1'] == x1_const]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
if not subset_x1.empty:
    plt.plot(subset_x1['x1'], subset_x1['y'], marker='o', linestyle='-', markersize=4)
    plt.title(f'y(x1) при x2 ≈ {x2_const:.2f}')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.grid(True)
    plt.ylabel('y')

plt.subplot(1, 2, 2)
if not subset_x2.empty:
    plt.plot(subset_x2['x2'], subset_x2['y'], marker='o', linestyle='-', markersize=4)
    plt.title(f'y(x2) при x1 ≈ {x1_const:.2f}')
    plt.xlabel('x2')
    plt.ylabel('y')
    plt.grid(True)

plt.tight_layout() # Автоматически подгоняет графики, чтобы подписи не налезали друг на друга.
plt.show()

# Построение 3D-графика функции y(x1, x2)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Используем исходные сетки для 3D-графика
X1, X2 = np.meshgrid(x_range, x_range)
Y = np.sin(X1 + X2)

ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D график: y = sin(x1 + x2)')

plt.show()