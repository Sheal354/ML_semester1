import pandas as pd
import re

pd.set_option("display.precision", 2)
# Увеличение количество отображаемых столбцов
pd.set_option('display.max_columns', None)
# Запретить перенос строк
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)



def first_name(full_name):
    """
    Извлекает имя из строки вида:
    'Braund, Mr. Owen Harris' -> 'Owen'
    'Nasser, Mrs. Nicholas (Adele Achem)' -> 'Adele'
    'Heikkinen, Miss. Laina' -> 'Laina'
    """
    # Ищем имя после титула и до скобок или пробелов
    match = re.search(r',\s*\w+\.?\s*([A-Za-z]+)', full_name)
    if match:
        name = match.group(1)
        # Если имя в скобках, ищем его внутри
        bracket_match = re.search(r'\(([A-Za-z]+)', full_name)
        if bracket_match:
            name = bracket_match.group(1)
        return name
    return None

def main():
    # Чтение данных в Pandas DataFrame
    data = pd.read_csv('titanic_train.csv',
                      index_col='PassengerId')

    # Сколько мужчин / женщин было на борту?
    print("Задача 1")
    print(data['Sex'].value_counts())

    # Определите распределение функции Pclass. Теперь Для мужчин и женщин отдельно.
    # Сколько людей из второго класса было на борту?
    print("Задача 2")
    print(data.groupby(['Pclass', 'Sex']).size().unstack(fill_value=0))

    # Каковы медиана и стандартное отклонение Fare? Округлите до 2-х знаков после запятой.
    print("Задача 3")
    print(f"Медиана Fare: {round(data['Fare'].median(), 2)}")
    print(f"Стандартное отклонение Fare: {round(data['Fare'].std(), 2)}")

    # Правда ли, что средний возраст выживших людей выше, чем у пассажиров, которые в конечном итоге умерли?
    print("Задача 4")
    print(data.groupby('Survived')['Age'].mean())

    # Это правда, что пассажиры моложе 30 лет. выжили чаще, чем те, кому больше 60 лет.
    # Каковы доли выживших людей среди молодых и пожилых людей?
    print("Задача 5")
    # Отфильтровываем строки с возрастом < 30
    young = data[data['Age'] < 30]
    # Отфильтровываем строки с возрастом > 60
    old = data[data['Age'] > 60]
    print(f"Доля выживших среди моложе 30: {round(young['Survived'].mean(), 3) * 100}")
    print(f"Доля выживших среди старше 60: {round(old['Survived'].mean(), 3) * 100}")

    # Правда ли, что женщины выживали чаще мужчин? Каковы доли выживших людей среди мужчин и женщин?
    print("Задача 6")
    # Отфильтровываем строки с мужчинами и женщинами
    man = data[data['Sex'] == "male"]
    women = data[data['Sex'] == "female"]
    print(f"Доля выживших среди мужчин: {round(man['Survived'].mean(), 3) * 100}")
    print(f"Доля выживших среди женщин: {round(women['Survived'].mean(), 3) * 100}")

    # Какое имя наиболее популярно среди пассажиров мужского пола?
    print("Задача 7")
    print(man['Name'].apply(first_name).value_counts().head())

    # Как средний возраст мужчин / женщин зависит от Pclass?
    print("Задача 8")
    print(data.groupby(['Pclass', 'Sex'])['Age'].mean().unstack(fill_value=0))

if __name__ == "__main__":
    main()