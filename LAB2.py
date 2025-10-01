
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_tangent_data():
    n_points = 450
    x1 = np.linspace(-3, 3, n_points)
    x2 = np.linspace(-3, 3, n_points)


    result = np.tan(x1 + x2) * np.sin(x1)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'result_value': result
    })

    df.to_csv('tangent_data.csv', index=False)
    print("Файл tangent_data.csv успешно создан")
    print(f"Количество строк: {len(df)}")
    return df


def plot_tangent_graphs(df):
    const_x2 = df['x2'].mean()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(df['x1'], df['result_value'], alpha=0.6, s=10)
    plt.title(f'result_value(x1) при x2 = {const_x2:.2f}')
    plt.xlabel('x1')
    plt.ylabel('result_value')
    plt.grid(True, alpha=0.3)

    const_x1 = df['x1'].mean()

    plt.subplot(1, 2, 2)
    plt.scatter(df['x2'], df['result_value'], alpha=0.6, s=10)
    plt.title(f'result_value(x2) при x1 = {const_x1:.2f}')
    plt.xlabel('x2')
    plt.ylabel('result_value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tangent_graphs.png', dpi=300)
    plt.show()


def print_tangent_statistics(df):
    print("\nСтатистика по столбцам:")
    print("=" * 40)

    for column in df.columns:
        print(f"{column}:")
        print(f"  Среднее: {df[column].mean():.4f}")
        print(f"  Минимальное: {df[column].min():.4f}")
        print(f"  Максимальное: {df[column].max():.4f}")
        print()


def filter_and_save_tangent(df):
    condition = (df['x1'] < df['x1'].mean()) | (df['x2'] < df['x2'].mean())
    filtered_df = df[condition]

    filtered_df.to_csv('filtered_tangent_data.csv', index=False)
    print(f"Отфильтрованный файл сохранен. Исходных строк: {len(df)}, отфильтрованных: {len(filtered_df)}")


def plot_tangent_3d_surface(df):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')


    x1_unique = np.linspace(df['x1'].min(), df['x1'].max(), 50)
    x2_unique = np.linspace(df['x2'].min(), df['x2'].max(), 50)
    X1, X2 = np.meshgrid(x1_unique, x2_unique)


    RESULT = np.tan(X1 + X2) * np.sin(X1)


    surf = ax.plot_surface(X1, X2, RESULT, cmap='plasma', alpha=0.8,
                           linewidth=0, antialiased=True)


    scatter = ax.scatter(df['x1'], df['x2'], df['result_value'],
                         c=df['result_value'], cmap='plasma', s=20, alpha=0.6,
                         edgecolors='black', linewidth=0.5)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('result_value')
    ax.set_title('3D поверхность: result_value = tan(x1 + x2) * sin(x1)')


    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='result_value')

    plt.tight_layout()
    plt.savefig('3d_tangent_surface.png', dpi=300)
    plt.show()


def main():
    df = generate_tangent_data()

    df_from_file = pd.read_csv('tangent_data.csv')
    print("Данные из файла:")
    print(df_from_file.head())
    print(f"Общее количество строк: {len(df_from_file)}")

    plot_tangent_graphs(df_from_file)

    print_tangent_statistics(df_from_file)

    filter_and_save_tangent(df_from_file)


    plot_tangent_3d_surface(df_from_file)



if __name__ == "__main__":
    main()