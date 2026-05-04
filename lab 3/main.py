import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def file_parsing(Data_name, sample_data):
    for name, values in sample_data[[Data_name]].items():
        values
    n_sample_data = int(len(values))
    S_real = np.zeros((n_sample_data))
    for i in range(n_sample_data):
        S_real[i] = float(values[i])
    return S_real


def matrix_generation(File_name):
    sample_data = pd.read_excel(File_name, header=None)
    print(sample_data)

    data_block = sample_data.iloc[1:, 1:11].reset_index(drop=True)
    data_block.columns = range(10)

    n_criteria = data_block.shape[0]
    n_items = data_block.shape[1]

    line_column_matrix = np.zeros((n_criteria, n_items))
    for i in range(n_criteria):
        for j in range(n_items):
            line_column_matrix[i, j] = float(data_block.iloc[i, j])

    return line_column_matrix


def matrix_adapter(line_column_matrix, line):
    column_sample_matrix = np.shape(line_column_matrix)
    line_matrix = np.zeros((column_sample_matrix[1]))
    for j in range(column_sample_matrix[1]):
        line_matrix[j] = line_column_matrix[line, j]
    return line_matrix


def Voronin(File_name, G1, G2, G3,
            G4, G5, G6, G7,
            G8, G9, G10):

    line_column_matrix = matrix_generation(File_name)
    column_matrix = np.shape(line_column_matrix)
    Integro = np.zeros((column_matrix[1]))

    F1 = matrix_adapter(line_column_matrix, 0)
    F2 = matrix_adapter(line_column_matrix, 1)
    F3 = matrix_adapter(line_column_matrix, 2)
    F4 = matrix_adapter(line_column_matrix, 3)
    F5 = matrix_adapter(line_column_matrix, 4)
    F6 = matrix_adapter(line_column_matrix, 5)
    F7 = matrix_adapter(line_column_matrix, 6)
    F8 = matrix_adapter(line_column_matrix, 7)
    F9 = matrix_adapter(line_column_matrix, 8)
    F10 = matrix_adapter(line_column_matrix, 9)

    F10n = np.zeros((column_matrix[1]))
    F20n = np.zeros((column_matrix[1]))
    F30n = np.zeros((column_matrix[1]))
    F40n = np.zeros((column_matrix[1]))
    F50n = np.zeros((column_matrix[1]))
    F60n = np.zeros((column_matrix[1]))
    F70n = np.zeros((column_matrix[1]))
    F80n = np.zeros((column_matrix[1]))
    F90n = np.zeros((column_matrix[1]))
    F100n = np.zeros((column_matrix[1]))

    GNorm = G1 + G2 + G3 + G4 + G5 + G6 + G7 + G8 + G9 + G10
    G10n = G1 / GNorm
    G20n = G2 / GNorm
    G30n = G3 / GNorm
    G40n = G4 / GNorm
    G50n = G5 / GNorm
    G60n = G6 / GNorm
    G70n = G7 / GNorm
    G80n = G8 / GNorm
    G90n = G9 / GNorm
    G100n = G10 / GNorm

    sum_F1 = sum_F2 = sum_F3 = sum_F4 = sum_F5 = 0
    sum_F6 = sum_F7 = sum_F8 = sum_F9 = sum_F10 = 0

    for i in range(column_matrix[1]):
        sum_F1 += F1[i]
        sum_F2 += F2[i]
        sum_F3 += F3[i]
        sum_F4 += F4[i]
        sum_F5 += F5[i]
        sum_F6 += F6[i]
        sum_F7 += F7[i]
        sum_F8 += F8[i]
        sum_F9 += (1 / F9[i])
        sum_F10 += (1 / F10[i])

    for i in range(column_matrix[1]):
        F10n[i] = F1[i] / sum_F1
        F20n[i] = F2[i] / sum_F2
        F30n[i] = F3[i] / sum_F3
        F40n[i] = F4[i] / sum_F4
        F50n[i] = F5[i] / sum_F5
        F60n[i] = F6[i] / sum_F6
        F70n[i] = F7[i] / sum_F7
        F80n[i] = F8[i] / sum_F8
        F90n[i] = (1 / F9[i]) / sum_F9
        F100n[i] = (1 / F10[i]) / sum_F10

        Integro[i] = (G10n * (1 - F10n[i]) ** (-1)) \
                   + (G20n * (1 - F20n[i]) ** (-1)) \
                   + (G30n * (1 - F30n[i]) ** (-1)) \
                   + (G40n * (1 - F40n[i]) ** (-1)) \
                   + (G50n * (1 - F50n[i]) ** (-1)) \
                   + (G60n * (1 - F60n[i]) ** (-1)) \
                   + (G70n * (1 - F70n[i]) ** (-1)) \
                   + (G80n * (1 - F80n[i]) ** (-1)) \
                   + (G90n * (1 - F90n[i]) ** (-1)) \
                   + (G100n * (1 - F100n[i]) ** (-1))

    min_val = 10000
    opt = 0
    for i in range(column_matrix[1]):
        if min_val > Integro[i]:
            min_val = Integro[i]
            opt = i

    print('Інтегрована оцінка (scor):')
    print(Integro)
    print('Номер_оптимального_товару:', opt + 1)

    return Integro


def print_report(products, criteria_names, matrix, maximize_idx,
                 integro, ranking_df):
    n_crit, n_alt = matrix.shape
    sep = "─" * 70

    print("\n" + sep)
    print("DSS — ОЦІНЮВАННЯ ЕФЕКТИВНОСТІ ВПРОВАДЖЕННЯ НОВОГО ТОВАРУ")
    print(sep)

    print("\nВХІДНА МАТРИЦЯ КРИТЕРІЇВ\n")
    header = f"{'Критерій':<28}" + "".join(f"{p[:10]:>11}" for p in products)
    print(header)
    print("─" * len(header))
    for i, cname in enumerate(criteria_names):
        tag = "↑макс" if i in maximize_idx else "↓мін "
        row_str = f"{cname[:26]:<26} {tag}" + "".join(f"{matrix[i, j]:>11.1f}" for j in range(n_alt))
        print(row_str)

    print(f"\n{sep}")
    print("ІНТЕГРОВАНА ОЦІНКА (метод Вороніна) — менше = краще")
    print(sep)
    for j, p in enumerate(products):
        marker = " ◄ ОПТИМАЛЬНИЙ" if integro[j] == integro.min() else ""
        print(f"  {p:<22}  Integro = {integro[j]:8.4f}{marker}")

    print(f"\n{sep}")
    print("РЕЙТИНГ АЛЬТЕРНАТИВ")
    print(sep)
    print(ranking_df.to_string())

    best_idx = int(np.argmin(integro))
    print(f"\nРЕКОМЕНДАЦІЯ: впровадити товар  «{products[best_idx]}»")
    print(f"Integro = {integro[best_idx]:.4f}  (мінімальна інтегрована оцінка)\n")


def plot_results(products, criteria_names, matrix, maximize_idx):

    plt.figure(figsize=(10, 6))
    plt.title("DSS — Теплова карта критеріїв", fontsize=12, fontweight="bold")

    # нормалізація (0..1, де 0 = добре)
    norm_vis = np.zeros_like(matrix, dtype=float)

    for i in range(matrix.shape[0]):
        mn, mx = matrix[i].min(), matrix[i].max()

        if mx == mn:
            norm_vis[i] = 0.5
        elif i in maximize_idx:
            norm_vis[i] = (mx - matrix[i]) / (mx - mn)
        else:
            norm_vis[i] = (matrix[i] - mn) / (mx - mn)

    im = plt.imshow(norm_vis, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    plt.xticks(range(len(products)),
               [p.split()[0] for p in products],
               rotation=45)

    short_names = [c.split("(")[0].strip() for c in criteria_names]

    plt.yticks(range(len(criteria_names)),
               [f"{'↑' if i in maximize_idx else '↓'} {short_names[i]}"
                for i in range(len(criteria_names))])

    plt.colorbar(im, label="0 = добре, 1 = погано")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    FILE_NAME = "Book1.xlsx"

    products = [f"Товар{i + 1}" for i in range(10)]

    criteria_names = [
        "Критерій 1", "Критерій 2", "Критерій 3", "Критерій 4",
        "Критерій 5", "Критерій 6", "Критерій 7",
        "Критерій 8", "Критерій 9", "Критерій 10"
    ]

    matrix = matrix_generation(FILE_NAME)

    MAXIMIZE_IDX = [8, 9]

    weights = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 0.8, 1.2, 1.8])

    integro = Voronin(FILE_NAME,
                      weights[0], weights[1], weights[2], weights[3], weights[4],
                      weights[5], weights[6], weights[7], weights[8], weights[9])

    ranking_df = pd.DataFrame({
        "Product": products,
        "Integro": integro
    }).sort_values(by="Integro")

    print_report(products, criteria_names, matrix,
                 MAXIMIZE_IDX, integro, ranking_df)

    plot_results(products, criteria_names, matrix, MAXIMIZE_IDX)