# ========================= Decision Support System (DSS) =========================
"""
Завдання: Оцінювання ефективності впровадження нового товару на ринок
         засобами багатокритеріальної оптимізації (метод Вороніна).

Тип товару : Смартфони (10 моделей)
Критерії   : 10 показників — 8 мінімізованих, 2 максимізованих
Метод      : Нелінійна схема компромісів (Voronin Integrated Score)
             http://sci-gems.math.bas.bg/jspui/bitstream/10525/49/1/ijita15-2-p02.pdf

Вхідні дані: файл Pr1_new.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ───────────────────────────── 1. Завантаження даних ─────────────────────────────

FILE_NAME = "Pr1_new.xlsx"

def load_data(file_name: str):
    """
    Читає аркуш 'Дані' (рядки 3…12 — рядок 1 заголовки, рядок 2 — тип критерію).
    Повертає: DataFrame з даними, список назв товарів, матрицю критеріїв (numpy).
    """
    df = pd.read_excel(file_name, sheet_name="Дані", header=0)
    # рядок 0 — заголовки (вже прочитані як header), рядок 1 (iloc[0]) — тип критерію
    df_data = df.iloc[1:].reset_index(drop=True)  # пропустити рядок "тип"
    products = df_data.iloc[:, 0].tolist()
    criteria_matrix = df_data.iloc[:, 1:].astype(float).values  # shape (10, 10)
    criteria_names = df.columns[1:].tolist()
    return products, criteria_names, criteria_matrix


# ───────────────────────── 2. Нормалізація та інтегрована оцінка ─────────────────

def voronin_score(
    matrix: np.ndarray,
    maximize_idx: list[int],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Розраховує інтегровану оцінку Вороніна для кожного товару (стовпця).

    Параметри
    ---------
    matrix       : (n_criteria × n_alternatives) — значення критеріїв
    maximize_idx : індекси рядків, які МАКСИМІЗУЮТЬСЯ
    weights      : вектор вагових коефіцієнтів (len = n_criteria)

    Повертає вектор Integro (n_alternatives,) — менше = краще.
    """
    n_crit, n_alt = matrix.shape
    norm_weights = weights / weights.sum()

    normalized = np.zeros_like(matrix, dtype=float)
    for i in range(n_crit):
        row = matrix[i].copy()
        if i in maximize_idx:
            # максимізований критерій → інвертуємо для нормування
            col_sum = np.sum(1.0 / row)
            normalized[i] = (1.0 / row) / col_sum
        else:
            col_sum = np.sum(row)
            normalized[i] = row / col_sum

    # Нелінійна схема компромісів: Integro_j = Σ [ g_i * (1 - f_ij)^(-1) ]
    integro = np.zeros(n_alt)
    for j in range(n_alt):
        score = 0.0
        for i in range(n_crit):
            f_ij = normalized[i, j]
            # Захист від f_ij >= 1 (теоретично неможливо при >1 альтернативи)
            if f_ij >= 1.0:
                f_ij = 0.999
            score += norm_weights[i] * (1.0 - f_ij) ** (-1)
        integro[j] = score
    return integro


# ────────────────────────────── 3. Аналіз та виведення ───────────────────────────

def rank_alternatives(integro: np.ndarray, products: list[str]) -> pd.DataFrame:
    """Повертає таблицю рейтингу — відсортовану за зростанням Integro."""
    df = pd.DataFrame({
        "Товар":       products,
        "Integro":     np.round(integro, 4),
        "Ранг":        pd.Series(integro).rank(method="min").astype(int),
    }).sort_values("Integro").reset_index(drop=True)
    df.index = df.index + 1
    return df


def print_report(products, criteria_names, matrix, maximize_idx,
                 integro, ranking_df):
    n_crit, n_alt = matrix.shape
    sep = "─" * 70

    print("\n" + sep)
    print("  DSS — ОЦІНЮВАННЯ ЕФЕКТИВНОСТІ ВПРОВАДЖЕННЯ НОВОГО ТОВАРУ")
    print(sep)

    print("\n📋  ВХІДНА МАТРИЦЯ КРИТЕРІЇВ\n")
    header = f"{'Критерій':<28}" + "".join(f"{p[:10]:>11}" for p in products)
    print(header)
    print("─" * len(header))
    for i, cname in enumerate(criteria_names):
        tag = "↑макс" if i in maximize_idx else "↓мін "
        row_str = f"{cname[:26]:<26} {tag}" + "".join(f"{matrix[i, j]:>11.1f}" for j in range(n_alt))
        print(row_str)

    print(f"\n{sep}")
    print("  ІНТЕГРОВАНА ОЦІНКА (метод Вороніна) — менше = краще")
    print(sep)
    for j, p in enumerate(products):
        marker = " ◄ ОПТИМАЛЬНИЙ" if integro[j] == integro.min() else ""
        print(f"  {p:<22}  Integro = {integro[j]:8.4f}{marker}")

    print(f"\n{sep}")
    print("  РЕЙТИНГ АЛЬТЕРНАТИВ")
    print(sep)
    print(ranking_df.to_string())

    best_idx = int(np.argmin(integro))
    print(f"\n✅  РЕКОМЕНДАЦІЯ: впровадити товар  «{products[best_idx]}»")
    print(f"    Integro = {integro[best_idx]:.4f}  (мінімальна інтегрована оцінка)\n")


# ──────────────────────────────── 4. Візуалізація ────────────────────────────────

def plot_results(products, integro, criteria_names, matrix, maximize_idx):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("DSS — Оцінювання ефективності впровадження нового товару\n"
                 "(тип товару: смартфони)", fontsize=13, fontweight="bold")

    # --- графік 1: стовпчаста діаграма Integro ---
    ax1 = axes[0]
    colors = ["#2ecc71" if v == integro.min() else
              "#e74c3c" if v == integro.max() else "#3498db"
              for v in integro]
    bars = ax1.bar(range(len(products)), integro, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(range(len(products)))
    ax1.set_xticklabels([p.replace(" ", "\n") for p in products], fontsize=7.5)
    ax1.set_ylabel("Integro (менше = краще)", fontsize=10)
    ax1.set_title("Інтегрована оцінка Вороніна", fontsize=11)
    ax1.axhline(integro.min(), color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.7)
    for bar, val in zip(bars, integro):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    green_patch = mpatches.Patch(color="#2ecc71", label="Оптимальний")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Найгірший")
    blue_patch  = mpatches.Patch(color="#3498db", label="Інші")
    ax1.legend(handles=[green_patch, blue_patch, red_patch], fontsize=8)

    # --- графік 2: нормалізована теплова карта критеріїв ---
    ax2 = axes[1]
    # нормалізація для візуалізації (0..1, де 0=добре)
    norm_vis = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        mn, mx = matrix[i].min(), matrix[i].max()
        if mx == mn:
            norm_vis[i] = 0.5
        elif i in maximize_idx:
            norm_vis[i] = (mx - matrix[i]) / (mx - mn)  # інвертуємо
        else:
            norm_vis[i] = (matrix[i] - mn) / (mx - mn)

    im = ax2.imshow(norm_vis, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(products)))
    ax2.set_xticklabels([p.split()[0] for p in products], rotation=45, ha="right", fontsize=8)
    short_names = [c.split("(")[0].strip() for c in criteria_names]
    ax2.set_yticks(range(len(criteria_names)))
    ax2.set_yticklabels(
        [f"{'↑' if i in maximize_idx else '↓'} {short_names[i]}" for i in range(len(criteria_names))],
        fontsize=8
    )
    ax2.set_title("Теплова карта критеріїв\n(зелений = краще)", fontsize=11)
    plt.colorbar(im, ax=ax2, label="0=добре  1=погано")

    plt.tight_layout()
    plt.savefig("/home/claude/dss_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Графік збережено: dss_results.png")


# ──────────────────────────────── ГОЛОВНИЙ БЛОК ──────────────────────────────────

if __name__ == "__main__":

    # ── завантаження даних ──
    products, criteria_names, raw_matrix = load_data(FILE_NAME)

    # matrix потрібна як (n_criteria × n_alternatives)
    matrix = raw_matrix.T   # shape (10 criteria, 10 alternatives)

    # ── визначення типів критеріїв ──
    # Індекси рядків (критеріїв) що МАКСИМІЗУЮТЬСЯ:
    #   8 → "Акумулятор (мАг)"
    #   9 → "Рейтинг (1-10)"
    MAXIMIZE_IDX = [8, 9]

    # ── вагові коефіцієнти (рівні за замовчуванням, можна змінити) ──
    # Формат: [Ціна, Вага, Товщина, Зарядка, Нагрів, Пам'ять ОС, Ремонт, Шум, Акумулятор, Рейтинг]
    weights = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 0.8, 1.2, 1.8])
    # (Ціна та Рейтинг мають дещо більшу вагу — важливіші для ринку)

    # ── розрахунок інтегрованої оцінки ──
    integro = voronin_score(matrix, MAXIMIZE_IDX, weights)

    # ── рейтинг ──
    ranking_df = rank_alternatives(integro, products)

    # ── звіт ──
    print_report(products, criteria_names, matrix, MAXIMIZE_IDX, integro, ranking_df)

    # ── візуалізація ──
    plot_results(products, integro, criteria_names, matrix, MAXIMIZE_IDX)