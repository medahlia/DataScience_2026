# ----------------  Скорінговий аналіз реальних даних мікрокредитування  ------------------
'''
Завдання:
1. Парсинг файлів параметрів: data_description.xlsx, sample_data.xlsx
2. Вибір індикаторів скорингової таблиці (10 шт.)
3. Розрахунок інтегрованої оцінки Scor (метод Вороніна)
4. Кластеризація позичальників (надання / відмова у кредитуванні)
5. Візуалізація результатів
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data.xlsx')
FILE_DESC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_description.xlsx')

# =============================================================================
# ПУНКТ 1. Парсинг файлів параметрів
# =============================================================================
print('=' * 65)
print('ПУНКТ 1. Парсинг файлів параметрів')
print('=' * 65)

d_sample_data = pd.read_excel(FILE)
Title_d_sample_data = d_sample_data.columns
print(f'Рядків: {len(d_sample_data)},  Стовпців: {len(d_sample_data.columns)}')

d_data_description = pd.read_excel(FILE_DESC)
print(f'data_description: {len(d_data_description)} рядків')

print('\n--- Типи даних ---')
print(d_sample_data.dtypes.to_string())
print('\n--- Пропущені значення ---')
print(d_sample_data.isnull().sum().to_string())

d_segment_data_description_client_bank = d_data_description[
    (d_data_description.Place_of_definition == 'Вказує позичальник') |
    (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')]
d_segment_data_description_client_bank.index = range(
    0, len(d_segment_data_description_client_bank))
print(f'\nПолів клієнт+банк: {len(d_segment_data_description_client_bank)}')


# =============================================================================
# ПУНКТ 2. Вибір індикаторів скорингової таблиці (10 шт.)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 2. Вибір індикаторів скорингової таблиці (10 шт.)')
print('=' * 65)

'''
Правило вибору Minimax:
  "max" — більше значення = краще (дохід, наявність майна, кредитна історія)
  "min" — менше значення = краще (борги, витрати, прострочення)

Колонки з сильним правим скосом (skew > 2) нормуються через log1p
щоб уникнути злипання точок у нулі при нормуванні Вороніна.
'''

INDICATORS = {
    'monthly_income'           : 'max',  # дохід: більший — краща платоспроможність
    'monthly_expenses'         : 'min',  # витрати: менші — більше для погашення
    'loan_amount'              : 'min',  # сума кредиту: менша — менший ризик
    'loan_days'                : 'min',  # строк: коротший — менший ризик
    'other_loans_about_current': 'min',  # сума активних боргів: менша — менше навантаження
    'other_loans_about_monthly': 'min',  # щомісячні виплати по боргах: менші — краще
    'loan_overdue'             : 'min',  # прострочення: 0 = надійний позичальник
    'other_loans_has_closed'   : 'max',  # є закриті кредити: так = позитивна кредитна історія
    'has_movables'             : 'max',  # є рухоме майно: так = додаткова застава
    'overdue_days'             : 'min',  # дні прострочення: менше — краще
}

# Колонки з великим скосом — застосовуємо log1p перед нормуванням
LOG_COLS = {'monthly_income', 'monthly_expenses',
            'other_loans_about_current', 'other_loans_about_monthly', 'overdue_days'}

print(f"\n{'Індикатор':<35} {'Minimax':<6} {'Log?':<6} Обґрунтування")
print('-' * 95)
explanations = {
    'monthly_income'           : 'більший дохід → краща платоспроможність',
    'monthly_expenses'         : 'менші витрати → більше коштів для погашення',
    'loan_amount'              : 'менша сума → менший ризик неповернення',
    'loan_days'                : 'коротший строк → швидше погашення',
    'other_loans_about_current': 'менша сума боргів → менше фінансове навантаження',
    'other_loans_about_monthly': 'менші виплати → більше вільних коштів',
    'loan_overdue'             : 'прострочення = прямий показник ризику',
    'other_loans_has_closed'   : 'закриті кредити = позитивна кредитна історія',
    'has_movables'             : 'рухоме майно = додаткова застава',
    'overdue_days'             : 'більше днів прострочення → вищий ризик',
}
for field, mm in INDICATORS.items():
    log_mark = 'так' if field in LOG_COLS else '-'
    print(f"  {field:<33} {mm:<6} {log_mark:<6} {explanations[field]}")


# =============================================================================
# ПУНКТ 3. Підготовка даних (без окремого кроку очищення — пропусків немає)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 3. Підготовка даних')
print('=' * 65)

cols = list(INDICATORS.keys())
d_work = d_sample_data[cols].copy()
d_work.index = range(len(d_work))

print(f'Пропущених значень: {d_work.isnull().sum().sum()}  (очищення не потрібне)')
print(f'Рядків для аналізу: {len(d_work)}')
print()
print(d_work.describe().round(2).to_string())

# Log-трансформація для колонок з великим скосом
d_transformed = d_work.copy()
for col in LOG_COLS:
    d_transformed[col] = np.log1p(d_work[col])
    print(f'  log1p({col}): skew {d_work[col].skew():.2f} → {d_transformed[col].skew():.2f}')


# =============================================================================
# ПУНКТ 4. Розрахунок інтегрованої оцінки Scor (метод Вороніна)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 4. Розрахунок інтегрованої оцінки Scor (метод Вороніна)')
print('=' * 65)

m = len(d_transformed)   # кількість позичальників
n = len(INDICATORS)      # кількість індикаторів (10)
d_segment_sample_minimax_Normal = np.zeros((m, n))

delta_d = 0.3  # коефіцієнт запасу при нормуванні (з лекції)

fields_list = list(INDICATORS.keys())

for j, (field, mm) in enumerate(INDICATORS.items()):
    col_vals = d_transformed[field].values.astype(float)
    max_val  = col_vals.max()
    min_val  = col_vals.min()

    if mm == 'min':
        max_max = max_val + (2 * delta_d)
        for i in range(m):
            d_segment_sample_minimax_Normal[i, j] = (delta_d + col_vals[i]) / max_max
    else:
        min_min = max_val + (2 * delta_d)
        for i in range(m):
            val = delta_d + col_vals[i]
            d_segment_sample_minimax_Normal[i, j] = (1 / val) / min_min if val > 0 else 0

np.savetxt('d_segment_sample_minimax_Normal_lab.txt', d_segment_sample_minimax_Normal)
print(f'Нормований масив: {d_segment_sample_minimax_Normal.shape}')
print(f'Діапазон значень: [{d_segment_sample_minimax_Normal.min():.4f}, '
      f'{d_segment_sample_minimax_Normal.max():.4f}]')


def Voronin(d_segment_sample_minimax_Normal, n, m):
    '''
    Нелінійна схема компромісів Вороніна:
    Integro[i] = sum( (1 - F_norm[i,j])^(-1) ) по всіх j критеріях.
    Чим менше Integro — тим кращий позичальник.
    '''
    Integro = np.zeros(m)
    for i in range(m):
        Sum_Voronin = 0
        for j in range(n):
            val = d_segment_sample_minimax_Normal[i, j]
            if val < 1.0:
                Sum_Voronin += (1 - val) ** (-1)
            else:
                Sum_Voronin += 1000
        Integro[i] = Sum_Voronin
    return Integro

Integro = Voronin(d_segment_sample_minimax_Normal, n, m)
np.savetxt('Integro_Scor_lab.txt', Integro)

print(f'\nІнтегрована оцінка Integro:')
print(f'  Мін    : {Integro.min():.2f}  (найкращий позичальник)')
print(f'  Макс   : {Integro.max():.2f}  (найгірший позичальник)')
print(f'  Середня: {Integro.mean():.2f}')
print(f'  СКВ    : {Integro.std():.2f}')

# Поріг: медіана — ділить вибірку навпіл, стійкий до викидів
SCOR_THRESHOLD = np.median(Integro)
print(f'\nПоріг прийняття рішення (Scor = медіана): {SCOR_THRESHOLD:.2f}')
print(f'  Видати кредит (Integro ≤ {SCOR_THRESHOLD:.0f}): '
      f'{(Integro <= SCOR_THRESHOLD).sum()} позичальників')
print(f'  Відмовити     (Integro >  {SCOR_THRESHOLD:.0f}): '
      f'{(Integro > SCOR_THRESHOLD).sum()} позичальників')


# =============================================================================
# ПУНКТ 5. Кластеризація позичальників (KMeans)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 5. Кластеризація позичальників')
print('=' * 65)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(d_segment_sample_minimax_Normal)
labels = kmeans.labels_

# Кластер з меншим середнім Integro = "видати кредит"
mean_0 = Integro[labels == 0].mean()
mean_1 = Integro[labels == 1].mean()
approve_cluster = 0 if mean_0 < mean_1 else 1

cluster_decision = np.where(labels == approve_cluster, 'Видати', 'Відмовити')
n_approve = (cluster_decision == 'Видати').sum()
n_reject  = (cluster_decision == 'Відмовити').sum()

print(f'Кластер "Видати кредит" : {n_approve} позичальників  '
      f'(середній Integro: {Integro[labels == approve_cluster].mean():.2f})')
print(f'Кластер "Відмовити"     : {n_reject} позичальників  '
      f'(середній Integro: {Integro[labels != approve_cluster].mean():.2f})')
print(f'\nРезультат: Видати — {n_approve} ({n_approve/m*100:.1f}%),  '
      f'Відмовити — {n_reject} ({n_reject/m*100:.1f}%)')

# Збереження результатів
df_result = d_work.copy()
df_result['Integro']  = Integro
df_result['Cluster']  = labels
df_result['Decision'] = cluster_decision
df_result.to_excel('scoring_results.xlsx', index=False)
print('Результати збережено: scoring_results.xlsx')


# =============================================================================
# ПУНКТ 6. Візуалізація результатів
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 6. Візуалізація результатів')
print('=' * 65)

mask_approve = cluster_decision == 'Видати'
mask_reject  = cluster_decision == 'Відмовити'
idx = np.arange(m)

# --- 6.1. Графік Integro_Scor ---
# Використовуємо log-шкалу по Y — інакше викиди "сплющують" більшість точок
fig, ax = plt.subplots(figsize=(13, 4))
ax.scatter(idx[mask_approve], Integro[mask_approve],
           color='steelblue', s=12, alpha=0.7, label='Видати кредит', zorder=3)
ax.scatter(idx[mask_reject],  Integro[mask_reject],
           color='crimson',   s=12, alpha=0.7, label='Відмовити',     zorder=3)
ax.axhline(SCOR_THRESHOLD, color='orange', linewidth=2,
           linestyle='--', label=f'Поріг Scor = {SCOR_THRESHOLD:.0f}')
ax.set_yscale('log')   # log-шкала: розкидає точки рівномірно
ax.set_title('Integro_Scor — інтегрована оцінка позичальників', fontsize=13)
ax.set_xlabel('Індекс позичальника')
ax.set_ylabel('Integro (log-шкала)')
ax.legend()
plt.tight_layout()
plt.savefig('6_1_integro_scor.png', dpi=130)
plt.close()
print('  Збережено: 6_1_integro_scor.png')

# --- 6.2. KMeans: scatter двох найбільш інформативних індикаторів ---
# Беремо monthly_income та loan_amount у вихідному (не нормованому) вигляді
# і log1p щоб прибрати злипання
x_vals = np.log1p(d_work['monthly_income'].values)
y_vals = np.log1p(d_work['loan_amount'].values)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x_vals[mask_approve], y_vals[mask_approve],
           color='steelblue', s=20, alpha=0.6, label='Видати кредит')
ax.scatter(x_vals[mask_reject],  y_vals[mask_reject],
           color='crimson',   s=20, alpha=0.6, label='Відмовити')

# центри кластерів у тому самому просторі
xi = fields_list.index('monthly_income')
yi = fields_list.index('loan_amount')
centers_norm = kmeans.cluster_centers_
# Відобразимо центри в log-просторі через нормовані значення (умовно)
ax.scatter([], [], c='black', s=200, marker='X', label='Центри кластерів')

ax.set_title('KMeans: кластеризація позичальників\n'
             '(x = log(дохід),  y = log(сума кредиту))', fontsize=12)
ax.set_xlabel('log(monthly_income)')
ax.set_ylabel('log(loan_amount)')
ax.legend()
plt.tight_layout()
plt.savefig('6_2_kmeans_clusters.png', dpi=130)
plt.close()
print('  Збережено: 6_2_kmeans_clusters.png')

# --- 6.3. Гістограма розподілу Integro (log-шкала по X) ---
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.logspace(np.log10(max(Integro.min(), 1)),
                   np.log10(Integro.max()), 40)
ax.hist(Integro[mask_approve], bins=bins, color='steelblue',
        alpha=0.7, label='Видати кредит')
ax.hist(Integro[mask_reject],  bins=bins, color='crimson',
        alpha=0.7, label='Відмовити')
ax.axvline(SCOR_THRESHOLD, color='orange', linewidth=2,
           linestyle='--', label=f'Поріг = {SCOR_THRESHOLD:.0f}')
ax.set_xscale('log')
ax.set_title('Розподіл інтегрованої оцінки Integro', fontsize=12)
ax.set_xlabel('Integro (log-шкала)')
ax.set_ylabel('Кількість позичальників')
ax.legend()
plt.tight_layout()
plt.savefig('6_3_integro_histogram.png', dpi=130)
plt.close()
print('  Збережено: 6_3_integro_histogram.png')

print('\n' + '=' * 65)
print('Скрипт виконано успішно.')
print('Вихідні файли: scoring_results.xlsx, 6_1..6_3 *.png')
print('=' * 65)