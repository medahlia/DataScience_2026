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
# 1. ПІДГОТОВКА ВХІДНИХ ДАНИХ
# =============================================================================
# 1.1. Парсинг файлу вхідних даних
print('=' * 65)
print('ПУНКТ 1. Парсинг файлів параметрів')
print('=' * 65)

d_sample_data = pd.read_excel(FILE)
Title_d_sample_data = d_sample_data.columns
#only head
print('d_sample_data=', d_sample_data.head())
print('---------  типи даних стовпців DataFrame  -----------')
print(d_sample_data.dtypes)

# 1.2. Парсинг файлу пояснень параметрів
d_data_description = pd.read_excel(FILE_DESC)
print('d_data_description=', d_data_description.head())
print('----------------------------------------------------')

# 1.3. Первинне формування скорингової таблиці — сегментація ознак клієнта та кредиту
d_segment_data_description_client_bank = d_data_description[
    (d_data_description.Place_of_definition == 'Вказує позичальник') |
    (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')]
n_client_bank = d_segment_data_description_client_bank['Place_of_definition'].size
d_segment_data_description_client_bank.index = range(
    0, len(d_segment_data_description_client_bank))

# 1.4. Перевірка наявності індикаторів у sample_data
b = d_segment_data_description_client_bank['Field_in_data']

if set(b).issubset(d_sample_data.columns):
    Flag_b = 'Flag_True'
else:
    Flag_b = 'Flag_False'

n_columns = d_segment_data_description_client_bank['Field_in_data'].size
j = 0
for i in range(0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):
        j = j + 1
print('j=', j)

Columns_Flag_True = np.zeros((j))
j = 0
for i in range(0, n_columns):
    a = d_segment_data_description_client_bank['Field_in_data'][i]
    if set([a]).issubset(d_sample_data.columns):
        Flag = 'Flag_True'
        Columns_Flag_True[j] = i
        j = j + 1
    else:
        Flag = 'Flag_False'

# 1.5. DataFrame співпадінь
d_segment_data_description_client_bank_True = \
    d_segment_data_description_client_bank.iloc[Columns_Flag_True]
d_segment_data_description_client_bank_True.index = range(
    0, len(d_segment_data_description_client_bank_True))


# =============================================================================
# ПУНКТ 2. Вибір індикаторів (10 шт.)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 2. Вибір індикаторів (10)')
print('=' * 65)

INDICATORS = {
    'loan_amount'              : 'min',
    'loan_days'                : 'min',
    'education_id'             : 'max',
    'seniority_years'          : 'max',
    'monthly_income'           : 'max',
    'monthly_expenses'         : 'min',
    'other_loans_about_current': 'min',
    'other_loans_about_monthly': 'min',
    'amount_limit'             : 'max',
    'product_dpr'              : 'min',
}

print(f"\n{'Індикатор':<35} {'Minimax':<8} Обґрунтування")
print('-' * 50)
for field, mm in INDICATORS.items():
    print(f"  {field:<33} {mm:<8}")


# =============================================================================
# ПУНКТ 3. Очищення даних (з лекції scoring.py)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 3. Очищення даних')
print('=' * 65)

cols = list(INDICATORS.keys())

# Формування сегменту вхідних даних за обраними індикаторами
d_segment_sample_cleaning = d_sample_data[cols].copy()
d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))

print('---- пропуски даних сегменту DataFrame --------')
print(d_segment_sample_cleaning.isnull().sum())
print('-----------------------------------------------')

# Вилучення рядків з пропусками
d_segment_sample_cleaning = d_segment_sample_cleaning.dropna()
d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))

d_segment_sample_cleaning.to_excel('d_segment_sample_cleaning.xlsx', index=False)
print('---------- DataFrame вхідних даних -----------')
print(d_segment_sample_cleaning.head())
print('-----------------------------------------------------------------')

# =============================================================================
# СКОРИНГОВА МОДЕЛЬ
# =============================================================================
# ПУНКТ 4. Розрахунок інтегрованої оцінки Scor (метод Вороніна)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 4. Розрахунок інтегрованої оцінки Scor (метод Вороніна)')
print('=' * 65)

# Мінімальні та максимальні значення стовпців (з лекції блок 2.2)
d_segment_sample_min = d_segment_sample_cleaning[cols].min()
d_segment_sample_max = d_segment_sample_cleaning[cols].max()
print('----------------- DataFrame: d_segment_sample_min  ----------------')
print(d_segment_sample_min)
print('----------------- DataFrame: d_segment_sample_max  ----------------')
print(d_segment_sample_max)

# Нормування критеріїв (з лекції scoring.py — блок 2.3)
m = d_segment_sample_cleaning[cols[0]].size
n = len(INDICATORS)
d_segment_sample_minimax_Normal = np.zeros((m, n))

delta_d = 0.3  # коефіцієнт запасу при нормуванні

for j, (field, mm) in enumerate(INDICATORS.items()):
    if mm == 'min':
        for i in range(m):
            max_max = d_segment_sample_max[field] + (2 * delta_d)
            d_segment_sample_minimax_Normal[i, j] = \
                (delta_d + d_segment_sample_cleaning[field][i]) / max_max
    else:
        for i in range(m):
            min_min = d_segment_sample_max[field] + (2 * delta_d)
            d_segment_sample_minimax_Normal[i, j] = \
                (1 / (delta_d + d_segment_sample_cleaning[field][i])) / min_min
np.savetxt('d_segment_sample_minimax_Normal.txt', d_segment_sample_minimax_Normal)

# Інтегрована багатокритеріальна оцінка — SCOR (з лекції scoring.py)
def Voronin(d_segment_sample_minimax_Normal, n, m):
    '''
    Нелінійна схема компромісів Вороніна:
    Integro[i] = sum_j( (1 - F_norm[i,j])^(-1) )
    Чим менше Integro — тим кращий позичальник.
    Scor = поріг прийняття рішення (адаптований до обсягу вибірки).
    '''
    Integro = np.zeros(m)
    Scor = np.zeros(m)
    for i in range(m):
        Sum_Voronin = 0
        for j in range(n):
            Sum_Voronin += (1 - d_segment_sample_minimax_Normal[i, j]) ** (-1)
        Integro[i] = Sum_Voronin
        # Поріг: медіана Integro — аналог Scor=1000 з лекції,
        # адаптований до реального розподілу даних вибірки
        Scor[i] = np.median(Integro[:i + 1])  # обчислюється після циклу нижче
    # Перераховуємо Scor як фіксовану медіану всього масиву
    scor_threshold = np.median(Integro)
    Scor[:] = scor_threshold
    np.savetxt('Integro_Scor.txt', Integro)
    return Integro, Scor

Integro, Scor = Voronin(d_segment_sample_minimax_Normal, n, m)
SCOR_THRESHOLD = Scor[0]

print(f'\nІнтегрована оцінка Integro:')
print(f'  Мін: {Integro.min():.2f}  (найкращий)')
print(f'  Макс: {Integro.max():.2f}  (найгірший)')
print(f'  Середня: {Integro.mean():.2f}')
print(f'  Медіана: {np.median(Integro):.2f}')
print(f'  СКВ: {Integro.std():.2f}')
print(f'  Поріг Scor: {SCOR_THRESHOLD:.2f}')
print(f'  Видати кредит (Integro < Scor): {(Integro < SCOR_THRESHOLD).sum()} позичальників')
print(f'  Відмовити     (Integro ≥ Scor): {(Integro >= SCOR_THRESHOLD).sum()} позичальників')


# =============================================================================
# ПУНКТ 5. Кластеризація позичальників (KMeans)
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 5. Кластеризація позичальників')
print('=' * 65)

# KMeans на нормованому масиві — 2 кластери: видати / відмовити
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(d_segment_sample_minimax_Normal)
labels = kmeans.labels_

# Кластер з меншим середнім Integro = "видати кредит"
mean_0 = Integro[labels == 0].mean()
mean_1 = Integro[labels == 1].mean()
approve_cluster = 0 if mean_0 < mean_1 else 1

cluster_decision = np.where(labels == approve_cluster, 'Видати', 'Відмовити')
n_approve = (cluster_decision == 'Видати').sum()
n_reject = (cluster_decision == 'Відмовити').sum()

print(f'Кластер "Видати кредит" | '
      f'середній Integro: {Integro[labels == approve_cluster].mean():.2f}')
print(f'Кластер "Відмовити"     | '
      f'середній Integro: {Integro[labels != approve_cluster].mean():.2f}')
print(f'\nРезультат: Видати: {n_approve} ({n_approve/m*100:.1f}%),  '
      f'Відмовити: {n_reject} ({n_reject/m*100:.1f}%)')

# Збереження результатів у таблицю
df_result = d_segment_sample_cleaning.copy()
df_result['Integro']  = Integro
df_result['Cluster']  = labels
df_result['Decision'] = cluster_decision
df_result.to_excel('scoring_results.xlsx', index=False)
print('Результати збережено: scoring_results.xlsx')

# =============================================================================
# ПУНКТ 6. Візуалізація результатів (графіки + файли)
# =============================================================================
mask_approve = cluster_decision == 'Видати'
mask_reject = cluster_decision == 'Відмовити'
idx = np.arange(m)

# --- 6.1. Графік Integro_Scor (з лекції: plt.plot(Integro) + plt.plot(Scor)) ---
plt.figure(figsize=(13, 4))
plt.title('Integro_Scor')
plt.scatter(idx[mask_approve], Integro[mask_approve],
            color='steelblue', s=12, alpha=0.7, label='Видати кредит', zorder=3)
plt.scatter(idx[mask_reject],  Integro[mask_reject],
            color='crimson',   s=12, alpha=0.7, label='Відмовити',     zorder=3)
plt.plot(Scor, color='orange', linewidth=2,
         linestyle='--', label=f'Scor = {SCOR_THRESHOLD:.0f}')
plt.xlabel('Індекс позичальника')
plt.ylabel('Integro')
plt.legend()
plt.tight_layout()
plt.savefig('6_1_integro_scor.png', dpi=130)
plt.close()
print('  Збережено: 6_1_integro_scor.png')

# --- 6.2. KMeans: scatter дохід vs сума кредиту ---
plt.figure(figsize=(9, 6))
plt.scatter(d_segment_sample_cleaning.loc[mask_approve, 'monthly_income'],
            d_segment_sample_cleaning.loc[mask_approve, 'loan_amount'],
            color='steelblue', s=20, alpha=0.6, label='Видати кредит')
plt.scatter(d_segment_sample_cleaning.loc[mask_reject,  'monthly_income'],
            d_segment_sample_cleaning.loc[mask_reject,  'loan_amount'],
            color='crimson',   s=20, alpha=0.6, label='Відмовити')
plt.title('KMeans: кластеризація позичальників\n(дохід vs сума кредиту)')
plt.xlabel('Місячний дохід (monthly_income), грн')
plt.ylabel('Сума кредиту (loan_amount), грн')
plt.legend()
plt.tight_layout()
plt.savefig('6_2_kmeans_clusters.png', dpi=130)
plt.close()
print('  Збережено: 6_2_kmeans_clusters.png')