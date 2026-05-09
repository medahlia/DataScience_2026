import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt


def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=130)
    plt.close()
    print(f'  Збережено: {name}')

# =============================================================================
# ПУНКТ 1. Парсинг файлу Pr_1.xls
# =============================================================================
print('=' * 65)
print('ПУНКТ 1. Парсинг файлу Pr_1.xls')
print('=' * 65)

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Pr_1.xls')
d = pd.read_excel(FILE, engine='xlrd', parse_dates=['Дата'])
dd = pd.read_excel(FILE, engine='xlrd', parse_dates=['Дата'], index_col='Дата')

d = d[d['Місяц'] != 'Стоп'].reset_index(drop=True)
dd = dd[dd['Місяц'] != 'Стоп']

print(f'  Рядків: {len(d)}')
print(f'  Місяці: {d["Місяц"].unique().tolist()}')
print(d.head(3).to_string())

# =============================================================================
# ПУНКТ 2. Попередній аналіз даних
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 2. Попередній аналіз даних')
print('=' * 65)

index = 'Ціна реализації'

print('\n--- 2.1 Описова статистика ---')
print(d[['КільКість реалізацій', 'Собівартість одиниці', 'Ціна реализації']].describe().to_string())

# Візуалізація ціни реалізації по часу
plt.figure(figsize=(12, 4))
plt.title(f'{index} — весь рік')
d[index].plot()
plt.xlabel('Індекс запису'); plt.ylabel('Ціна, грн')
savefig('2.1_price_index.png')

MONTHS = ['Січень','Лютий','Березень','Квітень','Травень','Червень',
          'Липень','Серпень','Вересень','Жовтень','Листопад','Грудень']

def segment_by_month(d, col):
    result = {}
    for m in MONTHS:
        mask = d['Місяц'] == m
        result[m] = d.loc[mask, col].values.astype(float)
    return result

segments = segment_by_month(d, index)
print('\n--- 2.2 Кількість записів по місяцях ---')
for m, arr in segments.items():
    print(f'  {m}: {len(arr)}')

plt.figure(figsize=(10, 4))
plt.title('Ціна реализації — місяці 1–6')
for m in MONTHS[:6]:
    plt.plot(segments[m], label=m)
plt.legend(fontsize=8); plt.xlabel('Індекс'); plt.ylabel('Ціна, грн')
savefig('2.2_price_month.png')

# =============================================================================
# ПУНКТ 3. Показники ефективності: продажі та прибуток
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 3. Показники ефективності: продажі та прибуток')
print('=' * 65)

def sale(d):
    n = d['КільКість реалізацій'].size
    F_sale = np.zeros(n)
    for i in range(n):
        F_sale[i] = d['КільКість реалізацій'][i] * d['Собівартість одиниці'][i]
    return F_sale

def profit(d):
    n = d['КільКість реалізацій'].size
    F_profit = np.zeros(n)
    for i in range(n):
        F_profit[i] = (d['КільКість реалізацій'][i] *
                       (d['Ціна реализації'][i] - d['Собівартість одиниці'][i]))
    return F_profit

def sum_by_month(d, F_in):
    F_month = np.zeros(12)
    for idx, m in enumerate(MONTHS):
        mask = (d['Місяц'] == m).values
        F_month[idx] = np.sum(F_in[mask])
    return F_month

F_sale = sale(d)
F_profit = profit(d)

F_month_sale = sum_by_month(d, F_sale)
F_month_profit = sum_by_month(d, F_profit)

# Зведена таблиця
df_eff = pd.DataFrame({
    'Місяць':            MONTHS,
    'Продажі':           F_month_sale,
    'Прибуток':          F_month_profit,
    'Рентабельність %':  np.where(
        F_month_sale > 0,
        (F_month_profit / F_month_sale) * 100,
        0
    )
})
print('\n--- 3.1 Зведена таблиця ефективності ---')
print(df_eff.to_string(index=False, float_format='%.2f'))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
s1 = pd.Series(F_month_sale,   index=MONTHS)
s2 = pd.Series(F_month_profit, index=MONTHS)
s1.plot(kind='bar', ax=axes[0], color='steelblue', title='Продажі по місяцях')
s2.plot(kind='bar', ax=axes[1], color='green',     title='Прибуток по місяцях')
for ax in axes:
    ax.set_xlabel('Місяць'); ax.set_ylabel('Грн')
    ax.tick_params(axis='x', rotation=45)
savefig('3.1_sale_profit_month.png')

# =============================================================================
# ПУНКТ 4. МНК: математична модель
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 4. МНК: математична модель')
print('=' * 65)

def MNK_Stat_characteristics(S0):
    iter_ = len(S0)
    Yin = np.zeros((iter_, 1))
    F = np.ones((iter_, 3))
    for i in range(iter_):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    return F.dot(C)

def Stat_characteristics(SL, Text):
    Yout = MNK_Stat_characteristics(SL)
    SL0 = np.array([SL[i] - Yout[i, 0] for i in range(len(Yout))])
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    mean_val = np.mean(np.abs(SL))
    quality = (1 - scvS / mean_val) * 100 if mean_val > 0 else 0
    print(f'  {Text}:  МС={mS:.1f}  СКВ={scvS:.1f}  якість≈{quality:.1f}%')

def MNK(Y_coord):
    iter_ = Y_coord.size
    Yin = np.zeros((iter_, 1))
    F = np.ones((iter_, 5))
    for i in range(iter_):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i);   F[i, 2] = float(i**2)
        F[i, 3] = float(i**3); F[i, 4] = float(i**4)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    return F.dot(C)

def MNK_extrapolation(Y_coord, koef):
    iter_ = Y_coord.size
    Yin = np.zeros((iter_, 1))
    F = np.ones((iter_, 5))
    for i in range(iter_):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i);   F[i, 2] = float(i**2)
        F[i, 3] = float(i**3); F[i, 4] = float(i**4)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    j = koef
    for i in range(koef):
        Yout[i, 0] = (C[0,0] + C[1,0]*j + C[2,0]*j**2
                      + C[3,0]*j**3 + C[4,0]*j**4)
        j += 1
    return Yout

# --- 4.1 МНК для продажів по місяцях ---
print('\n--- 4.1 МНК для продажів по місяцях ---')
Yout_sale_month = MNK(F_month_sale)
Stat_characteristics(F_month_sale,          'вхідна вибірка')
Stat_characteristics(Yout_sale_month[:, 0], 'МНК оцінка')

plt.figure(figsize=(10, 4))
plt.title('МНК: Продажі по місяцях')
plt.plot(F_month_sale,          label='Реальні дані', marker='o')
plt.plot(Yout_sale_month[:, 0], label='МНК тренд', lw=2, color='red')  # суцільна лінія
plt.xticks(range(12), MONTHS, rotation=45, ha='right')
plt.legend(); plt.ylabel('Продажі, грн')
savefig('4.1_mnk_sale_month.png')

print('\n--- 4.2 МНК для прибутку по місяцях ---')
Yout_profit_month = MNK(F_month_profit)
Stat_characteristics(F_month_profit,           'вхідна вибірка')
Stat_characteristics(Yout_profit_month[:, 0],  'МНК оцінка')

plt.figure(figsize=(10, 4))
plt.title('МНК: Прибуток по місяцях')
plt.plot(F_month_profit,          label='Реальні дані', marker='o')
plt.plot(Yout_profit_month[:, 0], label='МНК тренд', lw=2, color='red')  # суцільна лінія
plt.xticks(range(12), MONTHS, rotation=45, ha='right')
plt.legend(); plt.ylabel('Прибуток, грн')
savefig('4.2_mnk_profit_month.png')

# =============================================================================
# ПУНКТ 5. Динаміка продажів за регіонами
# =============================================================================
print('\n' + '=' * 65)
print('ПУНКТ 5. Динаміка продажів за регіонами')
print('=' * 65)

d_work = d.copy()
d_work['Продажі'] = F_sale
d_work['Прибуток'] = F_profit


lookup_df = d_work[d_work['Регіон'].notna()][['Код магазину.1','Регіон']].drop_duplicates()
store_region_map = dict(zip(lookup_df['Код магазину.1'].astype(int), lookup_df['Регіон']))
d_work['Регіон'] = d_work['Код магазину'].map(store_region_map)
d_region = d_work.dropna(subset=['Регіон']).copy()
regions = sorted(d_region['Регіон'].unique().tolist())
print(f'  Регіони: {regions}  |  Транзакцій: {len(d_region)}')

# Зведені таблиці по регіонах і місяцях
pivot_sale = d_region.pivot_table(values='Продажі',  index='Місяц', columns='Регіон', aggfunc='sum')
pivot_profit = d_region.pivot_table(values='Прибуток', index='Місяц', columns='Регіон', aggfunc='sum')
pivot_sale = pivot_sale.reindex([m for m in MONTHS if m in pivot_sale.index])
pivot_profit = pivot_profit.reindex([m for m in MONTHS if m in pivot_profit.index])

print('\n--- 5.1 Продажі по регіонах і місяцях (грн) ---')
print(pivot_sale.to_string(float_format='%.0f'))

print('\n--- 5.2 Прибуток по регіонах і місяцях (грн) ---')
print(pivot_profit.to_string(float_format='%.0f'))

# Підсумкова таблиця по регіонах
region_total = d_region.groupby('Регіон').agg(
    Продажі_всього=('Продажі',  'sum'),
    Прибуток_всього=('Прибуток', 'sum'),
    Транзакцій=('Продажі',      'count')
).reset_index()
region_total['Рентабельність %'] = (
    region_total['Прибуток_всього'] / region_total['Продажі_всього'] * 100
)
print('\n--- 5.3 Підсумкова таблиця по регіонах ---')
print(region_total.to_string(index=False, float_format='%.2f'))

# динаміка продажів по регіонах і місяцях
plt.figure(figsize=(13, 5))
for col in pivot_sale.columns:
    plt.plot(range(len(pivot_sale)), pivot_sale[col].values, marker='o', lw=1.8, label=col)
plt.xticks(range(len(pivot_sale)), pivot_sale.index, rotation=45, ha='right')
plt.title('Динаміка продажів по регіонах і місяцях')
plt.xlabel('Місяць'); plt.ylabel('Продажі, грн')
plt.legend(fontsize=9)
savefig('5.1_region_sale_month.png')

print('\n--- 5.4 МНК-тренди по регіонах ---')
plt.figure(figsize=(13, 5))
for region in regions:
    arr = pivot_sale[region].values.astype(float)
    if len(arr) < 3:
        continue
    Yout_r = MNK(arr)
    plt.plot(range(len(arr)), Yout_r[:, 0], lw=2, label=f'{region}')  # суцільна лінія
    Stat_characteristics(arr, region)

plt.xticks(range(len(pivot_sale)), pivot_sale.index, rotation=45, ha='right')
plt.title('МНК-тренди продажів по регіонах')
plt.xlabel('Місяць'); plt.ylabel('Продажі, грн')
plt.legend(fontsize=9)
savefig('5.2_region_mnk_trends.png')