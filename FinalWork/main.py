import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


filename = "/Users/dasha/projects/DataScience_2026/lab 9/Fire_Stations/Fire_Stations.shp"
fire_stations = gpd.read_file(filename)

# всі станції
fig, ax = plt.subplots(figsize=(10, 7))
fire_stations.plot(ax=ax, color="blue", markersize=1, alpha=0.4)
ax.set_title("Fire Stations USA (EPSG:3857)", fontsize=16)
plt.tight_layout()
plt.show()

print('\nПоточна система координат:')
print(fire_stations.crs)

SELECTED_IDX = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
sel_3857 = fire_stations.iloc[SELECTED_IDX].copy()

sel_4326 = sel_3857.to_crs('EPSG:4326')

print('\nНова система координат:')
print(sel_4326.crs)

print('\n' + '=' * 95)
print('Аналзі зміни координат після перетворення CRS')
print('=' * 95)

table = []

for i in range(len(sel_3857)):
    row_3857 = sel_3857.iloc[i]
    row_4326 = sel_4326.iloc[i]

    table.append([
        SELECTED_IDX[i],
        str(row_3857['NAME'])[:20],
        str(row_3857['STATE']),
        round(row_3857.geometry.x, 2),
        round(row_3857.geometry.y, 2),
        round(row_4326.geometry.x, 6),
        round(row_4326.geometry.y, 6)
    ])

df = pd.DataFrame(table,
    columns=[
        'IDX',
        'NAME',
        'STATE',
        'X_3857',
        'Y_3857',
        'LON_4326',
        'LAT_4326'
    ]
)

print(df.to_string(index=False))

# візуалізація обраних станцій
fig, ax = plt.subplots(figsize=(9, 7))
fire_stations.plot(ax=ax, color="blue", markersize=1, alpha=0.4)
sel_3857.plot(ax=ax, color="red", markersize=40)
ax.set_title("Fire Stations (selected = red)", fontsize=16)
plt.tight_layout()
plt.show()

# порівняння
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# ------------------ EPSG:3857 ------------------
sel_3857.plot(ax=axes[0], color='blue', markersize=45)
axes[0].set_title('EPSG:3857 (метри)', fontsize=14)
# ------------------ EPSG:4326 ------------------
sel_4326.plot(ax=axes[1], color='green', markersize=45)
axes[1].set_title('EPSG:4326 (градуси)', fontsize=14)
plt.tight_layout()
plt.show()

print('\n' + '=' * 95)
# ---------------- EPSG:3857 ----------------
distance_m = sel_3857.iloc[0].geometry.distance(sel_3857.iloc[1].geometry)
distance_km = distance_m / 1000
print('\nEPSG:3857')
print(f'Відстань: {distance_m:.2f} м')
print(f'Відстань: {distance_km:.2f} км')
# ---------------- EPSG:4326 ----------------
distance_deg = sel_4326.iloc[0].geometry.distance(sel_4326.iloc[1].geometry)
print('\nEPSG:4326')
print(f'Відстань: {distance_deg:.6f} градусів')