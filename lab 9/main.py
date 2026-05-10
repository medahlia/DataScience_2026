import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


filename = "/Users/dasha/projects/DataScience_2026/lab 9/Fire_Stations/Fire_Stations.shp"
fire_stations = gpd.read_file(filename)
print(type(fire_stations), 'Карта формата *.shp')
print(fire_stations)

#візуалізація всіх станцій
fig, ax = plt.subplots(figsize=(9, 7))
fire_stations.plot(ax=ax, color="blue", markersize=1, alpha=0.4)
ax.set_title("Fire Stations USA", fontsize=16)
plt.tight_layout()
plt.show()

'''
EPSG:4326 WGS 84 -- WGS84 - Всесвітня геодезична система координат 1984р., використовується в GPS - навігації
'''
print('Система координат', fire_stations.crs)

# вибір 10 станцій
SELECTED_IDX = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
sel = fire_stations.iloc[SELECTED_IDX].copy()

selected_stations = []

for i in SELECTED_IDX:
    row = fire_stations.iloc[i]
    # переводимо координати у WGS84
    point_wgs84 = gpd.GeoSeries(
        [row.geometry],
        crs=fire_stations.crs
    ).to_crs('EPSG:4326')

    lon = point_wgs84.iloc[0].x
    lat = point_wgs84.iloc[0].y

    selected_stations.append({
        'idx': i,
        'name': str(row['NAME'])[:28],
        'city': str(row['CITY']),
        'state': str(row['STATE']),
        'x': row.geometry.x,
        'y': row.geometry.y,
        'lon': lon,
        'lat': lat
    })

print(f"\n{'#':<4} {'Назва станції':<30} {'Місто':<20} {'Штат':<6} "
      f"{'Lon':>10} {'Lat':>10}")
print('-' * 90)

for i, s in enumerate(selected_stations):
    print(f"{i:<4} "
          f"{s['name']:<30} "
          f"{s['city']:<20} "
          f"{s['state']:<6} "
          f"{s['lon']:>10.4f} "
          f"{s['lat']:>10.4f}")


# візуалізація обраних станцій
fig, ax = plt.subplots(figsize=(9, 7))
fire_stations.plot(ax=ax, color="blue", markersize=1, alpha=0.4)
sel.plot(ax=ax, color="red", markersize=40)
ax.set_title("Fire Stations (selected = red)", fontsize=16)
plt.tight_layout()
plt.show()

#розрахунок сітки відстаней
n = len(selected_stations)
dist_grid_km = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        p1 = sel.iloc[i].geometry
        p2 = sel.iloc[j].geometry
        dist_grid_km[i, j] = p1.distance(p2) / 1000

print('\nСітка відстаней (км, EPSG:3857):')
names_short = [s['state'] + '_' + s['name'][:8]
for s in selected_stations
]
header = f"{'':>14}" + ''.join(
    f"{n[:8]:>12}" for n in names_short
)
print(header)
print('-' * (14 + 12 * n))

for i in range(n):
    row = f"{names_short[i]:>14}"
    for j in range(n):
        row += f"{dist_grid_km[i,j]:>12.1f}"
    print(row)

#розрахунок середнього, мінімум, максимум
upper_tri = [
    dist_grid_km[i, j]
    for i in range(n)
    for j in range(i + 1, n)
]

avg_dist = np.mean(upper_tri)
min_dist = np.min(upper_tri)
max_dist = np.max(upper_tri)

print(f'\nСередня відстань між станціями: {avg_dist:.1f} км')
print(f'Мінімальна відстань: {min_dist:.1f} км')
print(f'Максимальна відстань: {max_dist:.1f} км')

#теплова карта відстаней
fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(dist_grid_km, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='Відстань, км')
labels_short = [
    f"{s['state']}\n{s['name'][:10]}"
    for s in selected_stations
]
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels_short, fontsize=8, rotation=45, ha='right')
ax.set_yticklabels(labels_short, fontsize=8)

for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{dist_grid_km[i,j]:.0f}', ha='center', va='center',
            fontsize=6,
            color='black'
            if dist_grid_km[i,j] < dist_grid_km.max()*0.6
            else 'white'
        )

ax.set_title('Сітка відстаней між 10 пожежними станціями (км)', fontsize=14)
plt.tight_layout()
plt.show()

distance_m = fire_stations.iloc[0].geometry.distance(fire_stations.iloc[2].geometry)
distance_km = distance_m / 1000
print(f'Відстань у EPSG:3857: {distance_m:.2f} метрів')
print(f'Відстань у кілометрах: {distance_km:.2f} км')