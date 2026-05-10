#-------------------- Приклади можливостей та використання методів geopandas ---------------------------

'''
Розрахунок відмтані між двома пожежними станціями за даними від:
https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::fire-stations/explore

Цей набір даних містить точкові характеристики, що представляють розташування будівель пожежних станцій у
Сполучених Штатах, окрузі Колумбія, Пуерто-Ріко та Американських Віргінських островах.
Метою цієї колекції є зображення місць розташування пожежних станцій на картографічних виробах загального призначення.


Посібники / приклади
https://automating-gis-processes.github.io/2016/Lesson2-geopandas-basics.html
https://github.com/jorisvandenbossche/geopandas-tutorial
https://github.com/emiliom/geopandas-tutorial-maptime/blob/master/notebooks/geopandas_intro.ipynb
https://github.com/geopandas/geopandas/blob/main/doc/source/getting_started/introduction.ipynb

Геодані:
https://github.com/geopandas/geodatasets
Open Street Map: https://www.openstreetmap.org/
https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::fire-stations/
https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::fire-stations/explore

'''

import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#------------ парсінг фашлу карти *.shp - формату -------------
filename = "/Users/dasha/projects/DataScience_2026/lab 9/Fire_Stations/Fire_Stations.shp"
fire_stations = gpd.read_file(filename)
print(type(fire_stations), 'Карта формата *.shp')
print(fire_stations)

# візуалізація
fire_stations.plot()
plt.show()

distance = fire_stations.iloc[0].geometry.distance(fire_stations.iloc[2].geometry)
print(distance)

fire_stations = fire_stations.to_crs('EPSG:4326')
distance = fire_stations.iloc[0].geometry.distance(fire_stations.iloc[2].geometry) / 1000
print(distance)


#-------------------- система координат -----------------------
'''
EPSG:4326 WGS 84 -- WGS84 - Всесвітня геодезична система координат 1984р., використовується в GPS - навігації
'''
print('Система координат', fire_stations.crs)

#-------------------- створення Shapefile -----------------------

filename_out = "/Users/dasha/projects/DataScience_2026/lab 9/Fire_Stations_SELECTION/Fire_St_SELECT.shp"

# вибір 2 рядків
selection = fire_stations[0:2]

# запис відібраних радків в новий Shapefile
selection.to_file(filename_out)

# парсінг фашлу карти *.shp - формату
fire_stations_out = gpd.read_file(filename_out)

# візуалізація
fig, ax = plt.subplots(figsize=(8, 4))
fire_stations_out.plot(ax=ax, alpha=0.4, color="grey", zorder=1)
fire_stations.plot(ax=ax, markersize=20, color="blue", marker="o", zorder=2)
fire_stations_out.plot(ax=ax, markersize=20, color="red", marker="o", zorder=2)
plt.show()