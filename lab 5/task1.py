from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram

import seaborn as sns; sns.set()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def file_parsing(File_name):
    d = pd.read_csv(File_name)
    print(d.head())
    print(f"Рядків: {len(d)}, Колонок: {len(d.columns)}")

    n = len(d)
    X = np.zeros((n, 2))
    for i in range(n):
        X[i, 0] = float(i)
        X[i, 1] = float(d['Close'][i])
    return X, d


def KMeans_gold(X):
    print("=" * 60)
    print("1. KMeans кластеризація (k-середніх)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    y_kmeans = kmeans.predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print(f"Центри кластерів (індекс, ціна):")
    for k, c in enumerate(centers):
        print(f"  Кластер {k+1}: індекс={c[0]:.0f}, ціна={c[1]:.2f} USD")

    plt.figure(figsize=(12, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=15, cmap='viridis', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8,
                marker='X', label='Центри кластерів')
    plt.title("KMeans: кластеризація цін ETF GLD (4 кластери)", fontsize=13)
    plt.xlabel("Індекс спостереження (час)")
    plt.ylabel("Ціна закриття, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kmeans_gold.png", dpi=120)
    plt.close()
    print("Графік збережено: kmeans_gold.png")

    # метод ліктя
    inertia = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(k_range, inertia, marker='o', color='steelblue')
    plt.title("KMeans: метод ліктя (вибір оптимального k)", fontsize=12)
    plt.xlabel("Кількість кластерів k")
    plt.ylabel("Інерція (сума квадратів відстаней)")
    plt.tight_layout()
    plt.savefig("kmeans_elbow.png", dpi=120)
    plt.close()
    print("Графік збережено: kmeans_elbow.png")

    return y_kmeans


def Hierarchy_gold(X):
    print("=" * 60)
    print("3. Ієрархічна кластеризація")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    step = max(1, len(X) // 40)
    X_sub = X_scaled[::step]
    labels = list(range(1, len(X_sub) + 1))

    linked = linkage(X_sub, 'ward')

    plt.figure(figsize=(14, 5))
    dendrogram(linked, orientation='top', labels=labels,
               distance_sort='descending', show_leaf_counts=True)
    plt.title("Ієрархічна кластеризація", fontsize=13)
    plt.xlabel("Індекс точки")
    plt.ylabel("Відстань")
    plt.tight_layout()
    plt.savefig("hierarchy_dendrogram.png", dpi=120)
    plt.close()
    print("Графік збережено: hierarchy_dendrogram.png")

    cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
    cluster.fit_predict(X_scaled)

    plt.figure(figsize=(12, 5))
    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, s=15,
                cmap='rainbow', alpha=0.7)
    plt.title("Ієрархічна кластеризація: ціни ETF GLD (4 кластери)", fontsize=13)
    plt.xlabel("Індекс спостереження (час)")
    plt.ylabel("Ціна закриття, USD")
    plt.tight_layout()
    plt.savefig("hierarchy_clusters.png", dpi=120)
    plt.close()
    print("Графік збережено: hierarchy_clusters.png")

    return cluster.labels_


def compare_clusters(X, d, y_kmeans, y_hier):
    print("=" * 60)
    print("4. Аналіз результатів кластеризації")

    df_result = pd.DataFrame({
        'Date'    : d['Date'].values,
        'Close'   : X[:, 1],
        'KMeans'  : y_kmeans + 1,
        'Hierarchy': y_hier  + 1,
    })

    print("\nСтатистика по кластерах KMeans:")
    print(df_result.groupby('KMeans')['Close'].agg(['min','max','mean','count']).round(2))

    print("\nСтатистика по кластерах ієрархічної кластеризації:")
    print(df_result.groupby('Hierarchy')['Close'].agg(['min','max','mean','count']).round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=12, cmap='viridis', alpha=0.7)
    axes[0].set_title("KMeans (k=4)", fontsize=12)
    axes[0].set_xlabel("Індекс (час)")
    axes[0].set_ylabel("Ціна, USD")

    axes[1].scatter(X[:, 0], X[:, 1], c=y_hier, s=12, cmap='rainbow', alpha=0.7)
    axes[1].set_title("Ієрархічна кластеризація (k=4)", fontsize=12)
    axes[1].set_xlabel("Індекс (час)")
    axes[1].set_ylabel("Ціна, USD")

    plt.suptitle("Порівняння методів кластеризації: ETF GLD", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    plt.close()
    print("Графік збережено: comparison.png")

    print("=" * 60)
    print("ВИСНОВКИ:")
    print(f"  Всього спостережень: {len(X)}")
    print(f"  Діапазон цін: {X[:,1].min():.2f} – {X[:,1].max():.2f} USD")
    print()
    km_stats = df_result.groupby('KMeans')['Close'].agg(['min','max','mean'])
    for k, row in km_stats.iterrows():
        print(f"  KMeans кластер {k}: ціни {row['min']:.1f}–{row['max']:.1f} USD "
              f"(середня {row['mean']:.1f} USD)")
    print()


if __name__ == '__main__':

    FILE_NAME = "gold_prices.csv"

    X, d = file_parsing(FILE_NAME)

    y_kmeans = KMeans_gold(X)

    y_hier = Hierarchy_gold(X)

    compare_clusters(X, d, y_kmeans, y_hier)