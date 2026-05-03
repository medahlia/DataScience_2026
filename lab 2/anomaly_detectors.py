import StandardScaler
import requests
import pandas as pd
import numpy as np
import math as mt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, random, warnings
warnings.filterwarnings("ignore")


def tune_anomaly_detectors(prices_anom: np.ndarray,
                           true_indices: list) -> dict:
    """
    Тестування та підбір параметрів для трьох алгоритмів.
    Підхід:
      - Grid search по ключових гіперпараметрах
      - Критерій оптимальності: максимальний F1-score
    """
    print("=" * 60)
    print("10. Навчання параметрів алгоритмів виявлення аномалій")
    n = len(prices_anom)
    X = prices_anom.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_results = {}

    # ── (A) Isolation Forest ──────────────────────────────────
    print("\n  [A] Isolation Forest")
    contamination_vals = [0.02, 0.05, 0.08, 0.10, 0.12]
    n_estimators_vals = [50, 100, 200]
    best_if = {"f1": -1}
    if_rows = []
    for cont in contamination_vals:
        for n_est in n_estimators_vals:
            clf = IsolationForest(contamination=cont, n_estimators=n_est,
                                  random_state=42, n_jobs=-1)
            labels = clf.fit_predict(X_scaled)
            m = _evaluate_detector(labels, true_indices, n)
            if_rows.append({"cont": cont, "n_est": n_est, **m})
            if m["f1"] > best_if["f1"]:
                best_if = {"f1": m["f1"], "cont": cont, "n_est": n_est,
                           "precision": m["precision"], "recall": m["recall"]}

    print(f"  Кращі параметри: contamination={best_if['cont']}, n_estimators={best_if['n_est']}")
    print(f"  F1={best_if['f1']:.4f}  Precision={best_if['precision']:.4f}  Recall={best_if['recall']:.4f}")
    best_results["IsolationForest"] = best_if

    # ── (B) Local Outlier Factor ──────────────────────────────
    print("\n  [B] Local Outlier Factor (LOF)")
    n_neighbors_vals = [5, 10, 20, 30]
    contamination_vals2 = [0.02, 0.05, 0.08, 0.10]
    best_lof = {"f1": -1}
    for n_nbr in n_neighbors_vals:
        for cont in contamination_vals2:
            clf = LocalOutlierFactor(n_neighbors=n_nbr, contamination=cont)
            labels = clf.fit_predict(X_scaled)
            m = _evaluate_detector(labels, true_indices, n)
            if m["f1"] > best_lof["f1"]:
                best_lof = {"f1": m["f1"], "n_neighbors": n_nbr, "cont": cont,
                            "precision": m["precision"], "recall": m["recall"]}

    print(f"  Кращі параметри: n_neighbors={best_lof['n_neighbors']}, contamination={best_lof['cont']}")
    print(f"  F1={best_lof['f1']:.4f}  Precision={best_lof['precision']:.4f}  Recall={best_lof['recall']:.4f}")
    best_results["LOF"] = best_lof

    # ── (C) DBSCAN ────────────────────────────────────────────
    print("\n  [C] DBSCAN")
    eps_vals = [0.1, 0.2, 0.3, 0.5, 0.8]
    min_samples_vals = [3, 5, 10]
    best_db = {"f1": -1}
    for eps in eps_vals:
        for min_s in min_samples_vals:
            clf = DBSCAN(eps=eps, min_samples=min_s)
            raw_labels = clf.fit_predict(X_scaled)
            # у DBSCAN аномалії = -1 (шум), всі решта — кластери
            labels_bin = np.where(raw_labels == -1, -1, 1)
            m = _evaluate_detector(labels_bin, true_indices, n)
            if m["f1"] > best_db["f1"]:
                best_db = {"f1": m["f1"], "eps": eps, "min_samples": min_s,
                           "precision": m["precision"], "recall": m["recall"]}

    print(f"  Кращі параметри: eps={best_db['eps']}, min_samples={best_db['min_samples']}")
    print(f"  F1={best_db['f1']:.4f}  Precision={best_db['precision']:.4f}  Recall={best_db['recall']:.4f}")
    best_results["DBSCAN"] = best_db

    # ── Порівняльна таблиця ───────────────────────────────────
    print("\n  Порівняння алгоритмів:")
    print(f"  {'Алгоритм':<20} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    print("  " + "-" * 52)
    best_algo = max(best_results, key=lambda k: best_results[k]["f1"])
    for algo, res in best_results.items():
        marker = " ◄ КРАЩИЙ" if algo == best_algo else ""
        print(f"  {algo:<20} {res['f1']:>8.4f} {res['precision']:>12.4f} {res['recall']:>10.4f}{marker}")

    # ── Графік порівняння F1 ──────────────────────────────────
    algo_names = list(best_results.keys())
    f1_vals = [best_results[a]["f1"] for a in algo_names]
    colors_bar = ["steelblue", "crimson", "darkorange"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(algo_names, f1_vals, color=colors_bar, alpha=0.85, edgecolor="white")
    plt.axhline(max(f1_vals), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, f1_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 1.1)
    plt.title("Порівняння F1-score алгоритмів виявлення аномалій", fontsize=12)
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig("anomaly_algorithms_comparison.png", dpi=120)
    plt.close()
    print(f"\n  Графік збережено: {os.path.abspath('anomaly_algorithms_comparison.png')}")

    # ── Візуалізація найкращого детектора ─────────────────────
    print(f"\n  Застосовуємо найкращий алгоритм: {best_algo}")
    if best_algo == "IsolationForest":
        clf_best = IsolationForest(contamination=best_if["cont"],
                                   n_estimators=best_if["n_est"], random_state=42)
    elif best_algo == "LOF":
        clf_best = LocalOutlierFactor(n_neighbors=best_lof["n_neighbors"],
                                      contamination=best_lof["cont"])
    else:
        clf_best = DBSCAN(eps=best_db["eps"], min_samples=best_db["min_samples"])

    if best_algo == "DBSCAN":
        raw_labels = clf_best.fit_predict(X_scaled)
        best_labels = np.where(raw_labels == -1, -1, 1)
    else:
        best_labels = clf_best.fit_predict(X_scaled)

    detected_best = np.where(best_labels == -1)[0]
    plt.figure(figsize=(13, 5))
    plt.plot(prices_anom, color="steelblue", linewidth=1, label="Дані з аномаліями")
    plt.scatter(true_indices, prices_anom[true_indices],
                color="red", s=50, zorder=5, label="Істинні аномалії")
    plt.scatter(detected_best, prices_anom[detected_best],
                marker="x", color="orange", s=60, zorder=6,
                linewidths=2, label=f"Виявлені {best_algo}")
    plt.title(f"Найкращий детектор: {best_algo}", fontsize=13)
    plt.xlabel("Індекс спостереження")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_detector_result.png", dpi=120)
    plt.close()
    print(f"  Графік збережено: {os.path.abspath('best_detector_result.png')}")

    return best_results
