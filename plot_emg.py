import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("emg_for_fistmoveset.csv", delimiter='\t')

print("Columns:")
print(df.columns.tolist())

# =========================
# 2. Время
# =========================
start_time = df["Timestamp"].iloc[0]
df["time_sec"] = df["Timestamp"] - start_time

# =========================
# 3. Берём EMG каналы
# =========================
emg_cols = [f"FilteredChannel{i}" for i in range(1, 9)]
emg_data = df[emg_cols].values

# =========================
# 4. Считаем мощность сигнала
# =========================
signal_power = np.mean(np.abs(emg_data), axis=1)

# =========================
# 5. Убираем слишком большие пики (опционально, но полезно)
# =========================
signal_power = np.clip(signal_power, 0, 500)

# =========================
# 6. Сглаживание (moving average)
# =========================
window_size = 50  # ~0.1 сек при 500 Hz
signal_power_smooth = pd.Series(signal_power).rolling(window=window_size).mean()

# =========================
# 7. Выбор threshold
# =========================
THRESHOLD = 100  # попробуй 60–120 если не идеально

# =========================
# 8. Классификация
# =========================
labels = []

for val in signal_power_smooth:
    if np.isnan(val):
        labels.append("REST")
    elif val > THRESHOLD:
        labels.append("FIST")
    else:
        labels.append("REST")

df["predicted_label"] = labels

# =========================
# 9. Статистика
# =========================
print("\nSignal stats:")
print("Mean:", np.nanmean(signal_power_smooth))
print("Max:", np.nanmax(signal_power_smooth))

# =========================
# 10. Графики
# =========================
plt.figure(figsize=(12, 6))

# raw сигнал (прозрачный)
plt.plot(df["time_sec"], signal_power, alpha=0.3, label="Raw Signal")

# сглаженный
plt.plot(df["time_sec"], signal_power_smooth, linewidth=2, label="Smoothed Signal")

# threshold
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label="Threshold")

plt.xlabel("Time (seconds)")
plt.ylabel("Signal Power")
plt.title("EMG Signal Power with Smoothing and Threshold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 11. Визуализация классов
# =========================
plt.figure(figsize=(12, 4))

colors = [1 if l == "FIST" else 0 for l in labels]

plt.scatter(df["time_sec"], signal_power_smooth, c=colors, s=3)
plt.title("FIST (1) vs REST (0)")
plt.xlabel("Time (seconds)")
plt.ylabel("Smoothed Signal")

plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 12. Пример вывода
# =========================
print("\nSample predictions:")
print(df[["time_sec", "predicted_label"]].head(50))
print("\nUnique labels:")
print(set(labels))
print(Counter(labels))