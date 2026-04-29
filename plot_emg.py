import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import find_peaks

# =========================
# 1. Upload data
# =========================
df = pd.read_csv("emg_for_fistmoveset.csv", delimiter='\t')

# =========================
# 2. Time
# =========================
start_time = df["Timestamp"].iloc[0]
df["time_sec"] = df["Timestamp"] - start_time

# =========================
# 3. Take EMG channels
# =========================
emg_cols = [f"FilteredChannel{i}" for i in range(1, 9)]
emg_data = df[emg_cols].values

# =========================
# 4. Calculating signal strength (RMS лучше чем MAV)
# =========================
signal_power = np.sqrt(np.mean(emg_data**2, axis=1))

# =========================
# 5. Remove outliers
# =========================
signal_power = np.clip(signal_power, 0, np.percentile(signal_power, 99))

# =========================
# 6. Smoothing
# =========================
window_size = 50
signal_power_smooth = pd.Series(signal_power).rolling(window=window_size, center=True).mean()
signal_power_smooth = signal_power_smooth.bfill().ffill()

# =========================
# 7. Automatic threshold selection (improved)
# =========================
# Method 1: Based on histogram peaks
hist, bins = np.histogram(signal_power_smooth, bins=50)
peak_indices, _ = find_peaks(hist)
if len(peak_indices) >= 2:
    # Find two pics (REST и FIST)
    peak_values = bins[peak_indices]
    peak_heights = hist[peak_indices]
    # Sorting by heigh
    sorted_idx = np.argsort(peak_heights)[::-1]
    rest_peak = min(peak_values[sorted_idx[0]], peak_values[sorted_idx[1]])
    fist_peak = max(peak_values[sorted_idx[0]], peak_values[sorted_idx[1]])
    THRESHOLD = (rest_peak + fist_peak) / 2
    print(f"Threshold from peaks: REST={rest_peak:.2f}, FIST={fist_peak:.2f}, TH={THRESHOLD:.2f}")
else:
    # Method 2: Percentile-based fallback
    THRESHOLD = np.percentile(signal_power_smooth, 70)
    print(f"Threshold from percentile 70: {THRESHOLD:.2f}")

# Method 3: Alternative - use mean + std
# THRESHOLD = np.nanmean(signal_power_smooth) + np.nanstd(signal_power_smooth)

# =========================
# 8. Classification
# =========================
labels = []
for val in signal_power_smooth:
    if val > THRESHOLD:
        labels.append("FIST")
    else:
        labels.append("REST")

df["predicted_label"] = labels

# Добавляем задержку (опционально) - убираем короткие всплески
min_fist_duration = 10  # минимальная длина FIST в сэмплах (~0.02 сек)
fist_streak = 0
for i in range(len(labels)):
    if labels[i] == "FIST":
        fist_streak += 1
    else:
        if 0 < fist_streak < min_fist_duration:
            # Слишком короткое сжатие - меняем на REST
            for j in range(i-fist_streak, i):
                labels[j] = "REST"
        fist_streak = 0

df["predicted_label_filtered"] = labels

# =========================
# 9. Statistics
# =========================
print("\nSignal stats:")
print(f"Mean: {np.nanmean(signal_power_smooth):.2f}")
print(f"Std: {np.nanstd(signal_power_smooth):.2f}")
print(f"Max: {np.nanmax(signal_power_smooth):.2f}")
print(f"Min: {np.nanmin(signal_power_smooth):.2f}")
print(f"Threshold: {THRESHOLD:.2f}")

# Statistics for onditions
rest_values = signal_power_smooth[np.array(labels) == "REST"]
fist_values = signal_power_smooth[np.array(labels) == "FIST"]
print(f"\nREST: mean={np.mean(rest_values):.2f}, std={np.std(rest_values):.2f}")
print(f"FIST: mean={np.mean(fist_values):.2f}, std={np.std(fist_values):.2f}")
print(f"Separation ratio: {np.mean(fist_values)/np.mean(rest_values):.2f}x")

# =========================
# 10. Graphics - Improved
# =========================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Signal with threshold
ax1 = axes[0]
ax1.plot(df["time_sec"], signal_power, alpha=0.3, label="Raw Signal", linewidth=0.5)
ax1.plot(df["time_sec"], signal_power_smooth, linewidth=2, label="Smoothed Signal", color='orange')
ax1.axhline(y=THRESHOLD, color='r', linestyle='--', linewidth=2, label=f"Threshold ({THRESHOLD:.1f})")

# Закрашиваем области FIST
for i, label in enumerate(labels):
    if label == "FIST":
        ax1.axvspan(df["time_sec"].iloc[i], df["time_sec"].iloc[i+1] if i+1 < len(df) else df["time_sec"].iloc[i], 
                   alpha=0.3, color='red', linewidth=0)

ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Signal Power (RMS)")
ax1.set_title("EMG Signal with Automatic Threshold Detection")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Classification result
ax2 = axes[1]
colors = ['red' if l == "FIST" else 'green' for l in labels]
ax2.scatter(df["time_sec"], signal_power_smooth, c=colors, s=5, alpha=0.6)
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Smoothed Signal")
ax2.set_title("Classification: RED = FIST, GREEN = REST")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# 11. Additional analysis
# =========================
# Подсчет переходов и длительностей
changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
print(f"\nNumber of state transitions: {changes}")

# Длительности FIST периодов
fist_periods = []
current = 0
for label in labels:
    if label == "FIST":
        current += 1
    elif current > 0:
        fist_periods.append(current)
        current = 0
if current > 0:
    fist_periods.append(current)

if fist_periods:
    sampling_rate = len(df) / df["time_sec"].max()
    fist_durations_sec = [p / sampling_rate for p in fist_periods]
    print(f"FIST periods: {len(fist_periods)}")
    print(f"Average FIST duration: {np.mean(fist_durations_sec):.2f} seconds")
    print(f"Min/Max FIST duration: {min(fist_durations_sec):.2f}/{max(fist_durations_sec):.2f} sec")

# =========================
# 12. Output
# =========================
print("\nUnique labels:", set(labels))
print("Label counts:", Counter(labels))