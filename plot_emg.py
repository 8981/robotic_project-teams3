import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json

# =========================
# 1. Загрузка всех файлов
# =========================
files = {
    'FIST': "emg_for_fistmoveset.csv",
    'UP_DOWN': "emg_for_fist_up_down.csv",
    'LEFT_RIGHT': "emg_for_fist_left_right.csv"
}

# Функция обработки одного файла
def process_emg_file(filepath, gesture_name):
    print(f"\n{'='*50}")
    print(f"Processing: {gesture_name}")
    print(f"{'='*50}")
    
    # Загрузка
    df = pd.read_csv(filepath, delimiter='\t')
    
    # Время
    start_time = df["Timestamp"].iloc[0]
    df["time_sec"] = df["Timestamp"] - start_time
    
    # EMG каналы
    emg_cols = [f"FilteredChannel{i}" for i in range(1, 9)]
    emg_data = df[emg_cols].values
    
    # RMS сигнал
    signal_rms = np.sqrt(np.mean(emg_data**2, axis=1))
    
    # Обрезаем выбросы
    signal_rms = np.clip(signal_rms, 0, np.percentile(signal_rms, 99))
    
    # Сглаживание
    window_size = 50
    signal_smooth = pd.Series(signal_rms).rolling(window=window_size, center=True).mean()
    signal_smooth = signal_smooth.bfill().ffill().values
    
    return {
        'df': df,
        'time': df["time_sec"].values,
        'signal': signal_smooth,
        'duration': df["time_sec"].max()
    }

# =========================
# 2. Обработка всех файлов
# =========================
data = {}
for name, path in files.items():
    try:
        data[name] = process_emg_file(path, name)
    except FileNotFoundError:
        print(f"Файл не найден: {path}")

# =========================
# 3. Поиск порогов для каждого жеста
# =========================
def find_threshold(signal):
    hist, bins = np.histogram(signal, bins=50)
    peak_indices, _ = find_peaks(hist)
    if len(peak_indices) >= 2:
        peak_values = bins[peak_indices]
        peak_heights = hist[peak_indices]
        sorted_idx = np.argsort(peak_heights)[::-1]
        rest_peak = min(peak_values[sorted_idx[0]], peak_values[sorted_idx[1]])
        active_peak = max(peak_values[sorted_idx[0]], peak_values[sorted_idx[1]])
        return (rest_peak + active_peak) / 2
    else:
        return np.percentile(signal, 70)

print("\n=== ПОРОГИ ДЛЯ КАЖДОГО ЖЕСТА ===")
thresholds = {}
for name, d in data.items():
    th = find_threshold(d['signal'])
    thresholds[name] = th
    print(f"{name}: {th:.2f}")

# =========================
# 4. Сравнение всех сигналов на одном графике
# =========================
plt.figure(figsize=(14, 8))

colors = {'FIST': 'red', 'UP_DOWN': 'blue', 'LEFT_RIGHT': 'green'}

for name, d in data.items():
    plt.plot(d['time'], d['signal'], color=colors[name], alpha=0.7, linewidth=1, label=name)

plt.xlabel("Time (seconds)")
plt.ylabel("EMG Signal (RMS)")
plt.title("Сравнение EMG сигналов для разных жестов")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 5. Статистика по каждому жесту
# =========================
print("\n=== СТАТИСТИКА ПО КАЖДОМУ ФАЙЛУ ===")
for name, d in data.items():
    signal = d['signal']
    print(f"\n{name}:")
    print(f"  Длительность: {d['duration']:.1f} сек")
    print(f"  Среднее: {np.mean(signal):.2f}")
    print(f"  Максимум: {np.max(signal):.2f}")
    print(f"  Медиана: {np.median(signal):.2f}")
    print(f"  Порог: {thresholds[name]:.2f}")

# =========================
# 6. Анализ по периодам (5 сек REST, 5 сек жест)
# =========================
def analyze_cycles(signal, sampling_rate, pattern_name):
    """Анализирует циклы по 5 секунд"""
    cycle_duration = 5  # секунд
    samples_per_cycle = int(cycle_duration * sampling_rate)
    
    # Разбиваем на циклы
    n_cycles = len(signal) // samples_per_cycle
    
    cycle_means = []
    for i in range(n_cycles):
        segment = signal[i*samples_per_cycle:(i+1)*samples_per_cycle]
        cycle_means.append(np.mean(segment))
    
    return cycle_means

print("\n=== АНАЛИЗ ПО ЦИКЛАМ (5 СЕКУНД) ===")
for name, d in data.items():
    sampling_rate = len(d['signal']) / d['duration']
    cycles = analyze_cycles(d['signal'], sampling_rate, name)
    
    # Разделяем на REST и активные циклы
    # Паттерн: REST -> ACTIVE -> REST -> ACTIVE -> REST
    if len(cycles) >= 5:
        rest_cycles = [cycles[0], cycles[2], cycles[4]]  # четные индексы
        active_cycles = [cycles[1], cycles[3]]            # нечетные индексы
        
        print(f"\n{name}:")
        print(f"  REST (покой): среднее = {np.mean(rest_cycles):.2f} ± {np.std(rest_cycles):.2f}")
        print(f"  ACTIVE (жест): среднее = {np.mean(active_cycles):.2f} ± {np.std(active_cycles):.2f}")
        print(f"  Отношение: {np.mean(active_cycles)/np.mean(rest_cycles):.2f}x")

# =========================
# 7. Сохранение параметров для ML
# =========================
params_all = {
    'thresholds': thresholds,
    'sampling_rate': len(data['FIST']['signal']) / data['FIST']['duration'] if 'FIST' in data else 500,
    'window_size': 50,
    'gestures': list(data.keys())
}

with open('emg_all_gestures_params.json', 'w') as f:
    json.dump(params_all, f, indent=4)

print(f"\n✓ Параметры сохранены в 'emg_all_gestures_params.json'")
print(f"  Обнаружены жесты: {params_all['gestures']}")