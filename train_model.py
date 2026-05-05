import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# =========================
# 1. Загрузка и разметка данных
# =========================

def label_by_time(df, pattern, seconds_per_gesture=5):
    """
    Размечает данные по временному паттерну
    pattern: список жестов в порядке следования
    """
    sampling_rate = len(df) / df["time_sec"].max()
    samples_per_gesture = int(seconds_per_gesture * sampling_rate)
    
    labels = []
    # Повторяем паттерн несколько раз, чтобы покрыть всю длину
    while len(labels) < len(df):
        for gesture in pattern:
            labels.extend([gesture] * samples_per_gesture)
            if len(labels) >= len(df):
                break
        if len(labels) >= len(df):
            break
    
    # Обрезаем до точной длины
    return labels[:len(df)]

def load_and_label(filepath, pattern, gesture_name):
    """Загружает файл и добавляет метки"""
    df = pd.read_csv(filepath, delimiter='\t')
    
    # Время
    start_time = df["Timestamp"].iloc[0]
    df["time_sec"] = df["Timestamp"] - start_time
    
    print(f"  {gesture_name}: {len(df)} samples, {df['time_sec'].max():.1f} сек")
    
    # Разметка
    labels = label_by_time(df, pattern)
    df['gesture'] = labels
    
    print(f"    Распределение: {dict(pd.Series(labels).value_counts())}")
    
    return df

# Паттерны для каждого файла
patterns = {
    'FIST': ['REST', 'FIST', 'REST', 'FIST', 'REST', 'FIST', 'REST'],
    'UP_DOWN': ['REST', 'UP', 'REST', 'DOWN', 'REST', 'UP', 'REST', 'DOWN', 'REST'],
    'LEFT_RIGHT': ['REST', 'LEFT', 'REST', 'RIGHT', 'REST', 'LEFT', 'REST', 'RIGHT', 'REST']
}

# Загрузка всех файлов
print("="*50)
print("1. ЗАГРУЗКА ДАННЫХ")
print("="*50)

files = {
    'FIST': 'emg_for_fistmoveset.csv',
    'UP_DOWN': 'emg_for_fist_up_down.csv',
    'LEFT_RIGHT': 'emg_for_fist_left_right.csv'
}

all_dfs = []
for name, path in files.items():
    try:
        df = load_and_label(path, patterns[name], name)
        all_dfs.append(df)
    except FileNotFoundError:
        print(f"  Файл не найден: {path}")

if not all_dfs:
    print("❌ Нет файлов для обработки!")
    exit()

# Объединение
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nВсего samples: {len(df_all)}")
print(f"Все жесты: {df_all['gesture'].unique()}")

# =========================
# 2. Извлечение признаков
# =========================

print("\n" + "="*50)
print("2. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
print("="*50)

def extract_features_from_window(window_data):
    """Извлекает признаки из окна EMG данных (8 каналов)"""
    features = []
    
    for ch in range(8):
        ch_data = window_data[:, ch]
        features.append(np.mean(np.abs(ch_data)))      # MAV
        features.append(np.std(ch_data))                # STD
        features.append(np.max(ch_data) - np.min(ch_data))  # Range
        features.append(np.percentile(ch_data, 75))     # 75% percentile
        features.append(np.percentile(ch_data, 25))     # 25% percentile
    
    # Общие признаки
    features.append(np.mean(np.abs(window_data)))       # Общая MAV
    features.append(np.std(window_data))                # Общая STD
    features.append(np.max(window_data))                # Максимум
    
    return features

def prepare_dataset(df, window_size=50, step=25):
    """Подготавливает датасет с признаками"""
    emg_cols = [f"FilteredChannel{i}" for i in range(1, 9)]
    emg_data = df[emg_cols].values
    labels = df['gesture'].values
    
    X = []
    y = []
    
    for i in range(0, len(emg_data) - window_size, step):
        window = emg_data[i:i+window_size, :]
        
        # Извлекаем признаки
        features = extract_features_from_window(window)
        X.append(features)
        
        # Метка (самый частый жест в окне)
        window_labels = labels[i:i+window_size]
        from collections import Counter
        majority_label = Counter(window_labels).most_common(1)[0][0]
        y.append(majority_label)
    
    return np.array(X), np.array(y)

# Подготовка
window_size = 50
step = 25

X, y = prepare_dataset(df_all, window_size, step)
print(f"  Окон: {len(X)}")
print(f"  Признаков на окно: {X.shape[1]}")

# =========================
# 3. Кодирование меток
# =========================

print("\n" + "="*50)
print("3. КОДИРОВАНИЕ МЕТОК")
print("="*50)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"  Классы: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# =========================
# 4. Разделение на train/test
# =========================

print("\n" + "="*50)
print("4. РАЗДЕЛЕНИЕ ДАННЫХ")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# =========================
# 5. Обучение модели
# =========================

print("\n" + "="*50)
print("5. ОБУЧЕНИЕ МОДЕЛИ")
print("="*50)

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =========================
# 6. Оценка модели
# =========================

print("\n" + "="*50)
print("6. ОЦЕНКА МОДЕЛИ")
print("="*50)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {accuracy*100:.2f}%")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - EMG Gesture Recognition')

# Добавляем цифры
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# =========================
# 7. Сохранение модели
# =========================

print("\n" + "="*50)
print("7. СОХРАНЕНИЕ МОДЕЛИ")
print("="*50)

joblib.dump(model, 'emg_gesture_model.pkl')
joblib.dump(scaler, 'emg_scaler.pkl')
joblib.dump(label_encoder, 'emg_label_encoder.pkl')

# Сохраняем параметры
params = {
    'window_size': window_size,
    'step': step,
    'n_features': X.shape[1],
    'n_classes': len(label_encoder.classes_),
    'classes': label_encoder.classes_.tolist(),
    'accuracy': float(accuracy),
}

with open('emg_model_params.json', 'w') as f:
    json.dump(params, f, indent=4)

print("  ✓ model: emg_gesture_model.pkl")
print("  ✓ scaler: emg_scaler.pkl")
print("  ✓ encoder: emg_label_encoder.pkl")
print("  ✓ params: emg_model_params.json")

print("\n✅ Обучение завершено!")