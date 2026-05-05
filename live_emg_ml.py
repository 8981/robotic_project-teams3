from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import joblib
from collections import deque
import time
import signal
import sys

# =========================
# ЗАГРУЗКА МОДЕЛИ
# =========================
print("="*50)
print("ЗАГРУЗКА ML МОДЕЛИ")
print("="*50)

try:
    model = joblib.load('emg_gesture_model.pkl')
    scaler = joblib.load('emg_scaler.pkl')
    label_encoder = joblib.load('emg_label_encoder.pkl')
    print("✓ Модель загружена")
    print(f"  Доступные жесты: {label_encoder.classes_.tolist()}")
except FileNotFoundError:
    print("❌ Модель не найдена! Сначала запусти train_model.py")
    sys.exit(1)

# =========================
# НАСТРОЙКИ
# =========================
WINDOW_SIZE = 50      # Размер окна (0.1 сек при 500 Гц)
STEP = 25             # Шаг между окнами
BUFFER_SIZE = WINDOW_SIZE * 2  # 100 сэмплов буфер

# Буферы
emg_buffer = deque(maxlen=BUFFER_SIZE)
prediction_history = deque(maxlen=5)  # Сглаживание предсказаний

# Статистика
last_prediction = None
last_change_time = time.time()
debounce_time = 0.3  # секунд

# =========================
# ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ
# =========================
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
    features.append(np.sum(np.abs(np.diff(window_data, axis=0))))  # WL
    
    return np.array(features)

def predict_gesture(window_data):
    """Предсказывает жест по окну данных"""
    features = extract_features_from_window(window_data)
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    gesture = label_encoder.inverse_transform([pred])[0]
    return gesture

# =========================
# ПОДКЛЮЧЕНИЕ К БРАСЛЕТУ
# =========================
print("\n" + "="*50)
print("ПОДКЛЮЧЕНИЕ К MINDROVE")
print("="*50)

BoardShim.enable_dev_board_logger()

params = MindRoveInputParams()
board_id = BoardIds.MINDROVE_WIFI_BOARD

try:
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    print("✓ Подключено к браслету")
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    sys.exit(1)

# =========================
# ОБРАБОТКА ПРЕРЫВАНИЙ
# =========================
def signal_handler(sig, frame):
    print("\n\nОстановка...")
    board.stop_stream()
    board.release_session()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# =========================
# ОСНОВНОЙ ЦИКЛ
# =========================
print("\n" + "="*50)
print("ЗАПУСК КЛАССИФИКАЦИИ")
print("="*50)
print("Делайте жесты:")
print(f"  {', '.join(label_encoder.classes_)}")
print("\nНажмите Ctrl+C для остановки\n")
print("-"*50)

try:
    while True:
        # Получаем данные
        data = board.get_current_board_data(BUFFER_SIZE)
        
        if data.shape[1] > 0:
            # Берем 8 EMG каналов (первые 8)
            emg = data[0:8, :].T  # Транспонируем: (samples, channels)
            
            # Добавляем в буфер
            for sample in emg:
                emg_buffer.append(sample)
            
            # Если в буфере достаточно данных для окна
            if len(emg_buffer) >= WINDOW_SIZE:
                # Берем последние WINDOW_SIZE сэмплов
                window = np.array(list(emg_buffer))[-WINDOW_SIZE:, :]
                
                # Предсказываем
                gesture = predict_gesture(window)
                prediction_history.append(gesture)
                
                # Сглаживание - берем самый частый жест в истории
                from collections import Counter
                smooth_gesture = Counter(prediction_history).most_common(1)[0][0]
                
                # Дебаунс - не выводим слишком часто
                current_time = time.time()
                if smooth_gesture != last_prediction:
                    if current_time - last_change_time >= debounce_time:
                        print(f"[{current_time - start_time:.1f}s] {smooth_gesture}")
                        last_prediction = smooth_gesture
                        last_change_time = current_time
        
        # Небольшая задержка
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n\nОстановка...")
finally:
    board.stop_stream()
    board.release_session()
    print("Сессия завершена")