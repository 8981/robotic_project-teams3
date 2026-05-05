from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import joblib
from collections import deque
import time
import sys
from collections import Counter

print("="*50)
print("ЗАГРУЗКА ML МОДЕЛИ")
print("="*50)

try:
    model = joblib.load('emg_gesture_model.pkl')
    scaler = joblib.load('emg_scaler.pkl')
    label_encoder = joblib.load('emg_label_encoder.pkl')
    print("✓ Модель загружена")
    print(f"  Доступные жесты: {label_encoder.classes_.tolist()}")
except FileNotFoundError as e:
    print(f"❌ Модель не найдена: {e}")
    sys.exit(1)

# =========================
# НАСТРОЙКИ
# =========================
WINDOW_SIZE = 50
BUFFER_SIZE = 100
DEBOUNCE_TIME = 0.5
CONFIDENCE_THRESHOLD = 0.6  # Минимальная уверенность (0-1)

# Буферы
emg_buffer = deque(maxlen=WINDOW_SIZE)
prediction_history = deque(maxlen=5)
confidence_history = deque(maxlen=5)

# Статистика
last_prediction = None
last_change_time = time.time()

# =========================
# ФУНКЦИЯ ПРИЗНАКОВ
# =========================
def extract_features_from_window(window_data):
    features = []
    for ch in range(8):
        ch_data = window_data[:, ch]
        features.append(np.mean(np.abs(ch_data)))
        features.append(np.std(ch_data))
        features.append(np.max(ch_data) - np.min(ch_data))
        features.append(np.percentile(ch_data, 75))
        features.append(np.percentile(ch_data, 25))
    features.append(np.mean(np.abs(window_data)))
    features.append(np.std(window_data))
    features.append(np.max(window_data))
    return np.array(features)

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
    print("  Ожидание данных...")
    time.sleep(2)
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    sys.exit(1)

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

start_time = time.time()

try:
    while True:
        data = board.get_current_board_data(BUFFER_SIZE)
        
        if data is not None and data.shape[1] > 0:
            emg = data[0:8, :]
            
            for i in range(emg.shape[1]):
                emg_buffer.append(emg[:, i])
            
            if len(emg_buffer) == WINDOW_SIZE:
                window = np.array(list(emg_buffer)).T
                
                if np.any(window):
                    features = extract_features_from_window(window)
                    features_scaled = scaler.transform([features])
                    
                    # Получаем вероятности классов
                    proba = model.predict_proba(features_scaled)[0]
                    max_proba = np.max(proba)
                    pred_class = np.argmax(proba)
                    
                    # Используем предсказание только если уверенность высокая
                    if max_proba >= CONFIDENCE_THRESHOLD:
                        gesture = label_encoder.inverse_transform([pred_class])[0]
                        confidence_history.append(max_proba)
                        prediction_history.append(gesture)
                        
                        # Сглаживание
                        smooth_gesture = Counter(prediction_history).most_common(1)[0][0]
                        avg_confidence = np.mean(confidence_history)
                        
                        # Дебаунс
                        current_time = time.time()
                        if smooth_gesture != last_prediction:
                            if current_time - last_change_time >= DEBOUNCE_TIME:
                                elapsed = current_time - start_time
                                signal_val = np.mean(np.abs(window))
                                print(f"[{elapsed:.1f}s] {smooth_gesture} (conf: {avg_confidence:.2f}, signal: {signal_val:.1f})")
                                last_prediction = smooth_gesture
                                last_change_time = current_time
                    else:
                        # Низкая уверенность - показываем REST
                        current_time = time.time()
                        if current_time - last_change_time >= DEBOUNCE_TIME and last_prediction != "REST":
                            elapsed = current_time - start_time
                            print(f"[{elapsed:.1f}s] REST (low confidence: {max_proba:.2f})")
                            last_prediction = "REST"
                            last_change_time = current_time
                
                # Сдвиг окна
                if len(emg_buffer) > 0:
                    emg_buffer.popleft()
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nОстановка...")
finally:
    board.stop_stream()
    board.release_session()
    print("Сессия завершена")