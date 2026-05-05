from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import numpy as np
import time
from collections import deque

# =========================
# SETTINGS
# =========================
THRESHOLD = 100
WINDOW_SIZE = 50  # сглаживание

# буфер для сглаживания
buffer = deque(maxlen=WINDOW_SIZE)

# =========================
# CONNECT
# =========================
BoardShim.enable_dev_board_logger()

params = MindRoveInputParams()
board_id = BoardIds.MINDROVE_WIFI_BOARD

board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()

print("Streaming started... Press Ctrl+C to stop")

# =========================
# LIVE LOOP
# =========================
try:
    while True:
        data = board.get_current_board_data(50)

        if data.shape[1] > 0:
            emg = data[0:8, :]  # 8 каналов

            # аналог signal_power
            value = np.mean(np.abs(emg))

            # clip как у тебя
            value = np.clip(value, 0, 500)

            # сглаживание через буфер
            buffer.append(value)

            if len(buffer) == WINDOW_SIZE:
                smooth_value = np.mean(buffer)

                if smooth_value > THRESHOLD:
                    print("FIST")
                else:
                    print("REST")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")

    board.stop_stream()
    board.release_session()