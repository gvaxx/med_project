#!/usr/bin/env python3
"""
Простой скрипт для загрузки аудиофайла и получения ответа от сервиса.
"""

import requests
import sys
from pathlib import Path

# URL сервиса
SERVICE_URL = "http://localhost:8004"

# Путь к аудиофайлу
PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_FILE = PROJECT_DIR / "data" / "MVI_9180_audio.wav"

# Проверка существования файла
if not AUDIO_FILE.exists():
    print(f"Ошибка: Файл {AUDIO_FILE} не найден")
    sys.exit(1)

# Загрузка файла
print(f"Загрузка файла: {AUDIO_FILE}")
files = {'file': (AUDIO_FILE.name, open(AUDIO_FILE, 'rb'), 'audio/wav')}

try:
    # Отправка запроса с параметром longform=true
    print("Используем параметр longform=true для длинного аудио...")
    params = {
        'long_form': True,
        'model_type': 'rnnt'
    }
    response = requests.post(
        f"{SERVICE_URL}/transcribe", 
        files=files,
        data=params,
        timeout=300  # Увеличиваем таймаут
    )
    
    # Закрытие файла
    files['file'][1].close()
    
    # Проверка ответа
    if response.status_code == 200:
        print("Успешно получен ответ от сервиса:")
        print(response.text)
    else:
        print(f"Ошибка: Сервис вернул код {response.status_code}")
        print(f"Сообщение: {response.text[:200]}")
        sys.exit(1)
        
except Exception as e:
    print(f"Ошибка при отправке запроса: {e}")
    sys.exit(1) 