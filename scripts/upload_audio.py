#!/usr/bin/env python3
"""
Скрипт для загрузки аудиофайла из папки data в аудио-сервис и получения ответа.
"""

import os
import sys
import requests
import json
from pathlib import Path

def upload_audio_file(service_url, audio_file_path):
    """
    Загружает аудиофайл в сервис и получает ответ.
    
    Args:
        service_url (str): URL аудио-сервиса
        audio_file_path (str): Путь к аудиофайлу
        
    Returns:
        dict: Ответ от сервиса
    """
    try:
        print(f"Загрузка файла: {audio_file_path}")
        
        # Проверка существования файла
        if not os.path.exists(audio_file_path):
            print(f"Ошибка: Файл {audio_file_path} не найден")
            return None
            
        # Подготовка файла для загрузки
        files = {
            'audio': (os.path.basename(audio_file_path), open(audio_file_path, 'rb'), 'audio/wav')
        }
        
        # Загрузка файла в сервис
        response = requests.post(
            f"{service_url}/transcribe", 
            files=files,
            timeout=180  # Увеличенный таймаут для больших файлов
        )
        
        # Закрытие файла
        files['audio'][1].close()
        
        # Проверка ответа
        if response.status_code == 200:
            try:
                result = response.json()
                print("Файл успешно обработан!")
                return result
            except json.JSONDecodeError:
                print(f"Сервис вернул не JSON ответ: {response.text[:200]}")
        else:
            print(f"Сервис вернул код ошибки: {response.status_code}")
            print(f"Сообщение: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print(f"Ошибка соединения: Не удалось подключиться к сервису по адресу {service_url}")
    except requests.exceptions.Timeout:
        print(f"Ошибка таймаута: Сервис не ответил вовремя")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")
        
    return None

if __name__ == "__main__":
    # Определение директории проекта (два уровня выше директории скрипта)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    # Путь к файлу по умолчанию
    default_audio_file = project_dir / "data" / "MVI_9180_audio.wav"
    
    # URL сервиса по умолчанию
    default_url = "http://localhost:8004"
    
    # Использование аргументов командной строки, если они предоставлены
    service_url = default_url
    audio_file_path = default_audio_file
    
    if len(sys.argv) > 1:
        service_url = sys.argv[1]
    
    if len(sys.argv) > 2:
        audio_file_path = sys.argv[2]
    
    print(f"Сервис: {service_url}")
    print(f"Аудиофайл: {audio_file_path}")
    
    # Загрузка файла и получение ответа
    result = upload_audio_file(service_url, audio_file_path)
    
    # Вывод результата
    if result:
        print("\nРезультат обработки:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
    else:
        print("\nНе удалось получить результат от сервиса")
        sys.exit(1) 