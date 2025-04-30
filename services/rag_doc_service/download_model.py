from sentence_transformers import SentenceTransformer
import os

# Путь для сохранения модели
MODEL_PATH = "models/all-MiniLM-L6-v2"

def download_model():
    print("Downloading model all-MiniLM-L6-v2...")
    # Создаем директорию если её нет
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Загружаем модель
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Сохраняем модель
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    download_model() 