import soundfile as sf
import os
from silero_tts.silero_tts import SileroTTS
import csv

models = SileroTTS.get_available_models()
print("Available models:", models)

latest_model = SileroTTS.get_latest_model('ru')
print("Latest model for Russian:", latest_model)


tts = SileroTTS(model_id='v4_ru', language='ru', speaker='eugene', sample_rate=48000, device='cpu')

speakers = tts.get_available_speakers()
print("Available speakers:", speakers)

# Medical phrases for testing ASR
phrases = [
    "Пациент жалуется на умеренные боли в эпигастрии",
    "Рекомендую сдать общий анализ крови и биохимию",
    "Диагноз: хронический гастрит в стадии обострения",
    "Назначен курс антибактериальной терапии",
    "Показатели артериального давления 140 на 90",
    "Рекомендуется консультация кардиолога",
    "Пациент отмечает улучшение самочувствия",
    "Проведено ультразвуковое исследование брюшной полости",
    "Выявлены признаки хронического панкреатита",
    "Назначена диета стол номер 5",
    "Пациент жалуется на головные боли в затылочной области",
    "Рекомендуется МРТ головного мозга",
    "Выявлена артериальная гипертензия второй степени",
    "Назначены гипотензивные препараты",
    "Проведена электрокардиография",
    "Выявлены признаки ишемии миокарда",
    "Рекомендована консультация невролога",
    "Пациент отмечает снижение остроты зрения",
    "Проведено измерение внутриглазного давления",
    "Диагноз: начальная катаракта",
    "Назначены витаминные капли",
    "Пациент жалуется на боли в коленном суставе",
    "Проведена рентгенография коленного сустава",
    "Выявлены признаки артроза",
    "Рекомендована лечебная физкультура",
    "Назначены хондропротекторы",
    "Пациент отмечает улучшение после физиотерапии",
    "Проведен курс внутрисуставных инъекций",
    "Рекомендовано снижение нагрузки на сустав",
    "Назначен повторный прием через месяц"
]

def main():
    # Create output directory
    os.makedirs("data/asr_data", exist_ok=True)
    
    # Generate audio files and save metadata
    with open("data/asr_data/meta.csv", "w", encoding="utf8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['filename', 'text'])  # Header
        
        for i, text in enumerate(phrases):
            # Generate audio
            
            # Save audio file
            filename = f"{i:02d}.wav"
            tts.tts(text, f"data/asr_data/{filename}")            
            # Save metadata
            writer.writerow([filename, text])
            print(f"Generated {filename}")

if __name__ == "__main__":
    main() 