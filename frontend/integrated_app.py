import streamlit as st
import requests
import json
import os
from datetime import datetime
import tempfile
from moviepy import VideoFileClip
from pydub import AudioSegment
import io

# Service URLs from environment variables
MEDICAL_DOC_SERVICE_URL = os.getenv("MEDICAL_DOC_SERVICE_URL", "http://medical-doc-service:8000")
AUDIO_TRANSCRIPTION_SERVICE_URL = os.getenv("AUDIO_TRANSCRIPTION_SERVICE_URL", "http://audio-transcription-service:8004")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8003")

# Configure the page
st.set_page_config(
    page_title="Медицинская документация",
    page_icon="🏥",
    layout="wide"
)

def get_available_models():
    """Get available LLM models"""
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/available_models")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"gpt-3.5-turbo": True, "gpt-4": True}  # Default models if service unreachable

# Get available LLM models
available_models = get_available_models()


def extract_audio_from_video(video_file):
    """Extract audio from video file"""
    try:
        # Create a temporary file to save the video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        # Extract audio using moviepy
        video = VideoFileClip(temp_video_path)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            video.audio.write_audiofile(temp_audio.name)
            audio_segment = AudioSegment.from_wav(temp_audio.name)
            
        # Cleanup temporary files
        os.unlink(temp_video_path)
        os.unlink(temp_audio.name)
        video.close()
        
        return audio_segment
    except Exception as e:
        st.error(f"Ошибка при извлечении аудио из видео: {str(e)}")
        return None

def load_audio_file(audio_file):
    """Load audio file into AudioSegment"""
    try:
        # Read the uploaded file into memory
        audio_bytes = audio_file.read()
        
        # Create a temporary file with the correct extension
        ext = audio_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            audio_segment = AudioSegment.from_file(temp_audio.name, format=ext)
            os.unlink(temp_audio.name)
        return audio_segment
    except Exception as e:
        st.error(f"Ошибка при загрузке аудио файла: {str(e)}")
        return None

def combine_audio_segments(audio_segments):
    """Combine multiple audio segments into one"""
    try:
        combined = AudioSegment.empty()
        for segment in audio_segments:
            combined += segment
        return combined
    except Exception as e:
        st.error(f"Ошибка при объединении аудио файлов: {str(e)}")
        return None

def audio_segment_to_file(audio_segment):
    """Convert AudioSegment to file-like object"""
    try:
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Ошибка при конвертации аудио: {str(e)}")
        return None 


# Sidebar for model selection
with st.sidebar:
    st.header("Настройки модели")
    selected_model = st.selectbox(
        "Выберите модель ИИ",
        options=list(available_models.keys()),
        help="Выберите модель для генерации медицинской документации"
    )
    
    st.markdown("""
    **О моделях:**
    - **GPT-4**: Наиболее мощная модель для сложных медицинских случаев
    - **GPT-3.5**: Быстрая модель для стандартных случаев
    """)

def check_services():
    """Check if all required services are available"""
    services_status = {}
    
    # Check audio transcription service
    try:
        response = requests.get(f"{AUDIO_TRANSCRIPTION_SERVICE_URL}/test", timeout=5)
        if response.status_code == 200:
            result = response.json()
            services_status["audio"] = {
                "status": "available",
                "models": result.get("models", {})
            }
        else:
            services_status["audio"] = {"status": "error", "message": response.text}
    except Exception as e:
        services_status["audio"] = {"status": "error", "message": str(e)}
    
    # Check medical doc service
    try:
        response = requests.get(f"{MEDICAL_DOC_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            services_status["medical"] = {"status": "available"}
        else:
            services_status["medical"] = {"status": "error", "message": response.text}
    except Exception as e:
        services_status["medical"] = {"status": "error", "message": str(e)}
    
    return services_status

# Check services status
services_status = check_services()

# Display services status
with st.sidebar:
    st.header("Статус сервисов")
    
    # Audio service status
    st.subheader("Сервис транскрибации")
    audio_status = services_status["audio"]
    if audio_status["status"] == "available":
        st.success("✅ Доступен")
        if "models" in audio_status:
            st.markdown("**Доступные модели:**")
            for model, available in audio_status["models"].items():
                status = "✅" if available else "❌"
                st.markdown(f"- {model.upper()}: {status}")
    else:
        st.error(f"❌ Недоступен: {audio_status.get('message', 'Неизвестная ошибка')}")
    
    # Medical service status
    st.subheader("Медицинский сервис")
    medical_status = services_status["medical"]
    if medical_status["status"] == "available":
        st.success("✅ Доступен")
    else:
        st.error(f"❌ Недоступен: {medical_status.get('message', 'Неизвестная ошибка')}")

def transcribe_audio(audio_file):
    """Transcribe audio file using the transcription service"""
    try:
        files = {"file": (audio_file.name, audio_file, audio_file.type)}
        data = {
            "model_type": "rnnt",  # используем RNNT модель по умолчанию
            "long_form": "true"    # включаем поддержку длинных аудио
        }
        
        st.info(f"Отправка аудиофайла: {audio_file.name} ({audio_file.type})")
        
        response = requests.post(
            f"{AUDIO_TRANSCRIPTION_SERVICE_URL}/transcribe",
            files=files,
            data=data,
            timeout=300  # увеличиваем timeout для длинных аудио
        )
        
        if response.status_code == 200:
            result = response.json()
            # Проверяем формат ответа
            if "utterances" in result:
                # Для длинной транскрипции объединяем все сегменты
                return "\n".join([u["transcription"] for u in result["utterances"]])
            elif "transcription" in result:
                # Для обычной транскрипции
                return result["transcription"]
            else:
                st.error("Неожиданный формат ответа от сервиса транскрибации")
                return None
        else:
            st.error(f"Ошибка при транскрибации: {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера. Аудиофайл может быть слишком длинным.")
        return None
    except Exception as e:
        st.error(f"Ошибка при транскрибации: {str(e)}")
        return None

def process_transcript(text):
    """Process transcript into medical documentation"""
    try:
        # Создаем статус-контейнер для отображения прогресса
        status_container = st.empty()
        
        response = requests.post(
            f"{MEDICAL_DOC_SERVICE_URL}/process_transcript",
            json={
                "transcript": text,
                "model_type": selected_model,
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 8000
                }
            },
            stream=True,
            timeout=600  # 10 минут таймаут
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get("status")
                    
                    if status == "completed":
                        status_container.empty()
                        return data.get("structured_doc")
                    elif status == "error":
                        st.error(data.get("message"))
                        return None
                    else:
                        # Обновляем статус
                        with status_container:
                            st.info(data.get("message"))
        else:
            st.error(f"Ошибка при обработке транскрипта: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при обработке транскрипта: {str(e)}")
        return None

def analyze_medical_doc(medical_doc):
    """Analyze medical documentation and generate recommendations"""
    try:
        # Создаем статус-контейнер для отображения прогресса
        status_container = st.empty()
        
        response = requests.post(
            f"{MEDICAL_DOC_SERVICE_URL}/analyze",
            json={
                "medical_doc": medical_doc,
                "model_type": selected_model,
                "top_k": 3,
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 8000
                }
            },
            stream=True,
            timeout=600  # 10 минут таймаут
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get("status")
                    
                    if status == "completed":
                        status_container.empty()
                        return {
                            "recommendations": data.get("recommendations"),
                            "similar_documents": data.get("similar_documents")
                        }
                    elif status == "error":
                        st.error(data.get("message"))
                        return None
                    else:
                        # Обновляем статус
                        with status_container:
                            st.info(data.get("message"))
        else:
            st.error(f"Ошибка при анализе документа: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при анализе документа: {str(e)}")
        return None

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "medical_doc" not in st.session_state:
    st.session_state.medical_doc = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "similar_documents" not in st.session_state:
    st.session_state.similar_documents = None
if "doc_edited" not in st.session_state:
    st.session_state.doc_edited = False

# Main UI
st.title("🏥 Автоматизация медицинской документации")
st.markdown("""
    1. Загрузите аудиозапись приема
    2. Получите автоматическую транскрипцию
    3. Просмотрите и отредактируйте сгенерированную документацию
    4. Получите рекомендации на основе документации
""")

# File upload section
st.header("1. Загрузка файлов")
uploaded_files = st.file_uploader(
    "Загрузите аудио или видео файлы",
    type=['mp3', 'wav', 'm4a', 'mp4', 'avi', 'mov'],
    accept_multiple_files=True
)

# Process files and generate documentation
if uploaded_files:
    if st.button("🎯 Начать обработку"):
        with st.spinner("Обработка файлов..."):
            # List to store all audio segments
            audio_segments = []
            
            # Process each file
            for file in uploaded_files:
                file_ext = file.name.split('.')[-1].lower()
                
                if file_ext in ['mp4', 'avi', 'mov']:
                    st.info(f"Извлечение аудио из видео: {file.name}")
                    audio_segment = extract_audio_from_video(file)
                else:
                    st.info(f"Загрузка аудио файла: {file.name}")
                    audio_segment = load_audio_file(file)
                
                if audio_segment:
                    audio_segments.append(audio_segment)
                else:
                    st.error(f"Не удалось обработать файл: {file.name}")
                    break
            
            if audio_segments:
                # Combine all audio segments
                st.info("Объединение аудио файлов...")
                combined_audio = combine_audio_segments(audio_segments)
                
                if combined_audio:
                    # Convert to file-like object for transcription
                    audio_file = audio_segment_to_file(combined_audio)
                    
                    if audio_file:
                        with st.spinner("Транскрибация аудио..."):
                            transcription = transcribe_audio(audio_file)
                            if transcription:
                                st.session_state.transcription = transcription
                                
                                with st.spinner("Генерация медицинской документации..."):
                                    medical_doc = process_transcript(transcription)
                                    if medical_doc:
                                        st.session_state.medical_doc = medical_doc
                                        st.session_state.doc_edited = False
                                        st.rerun()

# Display and edit transcription
if st.session_state.transcription:
    st.header("2. Транскрипция")
    st.text_area(
        "Текст транскрипции",
        st.session_state.transcription,
        height=150,
        disabled=True
    )

# Display and edit medical documentation
if st.session_state.medical_doc:
    st.header("3. Медицинская документация")
    st.markdown("*Отредактируйте документацию при необходимости:*")
    
    edited_doc = st.text_area(
        "Медицинская документация",
        st.session_state.medical_doc,
        height=300,
        key="medical_doc_editor"
    )
    
    if edited_doc != st.session_state.medical_doc:
        st.session_state.medical_doc = edited_doc
        st.session_state.doc_edited = True
        st.session_state.recommendations = None  # Clear previous recommendations
        st.session_state.similar_documents = None

    # Generate recommendations button
    if st.button("🔍 Сгенерировать рекомендации", disabled=False):
        with st.spinner("Генерация рекомендаций..."):
            result = analyze_medical_doc(st.session_state.medical_doc)
            if result:
                st.session_state.recommendations = result["recommendations"]
                st.session_state.similar_documents = result["similar_documents"]
                st.rerun()

# Display recommendations and similar cases
if st.session_state.recommendations:
    st.header("4. Рекомендации")
    st.markdown(st.session_state.recommendations)
    
    if st.session_state.similar_documents:
        st.header("5. Похожие случаи")
        for idx, doc in enumerate(st.session_state.similar_documents, 1):
            with st.expander(f"Случай {idx} (Схожесть: {doc['similarity']:.2f})", expanded=False):
                st.markdown(doc['content'])
                if 'metadata' in doc:
                    st.markdown("**Метаданные:**")
                    if doc['metadata'].get('specialty'):
                        st.markdown(f"- Специальность: {doc['metadata']['specialty']}")
                    if doc['metadata'].get('diagnoses'):
                        st.markdown("- Диагнозы: " + ", ".join(doc['metadata']['diagnoses']))
    
    # Create download button for complete documentation
    complete_doc = f"""# Медицинская документация
    
{st.session_state.medical_doc}

# Рекомендации

{st.session_state.recommendations}

# Похожие случаи

"""
    st.download_button(
        label="📥 Скачать полную документацию",
        data=complete_doc,
        file_name=f"medical_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
