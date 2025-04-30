import streamlit as st
import requests
import json
from typing import Dict, Any
import os
from datetime import datetime

# Service URLs from environment variables
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8003")
MEDICAL_DOC_SERVICE_URL = os.getenv("MEDICAL_DOC_SERVICE_URL", "http://localhost:8000")
AUDIO_TRANSCRIPTION_SERVICE_URL = os.getenv("AUDIO_TRANSCRIPTION_SERVICE_URL", "http://localhost:8004")

# Configure the page
st.set_page_config(
    page_title="Медицинская RAG Система",
    page_icon="🏥",
    layout="wide"
)

def add_document(content: str, metadata: Dict[str, Any]) -> bool:
    """Add a document to the RAG service"""
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/documents",
            json={"content": content, "metadata": metadata}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error adding document: {str(e)}")
        return False

def get_documents():
    """Fetch all documents from the RAG service"""
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def delete_document(doc_id: str) -> bool:
    """Delete a document from the RAG service"""
    try:
        response = requests.delete(f"{RAG_SERVICE_URL}/documents/{doc_id}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def search_documents(query: str, top_k: int = 3):
    """Search for similar documents"""
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

# Functions for LLM service
def get_available_models():
    """Get available LLM models"""
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/available_models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching available models: {str(e)}")
        return {}

def generate_llm_response(prompt: str, model_type: str, parameters: Dict[str, Any] = None):
    """Generate response from LLM"""
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "model_type": model_type,
                "parameters": parameters
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def analyze_medical_doc(medical_doc: str, model_type: str = "openai", top_k: int = 3, parameters: Dict[str, Any] = None):
    """Analyze medical document using the medical doc service with streaming support"""
    try:
        with requests.post(
            f"{MEDICAL_DOC_SERVICE_URL}/analyze",
            json={
                "medical_doc": medical_doc,
                "model_type": model_type,
                "top_k": top_k,
                "parameters": parameters
            },
            stream=True,
            timeout=360  # 6 minutes timeout
        ) as response:
            response.raise_for_status()
            
            # Create placeholders for status messages and results
            status_placeholder = st.empty()
            result_container = st.container()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        status = data.get("status")
                        
                        if status in ["started", "searching", "preparing", "generating"]:
                            # Update status message with spinner
                            with status_placeholder:
                                with st.spinner(data["message"]):
                                    st.info(data["message"])
                        elif status == "completed":
                            # Clear status message and return results
                            status_placeholder.empty()
                            return data
                        elif status == "error":
                            # Show error message and stop processing
                            status_placeholder.error(data["message"])
                            return None
                    except json.JSONDecodeError as e:
                        st.error(f"Ошибка при обработке ответа сервера: {str(e)}")
                        return None
            
            return None
    except requests.Timeout:
        st.error("Превышено время ожидания ответа от сервера. Пожалуйста, попробуйте еще раз.")
        return None
    except requests.RequestException as e:
        st.error(f"Ошибка при анализе документа: {str(e)}")
        return None

# Audio transcription functions
def get_available_audio_models():
    """Get available audio transcription models"""
    try:
        response = requests.get(f"{AUDIO_TRANSCRIPTION_SERVICE_URL}/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching available audio models: {str(e)}")
        return {}

def transcribe_audio(audio_file, model_type="rnnt", long_form=False):
    """Transcribe audio file using the audio transcription service"""
    try:
        # Создаем файлы и параметры
        files = {"file": (audio_file.name, audio_file, audio_file.type)}
        
        # Отображаем информацию о запросе
        st.info(f"Отправка аудиофайла: {audio_file.name} ({audio_file.type})")
        st.info(f"Модель: {model_type}, Длинное аудио: {long_form}")
        
        # Выполняем запрос
        response = requests.post(
            f"{AUDIO_TRANSCRIPTION_SERVICE_URL}/transcribe",
            files=files,
            data={
                "model_type": model_type, 
                "long_form": "true" if long_form else "false"
            },
            timeout=300  # Увеличиваем timeout до 5 минут для длинных аудио
        )
        
        # Проверяем статус ответа
        if response.status_code != 200:
            st.error(f"Ошибка сервера: {response.status_code}")
            error_text = response.text
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_text = error_json["error"]
            except:
                pass
            st.error(f"Текст ошибки: {error_text}")
            return None
        
        # Возвращаем результат
        result = response.json()
        
        # Проверяем, какой тип результата мы получили
        if long_form and "transcription" in result:
            st.warning("Длинная транскрипция не была выполнена, вместо этого была использована обычная транскрипция")
        
        return result
    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера. Аудиофайл может быть слишком длинным.")
        return None
    except Exception as e:
        st.error(f"Ошибка при расшифровке: {str(e)}")
        return None

def test_audio_service():
    """Test if the audio transcription service is responsive"""
    try:
        response = requests.get(f"{AUDIO_TRANSCRIPTION_SERVICE_URL}/test")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error testing audio service: {str(e)}")
        return None

# Main UI
st.title("Медицинская RAG Система")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["База знаний", "Анализ документов", "Расшифровка аудио"])

# RAG System Tab
with tab1:
    st.header("База медицинских знаний")
    
    # Create three columns for the main layout
    doc_col, list_col, search_col = st.columns([1, 1, 1])
    
    # Document Creation Column
    with doc_col:
        st.subheader("Добавить новый документ")
        doc_content = st.text_area("Текст документа", height=150)
        
        # Separate inputs for metadata
        st.subheader("Метаданные документа")
        
        # Date input
        doc_date = st.date_input("Дата документа", datetime.now())
        
        # Medical specialty selection
        specialties = [
            "Терапия", "Кардиология", "Неврология", "Хирургия", 
            "Педиатрия", "Онкология", "Гинекология", "Урология",
            "Эндокринология", "Офтальмология", "Психиатрия", "Другое"
        ]
        specialty = st.selectbox("Область медицины", specialties)
        
        # Document type selection
        doc_types = [
            "Клинический случай", "Протокол лечения", 
            "Результаты исследования", "Рекомендации",
            "Заключение специалиста", "Другое"
        ]
        doc_type = st.selectbox("Тип документа", doc_types)
        
        # Multi-select for diagnoses
        diagnoses = st.text_area(
            "Диагнозы (каждый с новой строки)",
            height=100,
            help="Введите каждый диагноз с новой строки"
        )
        
        # Custom tags
        tags = st.text_area(
            "Дополнительные теги (каждый с новой строки)",
            height=100,
            help="Введите каждый тег с новой строки"
        )
        
        if st.button("Добавить документ"):
            try:
                # Process inputs
                diagnoses_list = [d.strip() for d in diagnoses.split('\n') if d.strip()]
                tags_list = [t.strip() for t in tags.split('\n') if t.strip()]
                
                metadata = {
                    "type": doc_type,
                    "specialty": specialty,
                    "date": doc_date.strftime("%Y-%m-%d"),
                    "diagnoses": diagnoses_list,
                    "tags": tags_list
                }
                
                if doc_content and add_document(doc_content, metadata):
                    st.success("Документ успешно добавлен!")
                    st.rerun()
            except Exception as e:
                st.error(f"Ошибка при добавлении документа: {str(e)}")

    # Document List Column
    with list_col:
        st.subheader("База документов")
        documents = get_documents()
        
        for doc in documents:
            with st.expander(f"Документ от {doc['metadata'].get('date', 'Дата не указана')}", expanded=False):
                st.text_area("Содержание", doc['content'], height=100, disabled=True, key=f"content_{doc['id']}")
                
                # Display metadata in a more readable format
                st.markdown("**Область медицины:** " + doc['metadata'].get('specialty', 'Не указано'))
                st.markdown("**Тип документа:** " + doc['metadata'].get('type', 'Не указано'))
                
                if doc['metadata'].get('diagnoses'):
                    st.markdown("**Диагнозы:**")
                    for d in doc['metadata']['diagnoses']:
                        st.markdown(f"- {d}")
                
                if doc['metadata'].get('tags'):
                    st.markdown("**Теги:**")
                    for t in doc['metadata']['tags']:
                        st.markdown(f"- {t}")
                
                if st.button("Удалить", key=f"del_{doc['id']}"):
                    if delete_document(doc['id']):
                        st.success("Документ удален!")
                        st.rerun()

    # Search Column
    with search_col:
        st.subheader("Семантический поиск")
        search_query = st.text_area("Введите поисковый запрос", height=100, key="search_query")
        top_k = st.slider("Количество результатов", min_value=1, max_value=10, value=3)
        
        if st.button("Поиск"):
            if search_query:
                results = search_documents(search_query, top_k)
                
                if results:
                    st.subheader("Результаты поиска")
                    for idx, doc in enumerate(results, 1):
                        with st.expander(f"Результат {idx} (Релевантность: {doc['similarity']:.3f})", expanded=True):
                            st.text_area("Содержание", doc['content'], height=100, disabled=True, key=f"result_{idx}")
                            
                            # Display metadata in a more readable format
                            st.markdown("**Область медицины:** " + doc['metadata'].get('specialty', 'Не указано'))
                            st.markdown("**Тип документа:** " + doc['metadata'].get('type', 'Не указано'))
                            
                            if doc['metadata'].get('diagnoses'):
                                st.markdown("**Диагнозы:**")
                                for d in doc['metadata']['diagnoses']:
                                    st.markdown(f"- {d}")
                            
                            if doc['metadata'].get('tags'):
                                st.markdown("**Теги:**")
                                for t in doc['metadata']['tags']:
                                    st.markdown(f"- {t}")
            else:
                st.warning("Пожалуйста, введите поисковый запрос")

# # LLM Chat Tab
# with tab2:
#     st.header("Чат с ассистентом")
    
#     # Get available models
#     available_models = get_available_models()
    
#     # Model selection
#     model_type = st.selectbox(
#         "Выберите модель ИИ",
#         options=list(available_models.keys()) if available_models else ["openai", "local"],
#         help="Выберите модель для генерации ответов"
#     )
    
#     # Chat interface
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if prompt := st.chat_input("Спросите меня что-нибудь"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Generate response
#         with st.chat_message("assistant"):
#             st.markdown(generate_llm_response(prompt, model_type))

# Medical Analysis Tab
with tab2:
    st.header("Анализ медицинских документов")
    
    # Get available models for selection
    available_models = get_available_models()
    
    # Input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        medical_doc = st.text_area(
            "Введите медицинский документ",
            height=300,
            help="Вставьте текст медицинского документа для анализа"
        )
    
    with col2:
        st.subheader("Настройки анализа")
        model_type = st.selectbox(
            "Выберите модель ИИ",
            options=list(available_models.keys()) if available_models else ["openai", "local"],
            help="Выберите модель для анализа"
        )
        
        top_k = st.slider(
            "Количество похожих случаев",
            min_value=1,
            max_value=10,
            value=3,
            help="Количество похожих случаев для анализа"
        )
        
        analyze_button = st.button("Анализировать")
    
    # Analysis results
    if analyze_button and medical_doc:
        result = analyze_medical_doc(medical_doc, model_type, top_k)
        
        if result:
            # Display recommendations
            st.subheader("Анализ и рекомендации")
            st.markdown(result["recommendations"])
            
            # Display similar cases
            st.subheader("Похожие случаи")
            for idx, doc in enumerate(result["similar_documents"], 1):
                with st.expander(f"Случай {idx} (Схожесть: {doc['similarity']:.3f})", expanded=False):
                    st.text_area("Содержание", doc['content'], height=100, disabled=True)
                    if "metadata" in doc:
                        # Display metadata in a more readable format
                        st.markdown("**Область медицины:** " + doc['metadata'].get('specialty', 'Не указано'))
                        st.markdown("**Тип документа:** " + doc['metadata'].get('type', 'Не указано'))
                        
                        if doc['metadata'].get('diagnoses'):
                            st.markdown("**Диагнозы:**")
                            for d in doc['metadata']['diagnoses']:
                                st.markdown(f"- {d}")
                        
                        if doc['metadata'].get('tags'):
                            st.markdown("**Теги:**")
                            for t in doc['metadata']['tags']:
                                st.markdown(f"- {t}")
            
            # Show model information
            with st.expander("Информация о модели"):
                st.json(result["model_info"])
    
    elif analyze_button:
        st.warning("Пожалуйста, введите текст документа для анализа")

# Audio Transcription Tab
with tab3:
    st.header("Расшифровка аудио")
    
    # Get available models
    audio_models = get_available_audio_models()
    
    # Check service status
    service_col, model_col = st.columns([1, 1])
    with service_col:
        if st.button("⚡ Проверка доступности сервиса"):
            test_result = test_audio_service()
            if test_result:
                st.success(f"✅ Сервис доступен! Статус: {test_result['status']}")
                st.write("Доступные модели:", test_result["models"])
            else:
                st.error("❌ Сервис недоступен! Пожалуйста, проверьте подключение.")
    
    # Two columns layout: upload and settings on the left, results on the right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Загрузить аудиофайл")
        uploaded_file = st.file_uploader("Выберите аудиофайл", type=['mp3', 'wav', 'ogg', 'm4a'])
        
        # Model selection
        model_options = {}
        if audio_models:
            if audio_models.get("ctc", False):
                model_options["GigaAM CTC (расшифровка аудио)"] = "ctc"
            if audio_models.get("rnnt", False):
                model_options["GigaAM RNNT (лучшая модель)"] = "rnnt"
        
        if not model_options:
            model_options = {"GigaAM RNNT (недоступно)": "rnnt", "GigaAM CTC (недоступно)": "ctc"}
        
        selected_model_name = st.selectbox(
            "Выберите модель для расшифровки", 
            options=list(model_options.keys()),
            index=0 if "GigaAM RNNT (лучшая модель)" in model_options else 0
        )
        selected_model = model_options[selected_model_name]
        
        # Display model information
        st.info("""
        **О моделях:**
        - **RNNT**: Лучшая модель для распознавания русской речи с наименьшим WER
        - **CTC**: Альтернативная модель с более быстрой работой
        
        Обе модели основаны на GigaAM-v2 фреймворке от Salute Developers.
        """)
        
        # Long-form option
        use_long_form = st.checkbox(
            "Использовать расшифровку длинных аудио",
            value=False,
            help="Включите для аудио длительностью более 30 секунд"
        )
        
        if use_long_form:
            st.warning("""
            Для расшифровки длинных аудио используется Voice Activity Detection для разделения аудио на сегменты.
            Результат будет содержать временные метки для каждого сегмента.
            """)
        
        # Transcribe button - give it a unique key to avoid conflicts
        transcribe_button = st.button("🎙️ Расшифровать аудио", key="transcribe_audio_btn")
    
    with col2:
        st.subheader("Результат расшифровки")
        
        if uploaded_file is not None and transcribe_button:
            with st.spinner("Выполняется расшифровка аудио..."):
                # Call the transcription service
                result = transcribe_audio(
                    uploaded_file,
                    model_type=selected_model,
                    long_form=use_long_form
                )
                
                if result:
                    st.success("Расшифровка завершена успешно!")
                    
                    # Display results based on the type (long-form or regular)
                    if "utterances" in result:  # Long-form result
                        st.markdown("### Расшифровка по сегментам")
                        
                        for i, utterance in enumerate(result["utterances"]):
                            start_time = utterance["boundaries"][0]
                            end_time = utterance["boundaries"][1]
                            
                            # Format times as MM:SS
                            start_formatted = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
                            end_formatted = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"
                            
                            # Display segment with time markers
                            st.markdown(f"**[{start_formatted} - {end_formatted}]** {utterance['transcription']}")
                    else:  # Regular transcription
                        st.markdown("### Текст расшифровки")
                        st.markdown(result["transcription"])
                    
                    # Add copy button for the transcription
                    if "utterances" in result:
                        full_text = "\n".join([u["transcription"] for u in result["utterances"]])
                    else:
                        full_text = result["transcription"]
                    
                    st.text_area("Копировать текст", full_text, height=150)
                else:
                    st.error("Не удалось выполнить расшифровку аудио.")
            
            # Sample display
            st.markdown("### Пример результата расшифровки")
            st.markdown("""
            **[00:00 - 00:05]** Добрый день, сегодня мы обсудим новые методы лечения.
            
            **[00:06 - 00:12]** Данная методика показала высокую эффективность при клинических испытаниях.
            
            **[00:13 - 00:20]** Результаты были опубликованы в последнем выпуске медицинского журнала.
            """) 