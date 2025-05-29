from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import os
import json
import asyncio

app = FastAPI(title="Medical Document Analysis Service")

# Service URLs from environment variables
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-doc-service:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8003")

class MedicalDocRequest(BaseModel):
    medical_doc: str
    model_type: str = "openai"
    top_k: int = 3
    parameters: Optional[Dict[str, Any]] = None

class MedicalDocResponse(BaseModel):
    recommendations: str
    similar_documents: List[Dict[str, Any]]
    model_info: Dict[str, Any]

class TranscriptRequest(BaseModel):
    transcript: str
    model_type: str = "openai"
    parameters: Optional[Dict[str, Any]] = None

class TranscriptResponse(BaseModel):
    structured_doc: str
    model_info: Dict[str, Any]

def create_recommendation_prompt(medical_doc: str, similar_docs: List[Dict[str, Any]]) -> str:
    """Создает промпт для LLM, фокусируясь на генерации рекомендаций на основе похожих случаев"""
    prompt_parts = [
        "Вы - опытный врач. Проанализируйте предоставленный медицинский случай и похожие случаи из базы данных. "
        "Ваша задача - предоставить рекомендации, основываясь на текущем случае и учитывая опыт похожих случаев.\n\n",
        
        "Текущий медицинский случай:\n",
        medical_doc,
        "\n\nПохожие случаи из базы данных:\n"
    ]
    
    for idx, doc in enumerate(similar_docs, 1):
        prompt_parts.append(f"\nСлучай {idx} (Схожесть: {doc['similarity']:.2f}):\n")
        prompt_parts.append(f"Содержание: {doc['content']}\n")
        
        # Добавляем информацию о диагнозах и специальности, если она есть
        if 'metadata' in doc:
            if doc['metadata'].get('diagnoses'):
                prompt_parts.append("Диагнозы: " + ", ".join(doc['metadata']['diagnoses']) + "\n")
            if doc['metadata'].get('specialty'):
                prompt_parts.append(f"Специальность: {doc['metadata']['specialty']}\n")
    
    prompt_parts.append(
        "\nНа основе предоставленной информации, пожалуйста, составьте структурированные рекомендации, включающие:\n"
        "1. Рекомендации по лечению, основанные на опыте похожих случаев\n"
        "2. Рекомендации по профилактике осложнений\n"
        "3. Рекомендации по образу жизни и реабилитации\n"
        "4. Особые указания и предостережения\n\n"
        "Пожалуйста, учитывайте:\n"
        "- Опыт лечения в похожих случаях\n"
        "- Возможные осложнения, наблюдавшиеся в похожих случаях\n"
        "- Успешные методы лечения из похожих случаев\n"
        "- Индивидуальные особенности текущего случая\n"
    )
    
    return "\n".join(prompt_parts)

def create_transcript_prompt(transcript: str) -> str:
    """Creates a prompt for processing medical transcript into structured documentation"""
    return f"""На основе следующего диалога между родителем и врачом составь подробное медицинское заключение в стиле психиатрического протокола. Структура документа должна быть следующей:

1. Дата приёма  
2. Имя пациента  
3. Возраст  
4. ФИО врача (если есть)  
5. Жалобы  
6. Цель консультации  
7. Анамнез заболевания  
8. Психический статус  
9. Обоснование диагноза  
10. Клинический диагноз (используй мкб нотацию)  
11. Сопутствующие диагнозы (если не указано — напиши "не выявлено")  
12. План обследования  

Пиши в официально-медицинском, нейтральном стиле. Избегай разговорных выражений. В каждой части используй информацию строго из диалога, не придумывай. Если чего-то нет — пропусти или отметь как "не указано".

Вот диалог:
{transcript}"""

async def search_similar_documents(medical_doc: str, top_k: int) -> List[Dict[str, Any]]:
    """Поиск похожих документов через RAG сервис"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=60.0)) as client:
        try:
            response = await client.post(
                f"{RAG_SERVICE_URL}/search",
                json={"query": medical_doc, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при поиске похожих документов: {str(e)}")

async def generate_llm_recommendations(prompt: str, model_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Генерация рекомендаций через LLM сервис"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600.0)) as client:  # 5 minutes timeout
        try:
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={
                    "prompt": prompt,
                    "model_type": model_type,
                    "parameters": parameters or {
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            if isinstance(e, httpx.TimeoutException):
                raise HTTPException(status_code=504, detail="Превышено время ожидания ответа от LLM сервиса")
            raise HTTPException(status_code=500, detail=f"Ошибка при генерации рекомендаций: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Medical Document Analysis Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_medical_doc(request: MedicalDocRequest):
    async def generate_analysis():
        try:
            # 1. Send initial status
            yield json.dumps({"status": "started", "message": "Начало анализа..."}) + "\n"
            await asyncio.sleep(0.1)
            
            # 2. Search for similar documents
            yield json.dumps({"status": "searching", "message": "Поиск похожих документов..."}) + "\n"
            try:
                similar_docs = await search_similar_documents(request.medical_doc, request.top_k)
                await asyncio.sleep(0.1)
            except Exception as e:
                yield json.dumps({
                    "status": "error",
                    "message": f"Ошибка при поиске документов: {str(e)}"
                }) + "\n"
                return
            
            # 3. Create recommendation prompt
            yield json.dumps({"status": "preparing", "message": "Подготовка анализа..."}) + "\n"
            recommendation_prompt = create_recommendation_prompt(request.medical_doc, similar_docs)
            await asyncio.sleep(0.1)
            
            # 4. Generate recommendations
            yield json.dumps({"status": "generating", "message": "Генерация рекомендаций... Это может занять несколько минут."}) + "\n"
            try:
                llm_response = await generate_llm_recommendations(
                    recommendation_prompt,
                    request.model_type,
                    request.parameters
                )
            except Exception as e:
                if isinstance(e, HTTPException) and e.status_code == 504:
                    yield json.dumps({
                        "status": "error",
                        "message": "Превышено время ожидания ответа от модели. Пожалуйста, попробуйте еще раз или используйте другую модель."
                    }) + "\n"
                else:
                    yield json.dumps({
                        "status": "error",
                        "message": f"Ошибка при генерации рекомендаций: {str(e)}"
                    }) + "\n"
                return
            
            # 5. Send final response
            final_response = {
                "status": "completed",
                "recommendations": llm_response["response"],
                "similar_documents": similar_docs,
                "model_info": llm_response["model_info"]
            }
            yield json.dumps(final_response) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "status": "error",
                "message": f"Ошибка при анализе документа: {str(e)}"
            }) + "\n"

    return StreamingResponse(
        generate_analysis(),
        media_type="application/x-ndjson"
    )

@app.post("/process_transcript")
async def process_transcript(request: TranscriptRequest):
    async def generate_documentation():
        try:
            # 1. Send initial status
            yield json.dumps({"status": "started", "message": "Начало обработки транскрипта..."}) + "\n"
            await asyncio.sleep(0.1)
            
            # 2. Create prompt
            yield json.dumps({"status": "preparing", "message": "Подготовка анализа..."}) + "\n"
            transcript_prompt = create_transcript_prompt(request.transcript)
            await asyncio.sleep(0.1)
            
            # 3. Generate structured documentation
            yield json.dumps({"status": "generating", "message": "Генерация медицинского заключения... Это может занять несколько минут."}) + "\n"
            try:
                llm_response = await generate_llm_recommendations(
                    transcript_prompt,
                    request.model_type,
                    request.parameters or {
                        "temperature": 0.3,
                        "max_tokens": 8000
                    }
                )
            except Exception as e:
                if isinstance(e, HTTPException) and e.status_code == 504:
                    yield json.dumps({
                        "status": "error",
                        "message": "Превышено время ожидания ответа от модели. Пожалуйста, попробуйте еще раз или используйте другую модель."
                    }) + "\n"
                else:
                    yield json.dumps({
                        "status": "error",
                        "message": f"Ошибка при генерации заключения: {str(e)}"
                    }) + "\n"
                return
            
            # 4. Send final response
            final_response = {
                "status": "completed",
                "structured_doc": llm_response["response"],
                "model_info": llm_response["model_info"]
            }
            yield json.dumps(final_response) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "status": "error",
                "message": f"Ошибка при обработке транскрипта: {str(e)}"
            }) + "\n"

    return StreamingResponse(
        generate_documentation(),
        media_type="application/x-ndjson"
    ) 