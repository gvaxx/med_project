import pandas as pd
import httpx
import asyncio
import json
from typing import Dict, Any, List
import logging
from pathlib import Path
import time
import re
from rouge_score import rouge_scorer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LLM_SERVICE_URL = "http://localhost:8003"  # Adjust if needed
DATA_PATH = Path("data/Cleaned_Medical_Recommendations_Enriched.csv")
RESULTS_DIR = Path("results/llm_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "gigachat": {
        "temperature": 0.25,
        "max_tokens": 1000
    },
    "openai": {
        "temperature": 0.25,
        "max_tokens": 1000
    },
    "local": {
        "temperature": 0.25,
        "max_tokens": 1000,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "model": "saiga2_7b"  # Указываем конкретную модель
    }
}

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge_scores(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate ROUGE scores between reference and candidate texts"""
    scores = rouge.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_structure(text: str) -> Dict[str, Any]:
    """Evaluate the structure of recommendations"""
    # Check for bullet points or numbered lists
    has_bullets = bool(re.search(r'[•\-\*]|\d+\.', text))
    
    # Count number of distinct recommendations
    recommendations = re.split(r'[•\-\*]|\d+\.', text)
    recommendations = [r.strip() for r in recommendations if r.strip()]
    
    return {
        "has_bullets": has_bullets,
        "num_recommendations": len(recommendations),
        "avg_recommendation_length": sum(len(r) for r in recommendations) / len(recommendations) if recommendations else 0
    }

async def generate_llm_response(
    client: httpx.AsyncClient,
    prompt: str,
    model_type: str,
    temperature: float = 0.25
) -> Dict[str, Any]:
    """Generate response from LLM service"""
    try:
        # Get model-specific parameters
        model_params = MODEL_CONFIGS.get(model_type, {}).copy()
        model_params["temperature"] = temperature
        
        response = await client.post(
            f"{LLM_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "model_type": model_type,
                "parameters": model_params
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None

async def evaluate_models():
    """Evaluate LLM models on medical recommendations dataset"""
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Prepare results storage
    results = {
        "gigachat": [],
        # "openai": [],
        # "local": []
    }
    
    # Create prompt template
    prompt_template = """На основании следующего медицинского случая предоставьте соответствующие рекомендации:

Случай:
{case}

Пожалуйста, предоставьте рекомендации в четком, структурированном формате."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, row in df.iterrows():
            case = row["symptom_history"]
            true_recommendations = row["recommendations"]
            
            # Create prompt
            prompt = prompt_template.format(case=case)
            
            # Test each model
            for model_type in ["gigachat"]:
                logger.info(f"Testing {model_type} on case {idx + 1}/{len(df)}")
                
                # Generate response
                start_time = time.time()
                response = await generate_llm_response(
                    client=client,
                    prompt=prompt,
                    model_type=model_type
                )
                end_time = time.time()
                
                if response:
                    generated_recommendations = response["response"]
                    print(generated_recommendations)
                    # Calculate metrics
                    rouge_scores = calculate_rouge_scores(
                        true_recommendations,
                        generated_recommendations
                    )
                    
                    structure_metrics = evaluate_structure(generated_recommendations)
                    
                    results[model_type].append({
                        "case_id": row["new_event_id"],
                        "case": case,
                        "true_recommendations": true_recommendations,
                        "generated_recommendations": generated_recommendations,
                        "model_info": response["model_info"],
                        "generation_time": end_time - start_time,
                        "metrics": {
                            "rouge": rouge_scores,
                            "structure": structure_metrics
                        }
                    })
                
                # Add small delay between requests
                await asyncio.sleep(1)
    # Calculate aggregate metrics
    for model_type in results:
        if results[model_type]:
            rouge1_scores = [r["metrics"]["rouge"]["rouge1"] for r in results[model_type]]
            rouge2_scores = [r["metrics"]["rouge"]["rouge2"] for r in results[model_type]]
            rougeL_scores = [r["metrics"]["rouge"]["rougeL"] for r in results[model_type]]
            generation_times = [r["generation_time"] for r in results[model_type]]
            has_bullets = [r["metrics"]["structure"]["has_bullets"] for r in results[model_type]]
            
            aggregate_metrics = {
                "rouge": {
                    "rouge1": {
                        "mean": np.mean(rouge1_scores),
                        "std": np.std(rouge1_scores)
                    },
                    "rouge2": {
                        "mean": np.mean(rouge2_scores),
                        "std": np.std(rouge2_scores)
                    },
                    "rougeL": {
                        "mean": np.mean(rougeL_scores),
                        "std": np.std(rougeL_scores)
                    }
                },
                "generation_time": {
                    "mean": np.mean(generation_times),
                    "std": np.std(generation_times)
                },
                "structure": {
                    "percent_with_bullets": sum(has_bullets) / len(has_bullets) * 100
                }
            }
            
            results[model_type].append({"aggregate_metrics": aggregate_metrics})
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for model_type, model_results in results.items():
        output_file = RESULTS_DIR / f"{model_type}_eval_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(model_results)} results for {model_type} to {output_file}")

if __name__ == "__main__":
    asyncio.run(evaluate_models()) 