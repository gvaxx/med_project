import os
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# model = "mxbai-embed-large-v1"
# model_name = f"mixedbread-ai/{model}"

model = "paraphrase-multilingual-mpnet-base-v2"
model_name = f"sentence-transformers/{model}"
class RAGEvaluator:
    def __init__(self):
        self.temp_dir = None
        self.chroma_client = None
        self.collection = None
        self.ef = None
        
    def setup(self):
        """Initialize temporary ChromaDB instance"""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.temp_dir, "chroma_db"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # Initialize embedding function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Create collection
        self.collection = self.chroma_client.create_collection(
            name="medical_docs",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaDB instance initialized")
    
    def prepare_evaluation_data(self, tsv_path):
        """Prepare evaluation data by selecting diagnoses and cases"""
        # Read TSV file
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
        logger.info(f"Loaded {len(df)} records from TSV")
        
        # Group by ICD-10 codes
        diagnosis_groups = df.groupby('icd10')
        
        # Select 10 diagnoses with most cases
        top_diagnoses = diagnosis_groups.size().nlargest(10).index.tolist()
        logger.info(f"Selected top 10 diagnoses: {top_diagnoses}")
        
        # Prepare seed and test cases
        seed_cases = []
        test_cases = []
        
        for diagnosis in top_diagnoses:
            diagnosis_df = diagnosis_groups.get_group(diagnosis)
            
            # Randomly select 5 cases for seed
            seed_df = diagnosis_df.sample(n=5, random_state=42)
            seed_cases.append(seed_df)
            
            # Use remaining cases for testing
            test_df = diagnosis_df.drop(seed_df.index)
            test_cases.append(test_df)
        
        return pd.concat(seed_cases), pd.concat(test_cases)
    
    def load_documents(self, df):
        """Load documents from DataFrame into ChromaDB"""
        try:
            # Prepare documents for insertion
            ids = [str(i) for i in range(len(df))]
            
            # Combine relevant columns into text
            texts = []
            metadatas = []
            
            for _, row in df.iterrows():
                # Combine relevant text fields
                text_parts = []
                if 'symptoms' in df.columns:
                    text_parts.append(str(row['symptoms']))
                if 'anamnesis' in df.columns:
                    text_parts.append(str(row['anamnesis']))
                
                text = " ".join(filter(None, text_parts))
                texts.append(text)
                
                # Prepare metadata
                metadata = {
                    "icd10": str(row.get('icd10', '')),
                    "patient_id": str(row.get('new_patient_id', '')),
                    "event_id": str(row.get('new_event_id', '')),
                    "event_time": str(row.get('new_event_time', '')),
                    "doc_type": "case"
                }
                metadatas.append(metadata)
            
            # Add documents to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Loaded {len(texts)} documents into ChromaDB")
            return len(texts)
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return 0
    
    def evaluate_queries(self, test_df):
        """Evaluate RAG performance on test cases"""
        hits = []
        reciprocal_ranks = []
        query_times = []
        diagnosis_metrics = defaultdict(lambda: {"hits": [], "ranks": [], "times": []})
        
        for _, row in tqdm(test_df.iterrows(), desc="Evaluating queries", total=len(test_df)):
            # Prepare query text
            text_parts = []
            if 'symptoms' in test_df.columns:
                text_parts.append(str(row['symptoms']))
            if 'anamnesis' in test_df.columns:
                text_parts.append(str(row['anamnesis']))
            query = " ".join(filter(None, text_parts))
            
            # Search in ChromaDB and measure time
            start_time = time.time()
            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=["metadatas", "distances"]
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Extract ICD-10 codes from results
            codes = [r["icd10"] for r in results["metadatas"][0]]
            gold_code = str(row['icd10'])
            
            # Calculate metrics
            try:
                rank = codes.index(gold_code) + 1
                hits.append(1)
                reciprocal_ranks.append(1 / rank)
                diagnosis_metrics[gold_code]["hits"].append(1)
                diagnosis_metrics[gold_code]["ranks"].append(rank)
                diagnosis_metrics[gold_code]["times"].append(query_time)
            except ValueError:
                hits.append(0)
                reciprocal_ranks.append(0)
                diagnosis_metrics[gold_code]["hits"].append(0)
                diagnosis_metrics[gold_code]["ranks"].append(0)
                diagnosis_metrics[gold_code]["times"].append(query_time)
        
        # Calculate final metrics
        recall_at_5 = np.mean(hits)
        mrr = np.mean(reciprocal_ranks)
        mean_query_time = np.mean(query_times)
        
        # Calculate per-diagnosis metrics
        diagnosis_results = {}
        for diagnosis, metrics in diagnosis_metrics.items():
            diagnosis_results[diagnosis] = {
                "recall_at_5": np.mean(metrics["hits"]),
                "mrr": np.mean([1/r if r > 0 else 0 for r in metrics["ranks"]]),
                "num_cases": len(metrics["hits"]),
                "mean_query_time": np.mean(metrics["times"])
            }
        
        return {
            "overall": {
                "recall_at_5": recall_at_5,
                "mrr": mrr,
                "total_queries": len(test_df),
                "mean_query_time": mean_query_time
            },
            "by_diagnosis": diagnosis_results
        }
    
    def cleanup(self):
        """Clean up temporary files and ChromaDB instance"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

def main():
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    try:
        # Setup ChromaDB
        evaluator.setup()
        
        # Load and prepare data
        tsv_path = "data/RuMedPrimeData.tsv"
        seed_df, test_df = evaluator.prepare_evaluation_data(tsv_path)
        
        # Load seed documents into ChromaDB
        num_docs = evaluator.load_documents(seed_df)
        
        if num_docs == 0:
            logger.error("No documents loaded, exiting")
            return
        
        # Run evaluation
        results = evaluator.evaluate_queries(test_df)
        
        # Print results
        print("\nOverall Evaluation Results:")
        print(f"Recall@5: {results['overall']['recall_at_5']:.4f}")
        print(f"MRR: {results['overall']['mrr']:.4f}")
        print(f"Mean query time: {results['overall']['mean_query_time']:.4f} seconds")
        print(f"Total queries: {results['overall']['total_queries']}")
        
        print("\nPer-Diagnosis Results:")
        for diagnosis, metrics in results['by_diagnosis'].items():
            print(f"\nDiagnosis {diagnosis}:")
            print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
            print(f"  MRR: {metrics['mrr']:.4f}")
            print(f"  Mean query time: {metrics['mean_query_time']:.4f} seconds")
            print(f"  Number of test cases: {metrics['num_cases']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data/rag_results_{timestamp}_{model}.json"
        with open(results_file, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
    finally:
        # Cleanup
        evaluator.cleanup()

if __name__ == "__main__":
    main()