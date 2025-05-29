import argparse
import csv
import requests
import jiwer
from tqdm import tqdm
import os
import soundfile as sf
import json
from datetime import datetime

def evaluate_model(model_name, audio_dir, meta_file):
    truth, hypothesis = [], []
    
    # Read metadata
    with open(meta_file, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            wav_file = row['filename']
            ref_text = row['text']
            
            # Read audio file
            audio_path = os.path.join(audio_dir, wav_file)
            
            # Send to transcription service
            try:
                with open(audio_path, 'rb') as audio_file:
                    files = {'file': (wav_file, audio_file, 'audio/wav')}
                    data = {
                        'model_type': model_name,
                        'long_form': 'false'
                    }
                    r = requests.post(
                        "http://localhost:8004/transcribe",
                        files=files,
                        data=data,
                        timeout=600
                    )
                r.raise_for_status()
                hyp_text = r.json()['transcription']
                
                truth.append(ref_text.lower())
                hypothesis.append(hyp_text.lower())
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
                continue
    
    # Calculate metrics
    wer = jiwer.wer(truth, hypothesis)
    cer = jiwer.cer(truth, hypothesis)
    
    return {
        "model": model_name,
        "wer": wer,
        "cer": cer,
        "samples": len(truth)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR models on medical speech')
    parser.add_argument('--models', nargs='+', default=['ctc', 'rnnt', 'whisperx'],
                      help='List of models to evaluate')
    args = parser.parse_args()
    
    audio_dir = "data/asr_data"
    meta_file = os.path.join(audio_dir, "meta.csv")
    
    results = []
    for model in args.models:
        print(f"\nEvaluating {model}...")
        result = evaluate_model(model, audio_dir, meta_file)
        results.append(result)
        
        print(f"WER: {result['wer']:.4f}")
        print(f"CER: {result['cer']:.4f}")
        print(f"Samples processed: {result['samples']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/asr_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()