import asyncio
import httpx
import json
from pathlib import Path

async def test_transcript_processing():
    # Read the dialog file
    dialog_path = Path("dialog.txt")
    if not dialog_path.exists():
        print("Error: dialog.txt file not found")
        return
    
    with open(dialog_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    # Prepare the request
    url = "http://localhost:8000/process_transcript"  # Adjust port if needed
    data = {
        "transcript": transcript,
        "model_type": "openai",
        "parameters": {
            "temperature": 0.3,
            "max_tokens": 4000
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600.0)) as client:
            print("Sending transcript for processing...")
            async with client.stream("POST", url, json=data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    update = json.loads(line)
                    if update.get("status") == "error":
                        print(f"Error: {update.get('message')}")
                        return
                    
                    if update.get("status") == "completed":
                        print("\nGenerated medical documentation:")
                        print("-" * 80)
                        print(update.get("structured_doc"))
                        print("-" * 80)
                        return
                    
                    print(f"Status: {update.get('status')} - {update.get('message')}")
                    
    except httpx.HTTPError as e:
        print(f"HTTP Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_transcript_processing()) 