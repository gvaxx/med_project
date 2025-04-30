import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Get the duration of an audio file in seconds
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds or None if there was an error
    """
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        return None

def convert_audio_format(input_path: str, target_format: str = "wav") -> Optional[str]:
    """
    Convert audio to a specified format
    
    Args:
        input_path: Path to the input audio file
        target_format: The desired output format (default: wav)
        
    Returns:
        Path to the converted file or None if conversion failed
    """
    try:
        # Create a temporary file with the target extension
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"converted.{target_format}")
        
        # Run ffmpeg to convert the file
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    except Exception as e:
        logger.error(f"Error converting audio format: {str(e)}")
        return None

def is_valid_audio_file(file_path: str) -> bool:
    """
    Check if the file is a valid audio file
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is a valid audio file, False otherwise
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if at least one stream is of type 'audio'
        return 'audio' in result.stdout
    except Exception as e:
        logger.error(f"Error validating audio file: {str(e)}")
        return False 