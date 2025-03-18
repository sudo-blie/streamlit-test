import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import ollama
from PIL import Image
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:7b")

def validate_image_path(image_path: str) -> None:
    """Validate if the image path exists and is a valid image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    if not Path(image_path).suffix.lower() in valid_extensions:
        raise ValueError(f"Invalid image format. Supported formats: {valid_extensions}")

def load_image(image_path: str) -> Image.Image:
    """Load an image file using PIL."""
    validate_image_path(image_path)
    return Image.open(image_path)

def image_to_base64(image_path: str, max_size_mb: float = 10.0) -> str:
    """Convert an image file to base64 string with size-aware resizing."""
    try:
        with load_image(image_path) as image:
            original_width, original_height = image.size
            original_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Original image size: {original_width}x{original_height}, {original_size:.2f}MB")

            # Start with standard resize
            long_side = max(original_width, original_height)
            scale_factor = 1024 / long_side
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try different quality levels if needed
            quality = 95
            while True:
                buffered = io.BytesIO()
                resized_image.save(buffered, format="JPEG", quality=quality, optimize=True)
                size_mb = len(buffered.getvalue()) / (1024 * 1024)
                
                if size_mb <= max_size_mb or quality <= 30:
                    logger.info(f"Final image size: {new_width}x{new_height}, {size_mb:.2f}MB (quality={quality})")
                    return base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Reduce quality and try again
                quality -= 10
    except Exception as e:
        logger.error(f"{Fore.RED}Failed to process image {image_path}: {str(e)}{Style.RESET_ALL}")
        raise

def perform_ocr(image_path: str, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> dict:
    """Process an image with OCR using Gemma model via Ollama.
    
    Args:
        image_path: Path to the image file
        host: Ollama API host (default: from environment or localhost)
        model: Model name to use (default: from environment or gemma:7b)
    
    Returns:
        dict: Extracted text as JSON object with key-value pairs
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is invalid or API response is invalid
        Exception: For other processing errors
    """
    logger.info(f"{Fore.GREEN}Processing image with Ollama Gemma: {image_path}{Style.RESET_ALL}")
    
    try:
        # Convert image to base64
        base64_image = image_to_base64(image_path)
        
        # Initialize Ollama client
        client = ollama.Client(host=host)
        
        # Process with Gemma
        response = client.chat(
            model=model,
            options={
                "temperature": 0.1,  # Lower temperature for more deterministic responses
                "top_p": 0.9,
                "top_k": 40,
            },
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant that recognizes text in images. Images will contain labels, tables, or documents.
                    Format all responses as valid JSON objects with key-value pairs.
                    Rules:
                    1. Response must start with an opening curly brace {
                    2. Each key-value pair must be properly quoted
                    3. Response must end with a closing curly brace }
                    4. No additional text or explanation should be included
                    5. Extract all visible text from the image""",
                },
                {
                    "role": "user",
                    "content": """
                        Extract all text from this image and format it as JSON with descriptive keys.
                        {
                           "key": "value",
                           "key": "value",
                           "key": "value",
                           ...
                        }
                        """,
                    "images": [base64_image],
                },
            ],
            stream=False,
        )

        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"].strip()
            logger.info(f"Raw response content:\n{content}")
            # First try to extract JSON using regex
            json_pattern = r"\{.*\}"
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(0)
                logger.info(f"Extracted JSON string:\n{json_str}")
                try:
                    # Validate JSON by parsing it
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # If JSON is invalid, clean up common issues
                    # Remove trailing commas
                    cleaned = re.sub(r',\s*}', '}', json_str)
                    # Fix missing commas
                    cleaned = re.sub(r'(?<![\{\s,])\s*"', ' "', cleaned)
                    # Remove extra spaces in keys and values
                    cleaned = re.sub(r'"\s*([^"]+)\s*":', '"\1":', cleaned)
                    cleaned = re.sub(r':\s*"([^"]+)\s*"', ':"\1"', cleaned)
                    # Handle standalone text at the end
                    cleaned = re.sub(r'\s*"([^"]+)\."\s*}', '', cleaned)
                    cleaned = re.sub(r'\s*([^"\s]+\.)\s*}', '', cleaned)
                    cleaned = cleaned.rstrip() + '}'
                    
                    # Handle duplicate keys by parsing and rebuilding
                    try:
                        data = json.loads(cleaned)
                        # Convert to dict to automatically handle duplicates
                        # (later values override earlier ones)
                        result = dict(data)
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse cleaned JSON: {e}")
                        raise
            else:
                raise ValueError("No JSON object found in response")
        raise ValueError("Invalid response format from Ollama API")
    except Exception as e:
        logger.error(f"{Fore.RED}Error in Ollama Gemma OCR process: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    # Example usage
    image_path = "./pics/razerbox.png"
    
    try:
        result = perform_ocr(image_path)
        print(f"\n{Fore.GREEN}Extracted Text:{Style.RESET_ALL}")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")