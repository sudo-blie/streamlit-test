import base64
import io
import json
import logging
import os
import re
from pathlib import Path

from ollama import Client
from PIL import Image
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pydantic import BaseModel

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# Configure logging with colored output and file/line information
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration for Ollama API
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:7b")

# =============================================================================
# Pydantic Model for Structured Output
# =============================================================================
class Label(BaseModel):
    """Model for product label OCR results.
    
    This Pydantic model defines the expected structure of the OCR output.
    It matches the structure used in llama3.2-vision implementation.
    Each field is required and will be validated during schema validation.
    
    Fields:
        Name: Product name or identifier
        Model: Model number or variant
        Buy_Date: Purchase or manufacture date
        Serial_Number: Unique serial number
    """
    Name: str
    Model: str
    Buy_Date: str
    Serial_Number: str

# =============================================================================
# Image Processing Functions
# =============================================================================
def validate_image_path(image_path: str) -> None:
    """Validate if the image path exists and is a valid image file.
    
    Args:
        image_path: Path to the image file to validate
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is not supported
    """
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
    """Convert an image file to base64 string with size-aware resizing.
    
    This function:
    1. Loads the image and gets original dimensions
    2. Resizes to max 1024px on longest side
    3. Compresses with adjustable JPEG quality
    4. Ensures output is under max_size_mb
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum allowed size in MB (default: 10MB)
    
    Returns:
        str: Base64 encoded image data
    """
    try:
        with load_image(image_path) as image:
            original_width, original_height = image.size
            original_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Original image size: {original_width}x{original_height}, {original_size:.2f}MB")

            # Scale image to 1024px on longest side
            long_side = max(original_width, original_height)
            scale_factor = 1024 / long_side
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try different JPEG quality levels until size is under limit
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

# =============================================================================
# Main OCR Function with Structured Output
# =============================================================================
def perform_structured_ocr(image_path: str, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> Label:
    """Process an image with OCR using Gemma model via Ollama with structured output validation.
    
    Key steps:
    1. Convert image to base64
    2. Set up Ollama chat with schema enforcement
    3. Process image with specific prompt for structured extraction
    4. Validate response against Label schema
    
    The function uses Pydantic's model_json_schema() to enforce the output structure,
    similar to how llama3.2-vision handles structured output. This ensures the
    response matches the Label class fields exactly.
    
    Args:
        image_path: Path to image file
        host: Ollama API host
        model: Model name to use
    
    Returns:
        Label: Validated OCR results matching Label schema
    
    Raises:
        Various exceptions for file, API, and validation errors
    """
    try:
        # Convert image to base64
        base64_image = image_to_base64(image_path)
        
        # Initialize Ollama client
        client = Client(host=host)

        # Process with Gemma using schema enforcement
        response = client.chat(
            model=model,
            options={
                # Match successful llama3.2-vision settings
                "temperature": 0.3,  # Lower temp for more consistent output
                "top_p": 0.8,       # Nucleus sampling parameter
                "top_k": 70,        # Top-k sampling parameter
            },
            # Key difference: Provide schema to guide model output
            format=Label.model_json_schema(),
            messages=[
                {
                    "role": "system",
                    "content": """Extract all text from this image in English, **strictly preserving** the structure.
                    - **Do not summarize, add, or modify any text.**
                    - Maintain hierarchical sections and subsections as they appear.
                    - Use keys that reflect the document's actual structure.
                    - Include all text, even if fragmented, blurry, or unclear."""
                },
                {
                    "role": "user",
                    "content": "Extract and structure all text from this image according to the schema.",
                    "images": [base64_image],
                },
            ],
            stream=False,
        )
        
        # Process response and validate against schema
        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"].strip()
            try:
                # Direct validation against Label schema
                # This will raise ValidationError if response doesn't match
                return Label.model_validate_json(content)
            except Exception as e:
                logger.error(f"{Fore.RED}Schema validation failed: {str(e)}{Style.RESET_ALL}")
                raise
        else:
            raise ValueError("Invalid response format from Ollama API")
            
    except Exception as e:
        logger.error(f"{Fore.RED}Error in Gemma OCR process: {str(e)}{Style.RESET_ALL}")
        raise

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*50}")
    print("Gemma OCR Demo")
    print(f"{'='*50}{Style.RESET_ALL}\n")

    # Process a sample image
    image_path = "./pics/razer_serial.png"

    try:
        # Extract text with structured output validation
        result = perform_structured_ocr(image_path)
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")


# SAMPLE OUTPUT:
# =============================================================================
# ==================================================
# Gemma OCR Demo
# ==================================================

# 2025-03-18 15:30:17,896 - INFO - [gemma_ocr.py:98] - Original image size: 3024x4032, 11.25MB
# 2025-03-18 15:30:18,213 - INFO - [gemma_ocr.py:116] - Final image size: 768x1024, 0.12MB (quality=95)
# 2025-03-18 15:31:29,508 - INFO - [_client.py:1025] - HTTP Request: POST https://it-test-ollama.nexon.net/api/chat "HTTP/1.1 200 OK"
# {
#   "Name": "RAZ-145",
#   "Model": "Racerblade 15 3070",
#   "Buy_Date": "2021-04-21",
#   "Serial_Number": "BY2116M19709883"
# }