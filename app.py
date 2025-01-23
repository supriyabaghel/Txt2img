from flask import Flask, render_template, request, url_for
from deep_translator import GoogleTranslator
import requests
import os
from PIL import Image
from io import BytesIO
import logging
import time
import json
import sys

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Hugging Face API token
HF_API_TOKEN = "hf_token"
# Using a realistic model
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def enhance_prompt(prompt):
    """Add parameters to make the image more realistic"""
    enhancements = [
        "highly detailed",
        "photorealistic",
        "8k resolution",
        "professional photography",
        "natural lighting"
    ]
    return f"{prompt}, {', '.join(enhancements)}"

def generate_image(prompt):
    try:
        
        enhanced_prompt = enhance_prompt(prompt)
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "num_inference_steps": 50,  
                "guidance_scale": 8.5,      
                "width": 512,               
                "height": 512
            }
        }
        
        # Log the request details
        logger.info(f"Sending request to API with payload: {json.dumps(payload)}")
        
        # Call Hugging Face API with a timeout
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        # Log the response status and headers
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 503:
            logger.info("Model is loading, waiting and retrying...")
            time.sleep(20)  # Wait longer for the larger model
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"API request failed with status code: {response.status_code}")
            try:
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail)}")
            except:
                logger.error(f"Raw response content: {response.content}")
            return None
            
        # Try to process the image data
        try:
            logger.info("Attempting to process image data")
            image = Image.open(BytesIO(response.content))
            logger.info(f"Image processed successfully: {image.format}, {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to process image data: {str(e)}")
            logger.error(f"Response content type: {response.headers.get('content-type')}")
            logger.error(f"First 100 bytes of response: {response.content[:100]}")
            return None
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {str(e)}")
        return None

def get_translation(text, source_lang):
    try:
        logger.info(f"Starting translation from {source_lang} to English: {text}")
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        logger.info(f"Translation successful: {translated}")
        return translated
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        text = request.form['text']
        language = request.form.get('language', 'en')
        
        logger.info(f"Received generation request - Text: {text}, Language: {language}")
        
        
        translation = get_translation(text, language) if language != 'en' else text
        
        if not translation:
            logger.error("Translation failed")
            return render_template('index.html', error="Failed to translate text. Please try again.")
            
        logger.info(f"Using prompt: {translation}")
        
        generated_image = generate_image(translation)
        
        if not generated_image:
            logger.error("Image generation failed")
            return render_template('index.html', error="Failed to generate image. Please try again in a few seconds.")
            
        
        os.makedirs('static/images', exist_ok=True)
        
        
        timestamp = int(time.time())
        filename = f"{text[:10]}_{timestamp}.png"
        image_path = os.path.join('static', 'images', filename)
        
        try:
            generated_image.save(image_path)
            logger.info(f"Image saved successfully to: {image_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return render_template('index.html', error="Failed to save generated image. Please try again.")
        
        
        image_url = url_for('static', filename=f'images/{filename}')
        return render_template('index.html', image_path=image_url)
                
    except Exception as e:
        logger.error(f"Unexpected error in generate route: {str(e)}")
        return render_template('index.html', error=f"An unexpected error occurred. Please try again.")

if __name__ == '__main__':
    
    os.makedirs('static/images', exist_ok=True)

    app.run(debug=True, port=5000)
