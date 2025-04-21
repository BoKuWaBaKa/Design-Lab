#Giving accurate results

import os
import json
import torch
from PIL import Image
import requests
from io import BytesIO
import time
from transformers import (
    LlavaProcessor, 
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)

# Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Smaller model version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################
# 1. Load Pre-trained Model
##############################################################################

def load_model(model_id=MODEL_ID, load_in_8bit=False, load_in_4bit=True):
    """
    Load the pre-trained LLaVA model using specific model class.
    """
    print(f"Loading model: {model_id}")
    
    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load model with the specific LLaVA class
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    processor = LlavaProcessor.from_pretrained(model_id)
    
    return model, processor

##############################################################################
# 2. Improved Inference Function
##############################################################################

def generate_response(model, processor, image_path, prompt, max_new_tokens=256):
    """
    Generate a response from the model with proper image token handling.
    """
    try:
        # Load image
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Format prompt correctly for LLaVA
        # This is the key fix - using the proper conversation format
        formatted_prompt = f"<image>\n{prompt}"
        
        # Process the inputs
        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move inputs to the correct device
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the output
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract response - remove the original prompt
        # LLaVA formatting means the response comes after the formatted prompt
        response_text = generated_text.replace(formatted_prompt, "").strip()
        
        return response_text
    
    except Exception as e:
        print(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

##############################################################################
# 3. Simple Chat Demo
##############################################################################

def chat_demo(model, processor, image_path):
    """
    Interactive chatbot demo with an image.
    """
    print(f"Starting chat with image: {image_path}")
    print("Type your questions or 'exit' to quit.")
    
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        
        start_time = time.time()
        response = generate_response(model, processor, image_path, user_input)
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Generation took {end_time - start_time:.2f} seconds")

##############################################################################
# 4. Main Function for Simplified Demo
##############################################################################

def main():
    # Load pre-trained model
    model, processor = load_model()
    
    # Get image path
    image_path = input("Enter path to image: ")
    
    # Run chat demo
    chat_demo(model, processor, image_path)

if __name__ == "__main__":
    main()