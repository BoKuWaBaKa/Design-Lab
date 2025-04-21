import os
import json
import random
import gc
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    OneCycleLR, 
    ReduceLROnPlateau,
    StepLR
)
from transformers import (
    CLIPModel, 
    CLIPProcessor, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, 
    get_peft_model, 
    PeftModel
)
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
gc.collect()

STOP_TOKEN = "###"      # Special token used to mark the end of segments
IMAGE_PLACEHOLDER = "<image>"  # Placeholder text to mark image location
SYSTEM_MESSAGE = "System: You are a helpful assistant."  # Following Vicuna-v0

##############################################################################
# 1. Vision Encoder using CLIP ViT-B/32 (smaller than ViT-L/14)
##############################################################################

class CLIPVisionEncoder(nn.Module):
    def __init__(self, proj_dim, fine_tune=False):
        """
        Loads a smaller pre-trained CLIP model (ViT-B/32).
        
        Args:
            proj_dim (int): Dimension of the projected embedding
            fine_tune (bool): If False, the CLIP parameters are frozen
        """
        super().__init__()
        # Use smaller CLIP model (ViT-B/32 instead of ViT-L/14)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if not fine_tune:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        hidden_size = self.clip_model.config.vision_config.hidden_size  # 768 for ViT-B/32
        self.projection = nn.Linear(hidden_size, proj_dim)
        
    def forward(self, images):
        # Process images through CLIP
        with torch.no_grad():
            outputs = self.clip_model.vision_model(pixel_values=images)
        
        # Extract visual features (use pooled output for smaller memory footprint)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Project to match LLM dimensions
        # Reshape to represent a single token per image for simplicity
        projected_tokens = self.projection(pooled_output).unsqueeze(1)  # (batch_size, 1, proj_dim)
        
        return projected_tokens

##############################################################################
# 2. Language Model (TinyLlama - much smaller than Vicuna)
##############################################################################

class LightweightLLM(nn.Module):
    def __init__(self, model_name, tokenizer_name=None, visual_embed_dim=768):
        """
        Loads a lightweight model with quantization.
        
        Args:
            model_name (str): Identifier/path for the model
            tokenizer_name (str): Identifier/path for the tokenizer (if None, uses model_name)
            visual_embed_dim (int): Dimension of visual features
        """
        super().__init__()
        
        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load smaller language model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure we have padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add LoRA adapters for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for kbit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Determine embedding dimension from model config
        text_embed_dim = self.model.config.hidden_size
        
        # Create projection for visual embeddings
        self.visual_proj = nn.Linear(visual_embed_dim, text_embed_dim)

    def forward(self, text_input, additional_embeddings=None):
        if additional_embeddings is not None:
            # Project visual embeddings to match text embedding dimensions
            projected_visual = self.visual_proj(additional_embeddings)
            
            # Get text embeddings
            with torch.no_grad():
                text_embeds = self.model.get_input_embeddings()(text_input)
            
            # Concatenate along sequence dimension
            inputs_embeds = torch.cat([projected_visual, text_embeds], dim=1)
            
            # Create attention mask (1s for all tokens)
            batch_size, visual_seq_len = projected_visual.size()[:2]
            text_seq_len = text_input.size(1)
            attention_mask = torch.ones(batch_size, visual_seq_len + text_seq_len, device=text_input.device)
            
            # Forward pass through the model
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return outputs.logits[:, visual_seq_len:, :]  # Return only text token logits
        else:
            outputs = self.model(input_ids=text_input)
            return outputs.logits

##############################################################################
# 3. Multimodal Model: Fusing Visual Tokens with LLM
##############################################################################

class MultimodalModel(nn.Module):
    def __init__(self, vision_encoder, llm):
        """
        Combines the visual encoder and language model.
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        
    def forward(self, images, text_input, additional_embeddings=None):
        # Obtain visual tokens from images
        if images is not None:
            visual_tokens = self.vision_encoder(images)
            # Pass text and visual tokens to the language model
            output = self.llm(text_input, additional_embeddings=visual_tokens)
        else:
            # Text-only forward pass
            output = self.llm(text_input)
        return output

##############################################################################
# 4. Dataset with Unified Multi-Turn Conversation Formatting
##############################################################################

class MultimodalInstructionDataset(Dataset):
    def __init__(self, json_path, image_dir, tokenizer, max_text_length=512, image_transform=None, limit_images=100):
        """
        Dataset for multimodal instructions with images.
        
        Args:
            json_path (str): Path to the JSON file with samples
            image_dir (str): Directory where image files are stored
            tokenizer: HuggingFace tokenizer
            max_text_length (int): Maximum text length
            image_transform: torchvision transforms to apply on images
            limit_images (int): Maximum number of images to use
        """
        # Load and filter data
        with open(json_path, "r") as f:
            data = json.load(f)
            
        filtered_data = []
        for record in data:
            image_path = os.path.join(image_dir, record["image"])
            if os.path.exists(image_path):
                filtered_data.append(record)
                if len(filtered_data) >= limit_images:
                    break
            else:
                print(f"Skipping missing image: {image_path}")
                
        self.data = filtered_data
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Default transform for CLIP compatibility
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.stop_token = STOP_TOKEN
        self.system_message = SYSTEM_MESSAGE
        self.image_placeholder = IMAGE_PLACEHOLDER

    def process_conversations(self, conversation):
        """
        Builds the unified sequence as a list of (text, segment_type) tuples.
        """
        segments = []
        # Add the system message segment
        segments.append((self.system_message + " " + self.stop_token, "system"))
        
        turns = []
        current_turn = []
        # Group conversation messages into turns [question, answer]
        for msg in conversation:
            if msg["from"].lower() == "human":
                current_turn = [msg["value"]]
            elif msg["from"].lower() == "gpt" and current_turn:
                current_turn.append(msg["value"])
                turns.append(current_turn)
                current_turn = []
                
        # Build segments for each turn
        for i, turn in enumerate(turns):
            question, answer = turn
            if i == 0:
                # For t=1: randomly choose the order of question and image
                if random.random() < 0.5:
                    instruct_text = question + " " + self.image_placeholder
                else:
                    instruct_text = self.image_placeholder + " " + question
            else:
                instruct_text = question
                
            segments.append(("Human: " + instruct_text + " " + self.stop_token, "human"))
            segments.append(("Assistant: " + answer + " " + self.stop_token, "assistant"))
            
        return segments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image_path = os.path.join(self.image_dir, record["image"])
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Provide a blank image as fallback
            image = torch.zeros(3, 224, 224)
            
        # Build the unified conversation segments
        segments = self.process_conversations(record["conversations"])
        
        # Tokenize segments individually and build a unified sequence
        input_ids = []
        labels = []
        
        for seg_text, seg_type in segments:
            tokens = self.tokenizer(seg_text, add_special_tokens=False)["input_ids"]
            input_ids.extend(tokens)
            # Only compute loss on assistant segments
            if seg_type == "assistant":
                labels.extend(tokens)
            else:
                labels.extend([-100] * len(tokens))
                
        # Truncate the sequence to max_text_length
        if len(input_ids) > self.max_text_length:
            input_ids = input_ids[:self.max_text_length]
            labels = labels[:self.max_text_length]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image": image
        }

##############################################################################
# 5. Collate Function for Padding
##############################################################################

def collate_fn(batch, tokenizer):
    """
    Custom collate function to handle padding properly.
    
    Args:
        batch: List of samples from the dataset
        tokenizer: The tokenizer to use for padding
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    images = [item["image"] for item in batch]
    
    # Use the tokenizer's pad function for input_ids
    batch_encoding = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt", padding="longest")
    input_ids_padded = batch_encoding["input_ids"]
    attention_mask = batch_encoding["attention_mask"]
    
    # Manually pad labels (using -100 for padding tokens)
    max_len = max(len(l) for l in labels)
    labels_padded = []
    for l in labels:
        padded = torch.cat([l, torch.tensor([-100] * (max_len - len(l)), dtype=torch.long)])
        labels_padded.append(padded)
    labels_padded = torch.stack(labels_padded)
    
    # Stack images
    images_tensor = torch.stack(images)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "images": images_tensor
    }

##############################################################################
# 6. Training Code
##############################################################################

def train_model(model, train_loader, tokenizer, num_epochs=2, learning_rate=5e-5, device="cuda"):
    """
    Train the multimodal model with gradient accumulation and mixed precision.
    
    Args:
        model: The multimodal model to train
        train_loader: DataLoader for training data
        tokenizer: Tokenizer for the language model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        device: Device to train on
    """
    # Set model to training mode
    model.train()
    
    # Configure optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Use gradient accumulation for effective larger batch size
    accumulation_steps = 4
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        # Training loop
        for i, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images=images, text_input=input_ids)
                loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights after accumulation steps
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Log progress
                total_loss += loss.item() * accumulation_steps
                batch_count += 1
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}")
                
                # Clear cache to prevent OOM
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_path = f"checkpoints/multimodal_model_checkpoint_epoch_{epoch+1}.pt"
        save_model(model, save_path)
 

##############################################################################
# 7. Model Saving/Loading Functions
##############################################################################

def save_model(model, path):
    """
    Save the model in an efficient format.
    
    Args:
        model: The model to save
        path: Path to save the model to
    """
    # Save only the state_dict of trainable components
    model_data = {
        "vision_encoder_projection": model.vision_encoder.projection.state_dict(),
        "llm_visual_proj": model.llm.visual_proj.state_dict(),
        "llm_lora_weights": model.llm.model.state_dict()
    }
    torch.save(model_data, path)
    print(f"Model saved to {path}")

def load_model(path, vision_model_name="openai/clip-vit-base-patch32", 
               llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda"):
    """
    Load the model from a saved checkpoint.
    
    Args:
        path: Path to the saved model
        vision_model_name: Name of the vision model to use
        llm_model_name: Name of the language model to use
        device: Device to load the model on
        
    Returns:
        The loaded model
    """
    # Create a new model with the base architecture
    vision_encoder = CLIPVisionEncoder(proj_dim=768, fine_tune=False)
    llm = LightweightLLM(model_name=llm_model_name)
    model = MultimodalModel(vision_encoder, llm)
    
    # Load state dict for each component
    checkpoint = torch.load(path, map_location=device)
    model.vision_encoder.projection.load_state_dict(checkpoint["vision_encoder_projection"])
    model.llm.visual_proj.load_state_dict(checkpoint["llm_visual_proj"])
    
    # Load LoRA weights
    model.llm.model.load_state_dict(checkpoint["llm_lora_weights"], strict=False)
    
    model.to(device)
    model.eval()
    return model

##############################################################################
# 8. Generation Functions
##############################################################################

def generate_response(model, image_tensor, prompt, tokenizer, device, max_new_tokens=100, temperature=0.7):
    """
    Generate a response using the model.
    
    Args:
        model: The multimodal model
        image_tensor: The image tensor
        prompt: The text prompt
        tokenizer: The tokenizer
        device: The device to use
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        
    Returns:
        The generated response
    """
    model.eval()
    
    # Tokenize the prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    
    # Process the image with vision encoder
    with torch.no_grad():
        visual_tokens = model.vision_encoder(image_tensor)
        
        # Project visual tokens
        projected_visual = model.llm.visual_proj(visual_tokens)
        
        # Get text embeddings
        text_embeds = model.llm.model.get_input_embeddings()(input_ids)
        
        # Concatenate embeddings
        inputs_embeds = torch.cat([projected_visual, text_embeds], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(
            inputs_embeds.size(0),
            inputs_embeds.size(1),
            device=device
        )
        
        # Generate text
        output_ids = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Get only the newly generated tokens
        new_tokens = output_ids[0, input_ids.size(1):]
        
        # Decode the output
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Stop at STOP_TOKEN if present
        if STOP_TOKEN in response:
            response = response.split(STOP_TOKEN)[0]
            
    return response

##############################################################################
# 9. Demo Functions
##############################################################################

def chat_demo(model, tokenizer, device, image_path):
    """
    Interactive chatbot demo with an image.
    """
    # Load and preprocess the image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    
    # Start conversation with system message
    conversation_history = f"{SYSTEM_MESSAGE} {STOP_TOKEN}\n"
    
    print(f"Chatbot demo with image: {image_path}")
    print("Type your message or 'exit' to quit.")
    
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
            
        if user_input.strip().lower() == "[start a new conversation]":
            conversation_history = f"{SYSTEM_MESSAGE} {STOP_TOKEN}\n"
            print("Conversation history cleared.")
            continue
            
        # Add image placeholder to first message
        if "Human:" not in conversation_history:
            user_input = f"{IMAGE_PLACEHOLDER} {user_input}"
            
        # Add user message to history
        conversation_history += f"Human: {user_input} {STOP_TOKEN}\n"
        
        # Prepare prompt for generation
        prompt = conversation_history + "Assistant: "
        
        # Generate response
        print("Generating response...")
        start_time = time.time()
        response = generate_response(model, image_tensor, prompt, tokenizer, device)
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Generation took {end_time - start_time:.2f} seconds")
        
        # Add response to history
        conversation_history += f"Assistant: {response} {STOP_TOKEN}\n"

##############################################################################
# 10. Main Function
##############################################################################

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Model parameters
    vision_model_name = "openai/clip-vit-base-patch32"  # Smaller CLIP model
    llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small LLM (1.1B)
    
    # Paths
    json_path = "instructions100.json"
    image_dir = "images100"
    
    # Training parameters
    batch_size = 32  # Small batch size to save memory
    num_epochs = 10000
    learning_rate = 5e-5
    
    # Choose mode: "train" or "demo"
    mode = input("Enter mode (train/demo): ").strip().lower()
    
    if mode == "train":
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Initialize model components
        vision_encoder = CLIPVisionEncoder(proj_dim=768, fine_tune=False)
        llm = LightweightLLM(model_name=llm_model_name)
        model = MultimodalModel(vision_encoder, llm)
        model.to(device)
        
        # Create dataset and dataloader
        dataset = MultimodalInstructionDataset(
            json_path=json_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            max_text_length=512,  # Reduced from 1024
            limit_images=100
        )
        
        # Create custom collate function with tokenizer
        my_collate_fn = lambda batch: collate_fn(batch, tokenizer)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=my_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        # Train the model
        train_model(
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )
        
    elif mode == "demo":
        # Load model checkpoint
        checkpoint_path = input("Enter path to model checkpoint: ")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = load_model(
            path=checkpoint_path,
            vision_model_name=vision_model_name,
            llm_model_name=llm_model_name,
            device=device
        )
        
        # Run chat demo
        image_path = input("Enter path to image for demo: ")
        chat_demo(model, tokenizer, device, image_path)
        
    else:
        print("Invalid mode. Please enter 'train' or 'demo'.")

if __name__ == "__main__":
    main()

