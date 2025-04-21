𝗟𝗟𝗮𝗩𝗔 𝗩𝗶𝘀𝘂𝗮𝗹 𝗖𝗼𝗻𝘃𝗲𝗿𝘀𝗮𝘁𝗶𝗼𝗻 𝗔𝗜\
\
This repository contains two implementations of multimodal conversational AI systems capable of understanding and discussing images. Both implementations allow you to have interactive conversations about any image, but they differ in their approach and capabilities.\
\
📋 Table of Contents

* Overview
* Repository Structure
* Requirements
* Installation
* Running the Pre-trained LLaVA Model
* Running the Custom Implementation
* Training Your Own Model
* Troubleshooting

🔍 Overview
Two Implementations:

1. Pre-trained LLaVA (LLAVA_Pretrained.py):

* Uses the official LLaVA 1.5 7B model for inference
* Simple to use with no training required
* Provides state-of-the-art visual understanding capabilities

2. Custom Implementation (LLAVA_Developed.py):

* Custom architecture combining CLIP (for vision) and TinyLlama (for language)
* Trainable on your own data
* Memory-efficient with parameter-efficient fine-tuning (LoRA)
* Includes complete training pipeline and inference code



📁 Repository Structure

* LLAVA_Pretrained.py                    # Pre-trained LLaVA inference code
* LLAVA_Developed.py                     # Custom implementation with training pipeline
* requirements.py                        # Script to install all dependencies
* images100/                             # Folder containing training images
* instructions100.json                   # Training data with conversations about images
* README.md                              # This documentation file

𝐍𝐎𝐓𝐄 : Download the images100 (https://shorturl.at/uTI9K) and instructions100.json (https://shorturl.at/VTAZ7) before you proceed.

💻 Requirements

* Python 3.8 or higher
* CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended for training)
* Approximately 10GB disk space for models and data

🔧 Installation

1. Clone the repository:
```c++ 
git clone https://github.com/BoKuWaBaKa/Design-Lab.git
cd Design-Lab
```

2. Install dependencies:
Run the provided installation script:
```python
python requirements.py
```

3. Check for other requirements
* Check Python version compatibility
* Install all required packages (PyTorch, Transformers, PEFT, etc.)
* Configure CUDA support if available
* Create necessary folders for the project

Note: The installation might take a few minutes as it downloads several large packages.


🚀 𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝘁𝗵𝗲 𝗣𝗿𝗲-𝘁𝗿𝗮𝗶𝗻𝗲𝗱 𝗟𝗟𝗮𝗩𝗔 𝗠𝗼𝗱𝗲𝗹\
\
The pre-trained model provides ready-to-use visual conversation capabilities without any training.

1. Run the inference script:
```python
python LLAVA_Pretrained.py
```

2. Provide an image path when prompted:

* Can be a local file path (e.g., ./images100/image1.jpg)
* Can be a URL (e.g., https://example.com/image.jpg)


3. Chat with the model by typing your questions.

4. Type "exit" to end the conversation



Example Usage:
```python
$ python LLAVA_Pretrained.py
Loading model: llava-hf/llava-1.5-7b-hf
Enter path to image: ./images100/cat.jpg
Starting chat with image: ./images100/cat.jpg
Type your questions or 'exit' to quit.
User: What can you see in this image?
Assistant: In this image, I can see a cat sitting on what appears to be a windowsill or ledge. The cat has a tabby coat pattern with orange/ginger and white coloring. It's looking directly at the camera with its characteristic cat eyes. The background seems to be indoors, possibly near a window with some light coming in.
Generation took 4.23 seconds
User: exit
```

🛠️ 𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝘁𝗵𝗲 𝗖𝘂𝘀𝘁𝗼𝗺 𝗜𝗺𝗽𝗹𝗲𝗺𝗲𝗻𝘁𝗮𝘁𝗶𝗼𝗻\
\
The custom implementation can be used in two modes: training and inference.

1. Run the script:
```python
python LLAVA_Developed.py
```

2. Select mode when prompted with Enter mode (train/demo):

* Type "train" to train the model on the provided dataset
* Type "demo" to run inference with a trained checkpoint


3. For training mode:

* create a directory named "checkpoints"
* The model will start training on the provided image dataset and instruction data
* Checkpoints will be saved in the "checkpoints" directory
* Training progress will be displayed


4. For demo mode:

* Enter the path to a saved checkpoint when prompted
* Enter the path to an image for the conversation
* Chat with the model by typing your questions
* Type "exit" to end the conversation
* Type [start a new conversation] to reset the conversation history

Example Usage:
```python
$ python LLAVA_Developed.py
Using device: cuda
Enter mode (train/demo): demo
Enter path to model checkpoint: ./checkpoints/multimodal_model_checkpoint_epoch_2.pt
Enter path to image for demo: ./images100/dog.jpg
Chatbot demo with image: ./images100/dog.jpg
Type your message or 'exit' to quit.
User: Describe what you see in this image.
Generating response...
Assistant: I can see a brown dog lying on what appears to be a wooden floor or deck. The dog looks relaxed and comfortable.
Generation took 2.18 seconds
User: exit
```

🏋️ 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗬𝗼𝘂𝗿 𝗢𝘄𝗻 𝗠𝗼𝗱𝗲𝗹\
\
To train the custom model on your own data:

1. Prepare your dataset:

* Place your images in the images100/ directory
* Create a JSON file similar to instructions100.json with conversations about your images
* See Dataset Information for the format


2. Run the training:
```python
python LLAVA_Developed.py
```

When prompted, enter "train".

3. Monitor training:

* Loss values will be displayed during training
* Checkpoints will be saved after each epoch in the checkpoints/ directory
* You can stop training at any time with Ctrl+C


4. Customize training parameters (optional):

Open LLAVA_Developed.py and modify these parameters in the main() function:

* batch_size: Number of samples per batch (reduce if running out of memory)
* num_epochs: Total number of training epochs
* learning_rate: Learning rate for the optimizer


 🔍 𝗧𝗿𝗼𝘂𝗯𝗹𝗲𝘀𝗵𝗼𝗼𝘁𝗶𝗻𝗴 \
 \
𝗖𝗼𝗺𝗺𝗼𝗻 𝗜𝘀𝘀𝘂𝗲𝘀 𝗮𝗻𝗱 𝗦𝗼𝗹𝘂𝘁𝗶𝗼𝗻𝘀:

1. CUDA out of memory error:

* Reduce batch size in LLAVA_Developed.py
* Close other applications using GPU memory
* For the pre-trained model, try enabling 8-bit quantization by changing load_in_4bit=True to load_in_8bit=True


2. Model downloads fail:

* Check your internet connection
* Try running the script again (downloads will resume)
* If persistent, manually download the models from Hugging Face


3. Missing dependencies:

* Run python requirements.py again
* Check if any specific package shows installation errors
* Install missing packages manually with pip install [package-name]


4. Slow inference on CPU:

* This is expected - these models are designed for GPU acceleration
* Reduce max_new_tokens parameter for faster (but shorter) responses
* Consider using a cloud GPU service if local GPU is unavailable


5. Image loading errors:

* Ensure the image path is correct
* Verify the image format is supported (jpg, png, etc.)
* For URLs, ensure they are accessible and point directly to an image file
