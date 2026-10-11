# SeeForMe: AI-Powered Visual Assistant for Accessibility

SeeForMe is an AI-powered visual assistant prototype designed to help people with visual impairments understand images through natural-language descriptions, text extraction, and audio output.

## Features

- **Image Description:** Generates a natural-language description of an uploaded image using BLIP.
- **Text Recognition:** Extracts visible text from images using EasyOCR.
- **Text-to-Speech:** Converts the image description and detected text into spoken audio using gTTS.
- **Visual Question Answering:** Allows users to ask questions about an uploaded image.

## Technology Stack

- Python
- Hugging Face Transformers
- BLIP image captioning and visual question answering
- EasyOCR
- Gradio
- gTTS
- PyTorch

## How It Works

1. The user uploads an image.
2. The image captioning model generates a scene description.
3. EasyOCR extracts readable text from the image.
4. Text-to-speech generates audio output.
5. The visual question-answering model answers questions about the uploaded image.

## Running the Project

Install the required dependencies:

```bash
pip install gradio transformers easyocr gTTS accelerate torch pillow
```

Run the application:

```bash
python seefor_me.py
```

Alternatively, open the included Google Colab notebook and run its cells.

## Limitations

SeeForMe is a prototype. AI-generated descriptions and answers may be inaccurate, particularly when identifying specific objects or similar-looking species. Users should not rely on the system for safety-critical decisions.

## Intended Impact

SeeForMe explores how computer vision, OCR, and speech technologies can make visual information more accessible to people with visual impairments.
