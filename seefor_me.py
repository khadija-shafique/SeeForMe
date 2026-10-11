
import os
import tempfile
import torch
import gradio as gr
import easyocr

from PIL import Image
from gtts import gTTS
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("SeeForMe device:", device)

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
caption_model.eval()

vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)
vqa_model.eval()

ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def analyze_image(image):
    if image is None:
        return "Please upload an image.", "", None

    image = image.convert("RGB")
    inputs = caption_processor(image, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = caption_model.generate(**inputs, max_new_tokens=60)

    caption = caption_processor.decode(
        output[0], skip_special_tokens=True
    ).strip()

    import numpy as np
    detected = ocr_reader.readtext(np.array(image))
    extracted_text = [
        item[1] for item in detected if item[2] > 0.25
    ]
    ocr_text = (
        "\n".join(extracted_text)
        if extracted_text else "No readable text detected."
    )

    speech_text = (
        "See For Me image description. " + caption +
        ". Text detected in the image: " +
        (", ".join(extracted_text)
         if extracted_text else "No readable text detected.")
    )

    audio_path = None
    try:
        audio_path = os.path.join(
            tempfile.gettempdir(), "seefor_me_audio.mp3"
        )
        gTTS(text=speech_text, lang="en").save(audio_path)
    except Exception as e:
        print("Speech generation unavailable:", e)

    return caption, ocr_text, audio_path

def ask_about_image(image, question):
    if image is None:
        return "Please upload an image first."
    if not question or not question.strip():
        return "Please enter a question."

    image = image.convert("RGB")
    inputs = vqa_processor(
        image, question.strip(), return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output = vqa_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            num_beams=1
        )

    answer = vqa_processor.decode(
        output[0], skip_special_tokens=True
    ).strip()

    return answer or "I couldn't determine an answer reliably."

with gr.Blocks(title="SeeForMe | Visual Assistant") as demo:
    gr.Markdown("""
    # SeeForMe
    ### AI Visual Assistant for Accessibility

    Upload an image to receive a scene description, extract visible text,
    listen to the results, or ask a question about the image.

    *This prototype may make mistakes. Do not rely on it for safety-critical decisions.*
    """)

    image_input = gr.Image(type="pil", label="Upload an image")
    analyze_button = gr.Button("Analyze Image", variant="primary")
    description_output = gr.Textbox(label="Scene Description", lines=3)
    text_output = gr.Textbox(label="Text Detected (OCR)", lines=4)
    audio_output = gr.Audio(label="Listen to Description", type="filepath")

    analyze_button.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[description_output, text_output, audio_output]
    )

    gr.Markdown("### Ask a Question About Your Image")
    question_input = gr.Textbox(
        label="Your question",
        placeholder="What does the sign say? What color is the bag?"
    )
    ask_button = gr.Button("Ask SeeForMe")
    answer_output = gr.Textbox(label="Answer")

    ask_button.click(
        fn=ask_about_image,
        inputs=[image_input, question_input],
        outputs=answer_output
    )

demo.launch(share=True, debug=False)
