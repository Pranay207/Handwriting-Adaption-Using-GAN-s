import gradio as gr
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# ----------------------
# Load TroCR model for handwritten OCR
# ----------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------
import gradio as gr
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# ----------------------
# Safe Default Image
# ----------------------
default_img_path = "default_sample.png"

if os.path.exists(default_img_path):
    try:
        default_img = Image.open(default_img_path).convert("RGB")
    except Exception:
        default_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
else:
    default_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
# Load TroCR model for handwritten OCR
# ----------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------


# ----------------------
# OCR function using TroCR
# ----------------------
def run_hf_ocr(image: Image.Image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# ----------------------
# Pipeline
# ----------------------
def process_pipeline(image: Image.Image):
    if image is None:
        return default_img, default_img, "Upload a prescription image to see results."
    
    # For now, GAN is skipped; just show original as normalized
    normalized_img = image

    # Run OCR
    ocr_text = run_hf_ocr(normalized_img)
    return image, normalized_img, ocr_text

# ----------------------
# Gradio Interface
# ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DoctorScribble2Text: Prescription OCR")
    gr.Markdown("Upload a prescription → OCR extracts text")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Prescription", value=default_img)
        run_btn = gr.Button("Process")

    with gr.Row():
        original = gr.Image(type="pil", label="Original")
        normalized = gr.Image(type="pil", label="Normalized")
        text_out = gr.Textbox(label="OCR Text")

    run_btn.click(process_pipeline, inputs=input_img, outputs=[original, normalized, text_out])

if __name__ == "__main__":
    demo.launch()


# ----------------------
# OCR function using TroCR
# ----------------------
def run_hf_ocr(image: Image.Image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# ----------------------
# Pipeline
# ----------------------
def process_pipeline(image: Image.Image):
    if image is None:
        return default_img, default_img, "Upload a prescription image to see results."
    
    # For now, GAN is skipped; just show original as normalized
    normalized_img = image

    # Run OCR
    ocr_text = run_hf_ocr(normalized_img)
    return image, normalized_img, ocr_text

# ----------------------
# Gradio Interface
# ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DoctorScribble2Text: Prescription OCR")
    gr.Markdown("Upload a prescription → OCR extracts text")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Prescription", value=default_img)
        run_btn = gr.Button("Process")

    with gr.Row():
        original = gr.Image(type="pil", label="Original")
        normalized = gr.Image(type="pil", label="Normalized")
        text_out = gr.Textbox(label="OCR Text")

    run_btn.click(process_pipeline, inputs=input_img, outputs=[original, normalized, text_out])

if __name__ == "__main__":
    demo.launch()
