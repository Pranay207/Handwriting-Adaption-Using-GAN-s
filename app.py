import os
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*_register_pytree_node.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*resume_download.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*early_stopping.*beam-based generation modes.*",
    category=UserWarning,
)

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


MODEL_CANDIDATES = [
    "microsoft/trocr-base-printed",
    "microsoft/trocr-base-handwritten",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None
trocr_model_name = None
easyocr_reader = None
ocr_backend = None
model_error = None
easyocr_error = None


def load_default_image() -> Image.Image:
    for path in ("default_sample.png", "sample.png"):
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    return create_demo_image()


def create_demo_image() -> Image.Image:
    image = Image.new("RGB", (900, 420), color=(252, 251, 247))
    draw = ImageDraw.Draw(image)
    lines = [
        "Prescription Sample",
        "Tab Paracetamol 500 mg",
        "1 tablet after food",
        "Twice daily for 3 days",
        "Drink plenty of water",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill=(35, 35, 35))
        y += 58
    return image


def load_trocr():
    global processor, model, trocr_model_name, ocr_backend, model_error

    errors = []
    for model_name in MODEL_CANDIDATES:
        try:
            processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=True)
            model.to(device)
            trocr_model_name = model_name
            ocr_backend = f"TrOCR ({model_name.split('-')[-1]}, cached, {device})"
            return
        except Exception as cached_error:
            errors.append(f"{model_name} cached load failed: {cached_error}")
            try:
                processor = TrOCRProcessor.from_pretrained(model_name)
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                model.to(device)
                trocr_model_name = model_name
                ocr_backend = f"TrOCR ({model_name.split('-')[-1]}, downloaded, {device})"
                return
            except Exception as download_error:
                errors.append(f"{model_name} download failed: {download_error}")

    processor = None
    model = None
    trocr_model_name = None
    model_error = " | ".join(errors)


def load_easyocr():
    global easyocr_reader, ocr_backend, easyocr_error

    if easyocr is None:
        return

    try:
        easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        ocr_backend = f"EasyOCR ({'cuda' if torch.cuda.is_available() else 'cpu'})"
    except Exception as error:
        easyocr_reader = None
        easyocr_error = str(error)


def detect_tesseract():
    global ocr_backend

    if pytesseract is None:
        return

    try:
        version = pytesseract.get_tesseract_version()
        ocr_backend = f"Tesseract OCR ({version})"
    except Exception:
        pass


def normalize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    grayscale = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(grayscale)
    boosted = ImageEnhance.Contrast(autocontrast).enhance(1.8)
    sharpened = boosted.filter(ImageFilter.SHARPEN)
    enlarged = sharpened.resize(
        (max(384, sharpened.width * 2), max(128, sharpened.height * 2)),
        Image.Resampling.LANCZOS,
    )
    return enlarged.convert("RGB")


def build_ocr_variants(image: Image.Image):
    base = image.convert("RGB")
    normalized = normalize_image(base)
    return [
        ("normalized", normalized),
        ("original", base),
    ]


def detect_text_regions(image: Image.Image):
    rgb = pil_to_numpy_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    connected = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 1200:
            continue
        if w < 40 or h < 12:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: (item[1], item[0]))

    regions = []
    for x, y, w, h in boxes[:20]:
        pad = 8
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(rgb.shape[1], x + w + pad)
        bottom = min(rgb.shape[0], y + h + pad)
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        regions.append(crop)

    if not regions:
        regions.append(image)

    return regions


def pil_to_numpy_rgb(image: Image.Image):
    return np.array(image.convert("RGB"))


def extract_easyocr_text(image: Image.Image):
    if easyocr_reader is None:
        return ""

    results = easyocr_reader.readtext(
        pil_to_numpy_rgb(image),
        detail=1,
        paragraph=True,
        decoder="greedy",
    )

    lines = []
    for item in results:
        if len(item) < 3:
            continue
        _, text, confidence = item
        cleaned = str(text).strip()
        if not cleaned:
            continue
        if confidence is not None and confidence < 0.15:
            continue
        lines.append(cleaned)

    return "\n".join(lines).strip()


def run_trocr_ocr(image: Image.Image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated = model.generate(
        pixel_values,
        max_new_tokens=48,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()

    token_probs = []
    for step_index, step_scores in enumerate(generated.scores):
        token_index = step_index + 1
        if token_index >= generated.sequences.shape[1]:
            break
        chosen_token_id = generated.sequences[0, token_index]
        probs = torch.softmax(step_scores[0], dim=-1)
        token_probs.append(probs[chosen_token_id].item())

    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return text, confidence


def run_tesseract_ocr(image: Image.Image) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed.")
    return pytesseract.image_to_string(image).strip()


def score_text(text: str, confidence: float) -> float:
    cleaned = text.strip()
    if not cleaned:
        return -1.0

    alpha_count = sum(char.isalpha() for char in cleaned)
    digit_count = sum(char.isdigit() for char in cleaned)
    useful_count = alpha_count + digit_count
    penalty = 0.0
    if useful_count <= 1:
        penalty += 0.4
    if len(cleaned) <= 2:
        penalty += 0.3

    return confidence + min(len(cleaned) / 40.0, 0.4) + min(useful_count / 20.0, 0.3) - penalty


def run_ocr(image: Image.Image):
    variants = build_ocr_variants(image)

    if easyocr_reader is not None:
        best_image = image
        best_text = ""
        best_len = -1

        for _, variant in variants:
            text = extract_easyocr_text(variant)
            if len(text) > best_len:
                best_image = variant
                best_text = text
                best_len = len(text)

        if best_text.strip():
            return best_image, best_text

    if processor is not None and model is not None:
        best_image = image
        best_text = ""
        best_score = float("-inf")

        for _, variant in variants:
            region_texts = []
            region_scores = []

            for region in detect_text_regions(variant):
                text, confidence = run_trocr_ocr(region)
                cleaned = text.strip()
                if cleaned:
                    region_texts.append(cleaned)
                    region_scores.append(score_text(cleaned, confidence))

            text = "\n".join(region_texts).strip()
            confidence = sum(region_scores) / len(region_scores) if region_scores else 0.0
            current_score = score_text(text, confidence)
            if current_score > best_score:
                best_image = variant
                best_text = text
                best_score = current_score

        if best_text.strip():
            return best_image, best_text
        return best_image, "No text detected by TrOCR."

    if ocr_backend and ocr_backend.startswith("Tesseract"):
        normalized = normalize_image(image)
        text = run_tesseract_ocr(normalized)
        if text:
            return normalized, text
        return normalized, "No text detected by Tesseract."

    return image, (
        "OCR backend is unavailable.\n\n"
        "To enable TrOCR, connect to the internet once so the Hugging Face model can download.\n"
        "Or install the Tesseract desktop engine and keep `pytesseract` available.\n\n"
        f"Startup details: {model_error or 'No OCR backend could be initialized.'}"
    )


def process_pipeline(image: Image.Image):
    if image is None:
        return default_img, default_img, "Upload a prescription image to see results."

    try:
        normalized_img, ocr_text = run_ocr(image)
    except Exception as error:
        normalized_img = normalize_image(image)
        ocr_text = f"OCR failed: {error}"

    return image, normalized_img, ocr_text


load_trocr()
load_easyocr()
if processor is None or model is None:
    detect_tesseract()

default_img = load_default_image()
initial_original, initial_normalized, initial_text = process_pipeline(default_img)
status_lines = [f"OCR backend: {ocr_backend or 'Unavailable'}"]
if model_error:
    status_lines.append("TrOCR download/load issue detected. The app will still open.")
if easyocr_error:
    status_lines.append("EasyOCR could not initialize yet. TrOCR fallback is active.")
status_text = "\n".join(status_lines)


with gr.Blocks() as demo:
    gr.Markdown("# DoctorScribble2Text: Prescription OCR")
    gr.Markdown("Upload a prescription image and extract text from it.")
    gr.Markdown(status_text)

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Prescription", value=default_img)
        run_btn = gr.Button("Process")

    with gr.Row():
        original = gr.Image(type="pil", label="Original", value=initial_original)
        normalized = gr.Image(type="pil", label="Normalized", value=initial_normalized)
        text_out = gr.Textbox(label="OCR Text", lines=8, value=initial_text)

    run_btn.click(
        process_pipeline,
        inputs=input_img,
        outputs=[original, normalized, text_out],
        queue=False,
    )
    input_img.change(
        process_pipeline,
        inputs=input_img,
        outputs=[original, normalized, text_out],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch()
