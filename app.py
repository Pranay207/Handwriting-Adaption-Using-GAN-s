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

from handwriting_gan import HandwritingAdapter

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
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-base-printed",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None
trocr_model_name = None
easyocr_reader = None
ocr_backend = None
model_error = None
easyocr_error = None
tesseract_available = False
gan_adapter = None
gan_error = None

GAN_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "handwriting_cyclegan",
    "handwriting_cyclegan.pt",
)


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
    global ocr_backend, tesseract_available

    if pytesseract is None:
        return

    windows_binary = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_binary):
        pytesseract.pytesseract.tesseract_cmd = windows_binary

    try:
        version = pytesseract.get_tesseract_version()
        tesseract_available = True
        if ocr_backend is None:
            ocr_backend = f"Tesseract OCR ({version})"
    except Exception:
        tesseract_available = False

def load_handwriting_gan():
    global gan_adapter, gan_error

    if not os.path.exists(GAN_CHECKPOINT):
        gan_error = f"Checkpoint not found: {GAN_CHECKPOINT}"
        return

    try:
        gan_adapter = HandwritingAdapter(GAN_CHECKPOINT, device)
    except Exception as error:
        gan_adapter = None
        gan_error = str(error)


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


def detect_text_regions(image: Image.Image, max_regions=None):
    rgb = pil_to_numpy_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    if max_regions is None:
        max_regions = 20 if device == "cuda" else 18

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        71,
        15,
    )

    # Handwritten prescriptions are primarily line-oriented. Row projection
    # avoids spending the CPU budget on each small printed header component.
    row_counts = (threshold > 0).sum(axis=1)
    active = (row_counts > max(8, int(width * 0.02))).astype(np.uint8)
    active = cv2.morphologyEx(
        active[:, None],
        cv2.MORPH_CLOSE,
        np.ones((9, 1), dtype=np.uint8),
    ).ravel()
    changes = np.diff(np.r_[0, active.astype(np.int16), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    line_boxes = []
    for top, bottom in zip(starts, ends):
        band_height = int(bottom - top)
        center = (top + bottom) / 2 / height
        if band_height < 12 or band_height > int(height * 0.12):
            continue
        if center < 0.03 or center > 0.98:
            continue

        _, xs = np.where(threshold[top:bottom] > 0)
        if xs.size == 0:
            continue

        left = max(0, int(np.percentile(xs, 1)) - 12)
        right = min(width, int(np.percentile(xs, 99)) + 12)
        if right - left < 80:
            continue

        pad_y = max(8, int(band_height * 0.2))
        line_boxes.append(
            (
                left,
                max(0, int(top) - pad_y),
                right,
                min(height, int(bottom) + pad_y),
            )
        )

    if line_boxes:
        if len(line_boxes) > max_regions:
            selected = np.linspace(0, len(line_boxes) - 1, max_regions, dtype=int)
            line_boxes = [line_boxes[index] for index in selected]
        return [image.crop(box).convert("RGB") for box in line_boxes]

    # Fallback for tightly cropped word images and unusual page layouts.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    connected = cv2.dilate(threshold, kernel, iterations=1)
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    page_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 1200 or area > page_area * 0.25:
            continue
        if w < 40 or h < 12:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: (item[1], item[0]))
    if len(boxes) > max_regions:
        selected = np.linspace(0, len(boxes) - 1, max_regions, dtype=int)
        boxes = [boxes[index] for index in selected]

    regions = []
    for x, y, w, h in boxes:
        pad = 8
        crop = image.crop(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(width, x + w + pad),
                min(height, y + h + pad),
            )
        ).convert("RGB")
        regions.append(crop)

    return regions or [image.convert("RGB")]

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


def run_trocr_batch(images):
    if not images:
        return []

    results = []
    batch_size = 8 if device == "cuda" else 4

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        pixel_values = processor(
            images=batch,
            return_tensors="pt",
        ).pixel_values.to(device)

        with torch.inference_mode():
            generated = model.generate(
                pixel_values,
                max_new_tokens=48,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

        texts = processor.batch_decode(
            generated.sequences,
            skip_special_tokens=True,
        )
        confidence_sums = torch.zeros(len(batch), device=generated.sequences.device)
        confidence_counts = torch.zeros(len(batch), device=generated.sequences.device)
        pad_token_id = processor.tokenizer.pad_token_id

        for step_index, step_logits in enumerate(generated.scores):
            token_index = step_index + 1
            if token_index >= generated.sequences.shape[1]:
                break

            token_ids = generated.sequences[:, token_index]
            chosen_log_probs = torch.log_softmax(step_logits, dim=-1).gather(
                1,
                token_ids.unsqueeze(1),
            ).squeeze(1)
            valid = token_ids.ne(pad_token_id)
            confidence_sums += torch.where(valid, chosen_log_probs, 0.0)
            confidence_counts += valid

        confidences = (
            confidence_sums / confidence_counts.clamp_min(1)
        ).exp().tolist()

        for text, confidence in zip(texts, confidences):
            results.append((text.strip(), confidence))

    return results


def run_trocr_ocr(image: Image.Image):
    return run_trocr_batch([image])[0]

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
        base = image.convert("RGB")
        preview = ImageOps.autocontrast(ImageOps.grayscale(base)).convert("RGB")
        regions = detect_text_regions(base)

        candidate_images = []
        candidate_groups = []
        for region in regions:
            group = [len(candidate_images)]
            candidate_images.append(region)

            if gan_adapter is not None:
                group.append(len(candidate_images))
                candidate_images.append(gan_adapter.adapt(region))

            candidate_groups.append(group)

        predictions = run_trocr_batch(candidate_images)
        region_texts = []

        for group in candidate_groups:
            best_text = ""
            best_score = float("-inf")
            for candidate_index in group:
                text, confidence = predictions[candidate_index]
                candidate_score = score_text(text, confidence)
                if candidate_score > best_score:
                    best_text = text.strip()
                    best_score = candidate_score

            if best_text:
                region_texts.append(best_text)

        handwriting_text = "\n".join(region_texts).strip()
        printed_text = ""
        if tesseract_available:
            printed_text = run_tesseract_ocr(base)

        sections = []
        if printed_text:
            sections.append("DOCUMENT TEXT\n" + printed_text)
        if handwriting_text:
            sections.append("HANDWRITING TEXT\n" + handwriting_text)

        if sections:
            return preview, "\n\n".join(sections)
        return preview, "No text detected."
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
if processor is None or model is None:
    load_easyocr()
load_handwriting_gan()
detect_tesseract()

default_img = load_default_image()
initial_original = default_img
initial_normalized = normalize_image(default_img)
initial_text = "Ready. Upload a prescription image and select Process."
status_lines = [f"OCR backend: {ocr_backend or 'Unavailable'}"]
status_lines.append(
    f"Printed-text OCR: {'Ready' if tesseract_available else 'Unavailable'}"
)
if gan_adapter is not None:
    status_lines.append(
        f"Handwriting GAN: Ready ({gan_adapter.completed_steps} training steps)"
    )
else:
    status_lines.append("Handwriting GAN: Unavailable")
if model_error:
    status_lines.append("TrOCR download/load issue detected. The app will still open.")
if easyocr_error:
    status_lines.append("EasyOCR could not initialize yet. TrOCR fallback is active.")
if gan_error:
    status_lines.append(f"GAN load issue: {gan_error}")
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
        normalized = gr.Image(type="pil", label="Preprocessed", value=initial_normalized)
        text_out = gr.Textbox(label="OCR Text", lines=8, value=initial_text)

    run_btn.click(
        process_pipeline,
        inputs=input_img,
        outputs=[original, normalized, text_out],
        queue=False,
    )



if __name__ == "__main__":
    demo.launch()
