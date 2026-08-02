# Architecture

## System Boundary

DoctorScribble2Text is a research OCR pipeline. It does not diagnose, prescribe,
validate medicines, or replace professional review.

## Components

### Gradio interface

**app.py** owns model startup, document preprocessing, OCR orchestration, and the
browser interface.

### Document segmentation

OpenCV adaptive thresholding and horizontal row projection identify line-like
regions. The detector searches the complete page and retains up to 18 regions
on CPU or 20 on CUDA. A contour-based method handles unusual layouts and
tightly cropped inputs.

### Printed OCR

Tesseract processes the original page when its desktop executable is available.
On Windows the application checks:

~~~text
C:\Program Files\Tesseract-OCR\tesseract.exe
~~~

### Handwriting adaptation

**handwriting_gan.py** defines a grayscale ResNet CycleGAN generator. Input
images are resized with aspect-ratio-preserving white padding, normalized to
-1 to 1, and transformed at 256 x 64 resolution.

The runtime loads only the medical-to-CVL state dictionary with strict key
validation. Tensor-only loading is requested where supported by PyTorch.

### Handwritten OCR

Microsoft TrOCR processes line candidates in batches. Each line is evaluated in
its original form and, when the checkpoint is available, in GAN-adapted form.
Candidate selection combines token confidence with basic output quality
signals.

### Output

Printed document OCR and selected handwriting OCR are returned as separate
sections. This preserves raw evidence and avoids silently rewriting uncertain
medical text.

## Startup Sequence

1. Try cached handwritten TrOCR.
2. Download handwritten TrOCR if necessary.
3. Try the printed TrOCR fallback.
4. Load EasyOCR only when TrOCR is unavailable.
5. Load and validate the CycleGAN checkpoint.
6. Detect the local Tesseract executable.
7. Start Gradio.

## Performance

The application avoids automatic processing on upload. OCR begins only after
the Process command. CPU inference is batched in groups of four and can require
60 to 100 seconds for a full prescription. CUDA uses larger batches.

## Trust Model

- GAN output is never assumed to preserve every character.
- Original regions always remain OCR candidates.
- No medicine dictionary silently changes recognized text.
- Model files and datasets are not committed to Git.
- Prescription images should be treated as sensitive medical data.

## Failure Modes

- Poor lighting can connect background shadows into text components.
- Cursive writing can be segmented into incorrect line boundaries.
- TrOCR can produce fluent but incorrect text.
- Tesseract is weak on handwriting.
- A missing checkpoint disables GAN adaptation but not OCR startup.
- A missing Tesseract binary disables printed OCR but not TrOCR.
