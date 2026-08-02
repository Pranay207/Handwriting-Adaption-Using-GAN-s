# Project Reference

## Implemented System

This repository now contains a working prescription handwriting adaptation and
OCR application.

The application combines:

- Gradio for the browser interface
- OpenCV for handwritten text-region detection
- A custom grayscale CycleGAN generator for style adaptation
- Microsoft TrOCR for English handwritten OCR
- Optional EasyOCR and Tesseract fallbacks

## Runtime Flow

1. The user uploads a prescription image.
2. OpenCV detects likely handwritten text regions.
3. Each region is evaluated in its original form.
4. The trained medical-to-CVL generator creates an adapted candidate.
5. TrOCR recognizes both candidates.
6. The application keeps the candidate with the stronger OCR score.
7. Gradio displays the preprocessing result and extracted text.

## GAN Checkpoint

Expected location:

```text
models/handwriting_cyclegan/handwriting_cyclegan.pt
```

The checkpoint contains two generators, two discriminators, image dimensions,
and the completed training-step count. Runtime loading uses the
`generator_medical_to_cvl` state dictionary with strict key validation.

The generator architecture is defined in `handwriting_gan.py`. It accepts a
one-channel 256 x 64 tensor normalized to the range -1 to 1.

## OCR Backends

The preferred backend is:

```text
microsoft/trocr-base-handwritten
```

The printed TrOCR model, EasyOCR, and Tesseract are fallbacks. EasyOCR is not
initialized when handwritten TrOCR is available, avoiding unnecessary startup
downloads.

## Data

Local datasets and processed images are stored under `data/` and ignored by
Git. The application does not require the training datasets for inference; it
only requires the trained checkpoint and cached or downloadable OCR model.

## Important Boundary

The current GAN is a style-adaptation component, not an OCR model. TrOCR
performs character recognition. GAN output is treated as an optional candidate
because generative models can alter letter shapes.

This project is suitable for research and demonstration. Extracted prescription
text must not be used for clinical decisions without human verification.
