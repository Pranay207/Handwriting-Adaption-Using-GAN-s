# Project Reference

## Purpose of This Document

This file explains how the current project is actually built based on the code present in the repository today. It is meant as a technical reference for you, while `README.md` remains in its original presentation style.

## What The Current Project Really Is

The current implementation is a small OCR web app for handwritten prescription images.

It is built with:

- `Gradio` for the browser interface
- `PyTorch` for model execution
- `Transformers` from Hugging Face
- `TrOCR` (`microsoft/trocr-base-handwritten`) for handwritten text recognition
- `Pillow` for image handling

Even though the repository name mentions GANs, the present code does not yet include a real GAN-based handwriting normalization pipeline.

## Files In The Project

```text
Handwriting-Adaption-Using-GAN-s-main/
|-- app.py
|-- README.md
|-- PROJECT_REFERENCE.md
|-- requirements.txt
`-- sample.png
```

## Main File

The main logic lives in [app.py](/c:/Users/Shiva/Downloads/Handwriting-Adaption-Using-GAN-s-main/Handwriting-Adaption-Using-GAN-s-main/app.py).

This file does four main things:

1. Imports the libraries needed for the app.
2. Loads the TrOCR model and processor.
3. Defines functions to run OCR on uploaded images.
4. Creates a Gradio interface for users to test the OCR.

## How The App Works

### 1. Model Loading

When `app.py` starts, it loads:

- `TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")`
- `VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")`

This means the app uses a pretrained Microsoft handwritten OCR model from Hugging Face instead of a custom-trained model stored in the repository.

The code then checks whether CUDA is available:

- If yes, it uses GPU
- If no, it uses CPU

## 2. Default Image Setup

The app tries to open `default_sample.png`.

If that file is not found, it creates a blank white image using PIL. This acts as a safe fallback so the interface still opens even without a default image file.

One thing to note:

- The repository contains `sample.png`
- The code looks for `default_sample.png`

So at the moment, the fallback blank image is likely being used unless you add or rename the sample file.

## 3. OCR Function

The OCR logic is handled by `run_hf_ocr(image)`.

This function:

- Converts the input image into model tensors
- Sends the tensors to the selected device
- Runs text generation using the TrOCR model
- Decodes the generated token IDs back into text

In simple terms, this is the core step where the handwriting image becomes machine-readable text.

## 4. Processing Pipeline

The app uses `process_pipeline(image)` as the user-facing processing function.

Current behavior:

- If no image is given, it returns placeholder images and a help message
- If an image is uploaded, it stores that same image as `normalized_img`
- It then runs OCR on `normalized_img`
- Finally, it returns:
  - Original image
  - Normalized image
  - OCR text

Important detail:

The "normalized" image is not actually transformed right now. It is simply the original uploaded image passed through unchanged.

## 5. Gradio Interface

The user interface is created with `gr.Blocks()`.

The interface includes:

- A title and short description
- An image upload component
- A `Process` button
- Three outputs:
  - Original image
  - Normalized image
  - OCR text

When the button is clicked, Gradio calls:

`process_pipeline(input_image)`

and shows the returned outputs in the UI.

## Real Build Flow

Here is the actual build flow of the application:

1. User runs `python app.py`
2. Python loads the TrOCR model from Hugging Face
3. Gradio starts a local web server
4. User uploads a prescription image in the browser
5. The image is sent to `process_pipeline`
6. `process_pipeline` calls `run_hf_ocr`
7. TrOCR predicts the text
8. Gradio displays the images and extracted OCR result

## Dependencies

The current `requirements.txt` contains:

- `torch`
- `torchvision`
- `gradio`
- `pytesseract`
- `opencv-python`
- `pillow`
- `transformers`
- `sentencepiece`

### Which dependencies are actually used now

Directly used in `app.py`:

- `gradio`
- `pillow`
- `torch`
- `transformers`

Listed but not currently used in code:

- `torchvision`
- `pytesseract`
- `opencv-python`
- `sentencepiece`

`sentencepiece` may still be needed indirectly by transformer tokenization/model loading depending on environment, so it is reasonable to keep it.

## Important Code Observations

From reviewing the current code, these are the main technical observations:

- `app.py` contains duplicated imports
- `app.py` contains duplicated function definitions
- `app.py` contains duplicated Gradio interface code
- The app works more like an OCR prototype than a GAN project
- The documentation in the original README is broader than the implementation currently present

## What Is Implemented Right Now

- Handwritten OCR using a pretrained TrOCR model
- Local web app using Gradio
- CPU/GPU support through PyTorch
- Basic image input and OCR result display

## What Is Not Yet Implemented

- GAN model training
- GAN-based handwriting normalization
- Dataset loading and training pipeline
- Custom preprocessing stages
- End-to-end prescription understanding pipeline

## How To Run It

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Suggested Next Steps

If you want this project to become cleaner and closer to the README vision, these would be the best next improvements:

1. Remove duplicate code from `app.py`
2. Rename `sample.png` to `default_sample.png` or update the code
3. Split model logic and UI logic into separate files
4. Add preprocessing before OCR
5. Add a real normalization stage if you want the GAN idea to become real

## Short Summary

The current repository is best described as a handwritten prescription OCR demo built with Gradio and Hugging Face TrOCR. It is a good prototype, but the GAN portion is still a future idea rather than something implemented in the code today.
