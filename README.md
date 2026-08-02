# DoctorScribble2Text

Prescription handwriting adaptation and OCR using a CycleGAN, Microsoft TrOCR,
OpenCV, PyTorch, and Gradio.

## Current Status

The project is implemented and runs locally. Its inference flow is:

```text
Prescription image
    -> text-region detection and preprocessing
    -> medical-to-CVL CycleGAN adaptation
    -> original and adapted TrOCR candidates
    -> confidence-based OCR result
```

The original image remains an OCR candidate because GAN adaptation does not
improve every word. The application selects the stronger OCR result for each
detected region.

## Features

- Gradio prescription upload interface
- Handwritten text-region detection with OpenCV
- Grayscale CycleGAN inference at 256 x 64 pixels
- Microsoft TrOCR handwritten recognition
- Original-image fallback when GAN adaptation is weaker
- EasyOCR and Tesseract fallback support
- CPU and CUDA device selection
- Offline loading after TrOCR has been cached once

## Project Files

```text
.
|-- app.py
|-- handwriting_gan.py
|-- models/
|   `-- handwriting_cyclegan/
|       `-- handwriting_cyclegan.pt
|-- requirements.txt
|-- README.md
`-- PROJECT_REFERENCE.md
```

The trained `.pt` checkpoint is ignored by Git because it is approximately
85 MB. It must be placed at:

```text
models/handwriting_cyclegan/handwriting_cyclegan.pt
```

## Datasets Used

| Dataset | Purpose | Local observations |
|---|---|---|
| RxHandBD | Medical handwriting domain | 5,578 labeled word images |
| CVL cropped database | General handwriting domain | 1,604 page images; word regions were extracted during training |
| Bilingual prescription dataset | Bangla and English prescription research | 1,000 image pairs |

CycleGAN training used unpaired CVL word crops as domain A and RxHandBD medical
word images as domain B. The bilingual dataset was not used as direct OCR
ground truth because its available CSV descriptions are not reliable
word-for-word transcriptions.

Check each dataset's original license and terms before redistributing it. The
datasets are intentionally excluded from Git under `data/`.

## Trained GAN

The supplied checkpoint was trained on Kaggle GPU for 2,000 update steps.

- Input and output: one-channel grayscale
- Resolution: 256 x 64
- Generators: six residual blocks
- Discriminators: PatchGAN
- Training: unpaired CycleGAN
- Losses: adversarial, cycle consistency, identity, and structure preservation

Both generator directions are stored in the checkpoint:

- `generator_cvl_to_medical`
- `generator_medical_to_cvl`

The OCR application uses `generator_medical_to_cvl` to make medical words
closer to the general handwriting style recognized by TrOCR.

## Installation

Python 3.10 or 3.11 is recommended.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TrOCR downloads from Hugging Face on its first run. An internet connection is
required once unless the model is already cached.

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:7860
```

The interface should report:

```text
OCR backend: TrOCR (handwritten, cached, cpu)
Handwriting GAN: Ready (2000 training steps)
```

Upload a prescription and select **Process**. CPU processing can be slow when a
page contains many detected text regions. CUDA is used automatically when
available.

## Verification

The current checkpoint has been verified for:

- Safe tensor-only checkpoint loading
- Strict generator state-dictionary compatibility
- A real RxHandBD image forward pass
- Nonblank 256 x 64 generated output
- Cached TrOCR startup
- Gradio HTTP response on the local server

## Limitations

- The GAN was trained for only 2,000 steps and can blur or distort some words.
- OCR quality depends heavily on image sharpness, lighting, and segmentation.
- The bilingual dataset is not currently used for Bangla OCR.
- TrOCR in this application recognizes English handwriting.
- Prescription OCR results must be reviewed by a person.
- This software is a research prototype, not a medical decision system.

## Improving the Model

For production-quality research, train for more epochs with validation
checkpoints, evaluate character error rate and word error rate on held-out
transcriptions, improve word segmentation, and add a separately evaluated
Bangla OCR model.

## License

Review the upstream repository license and each dataset license before
distribution. Model and dataset licenses may differ from the application code
license.
