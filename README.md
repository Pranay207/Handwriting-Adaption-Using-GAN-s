

[![CI](https://github.com/Pranay207/Handwriting-Adaption-Using-GAN-s/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranay207/Handwriting-Adaption-Using-GAN-s/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CycleGAN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Interface](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)

Research-oriented prescription OCR that combines document segmentation,
CycleGAN handwriting adaptation, Microsoft TrOCR, Tesseract, and a Gradio web
interface.

> [!WARNING]
> This project is not a medical device. OCR output can be incomplete or wrong.
> A qualified person must verify every medicine, dose, duration, and instruction.

## Overview

Medical prescriptions contain printed letterheads, patient details, handwritten
medicines, dosage schedules, advice, signatures, and footer text. This project
uses separate components for those different visual domains:

1. OpenCV detects text-line regions across the complete page.
2. Tesseract extracts printed document text when installed.
3. A grayscale CycleGAN adapts medical handwriting toward the CVL domain.
4. TrOCR evaluates the original and GAN-adapted versions of each line.
5. The stronger TrOCR candidate is retained and combined with printed OCR.

~~~text
Prescription image
        |
        +--> full-page printed OCR ----------------------+
        |                                                |
        +--> line detection --> original line --> TrOCR  |
                           \--> CycleGAN --> TrOCR       |
                                    |                    |
                           confidence selection          |
                                    +--------------------+
                                                 |
                                           combined text
~~~

## Features

- Full-page extraction covering header, body, instructions, and footer
- Adaptive line segmentation for photographed prescriptions
- Medical-to-CVL CycleGAN inference at 256 x 64 pixels
- Original-image fallback when GAN adaptation is weaker
- Batched TrOCR inference for practical CPU execution
- Printed-text extraction with local Tesseract
- CUDA acceleration when available
- Safe tensor-only checkpoint loading
- Gradio upload, preview, and OCR interface

## Repository Structure

~~~text
.
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |-- workflows/ci.yml
|   +-- pull_request_template.md
|-- data/
|   +-- README.md
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- DATASETS.md
|   +-- TRAINING.md
|-- models/
|   |-- README.md
|   +-- handwriting_cyclegan/
|       +-- handwriting_cyclegan.pt    # local, ignored by Git
|-- scripts/
|   +-- verify_setup.py
|-- tests/
|   +-- test_handwriting_gan.py
|-- app.py
|-- handwriting_gan.py
|-- requirements-dev.txt
|-- requirements.txt
|-- CONTRIBUTING.md
|-- SECURITY.md
+-- README.md
~~~

## Quick Start

### 1. Clone

~~~bash
git clone https://github.com/Pranay207/Handwriting-Adaption-Using-GAN-s.git
cd Handwriting-Adaption-Using-GAN-s
~~~

### 2. Create an environment

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Linux or macOS:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Python 3.10 or 3.11 is recommended.

### 3. Add the CycleGAN checkpoint

Place the exported Kaggle checkpoint at:

~~~text
models/handwriting_cyclegan/handwriting_cyclegan.pt
~~~

The checkpoint is approximately 85 MB and is intentionally not committed to
Git. See [models/README.md](models/README.md) for its required keys.

### 4. Install Tesseract

Tesseract is optional but recommended for printed headers and footers.

Windows:

~~~powershell
winget install --id UB-Mannheim.TesseractOCR --exact --source winget
~~~

Ubuntu or Debian:

~~~bash
sudo apt-get update
sudo apt-get install tesseract-ocr
~~~

### 5. Verify and run

~~~bash
python scripts/verify_setup.py
python app.py
~~~

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), upload a prescription,
and select **Process** once.

The first TrOCR run downloads the Hugging Face model unless it is already
cached. CPU processing of a full page can take roughly 60 to 100 seconds.

## Model

The development checkpoint was trained for 2,000 update steps with unpaired CVL
and RxHandBD word images.

| Property | Value |
|---|---|
| Input | One-channel grayscale |
| Resolution | 256 x 64 |
| Generator | ResNet with six residual blocks |
| Discriminator | PatchGAN |
| Directions | CVL to medical and medical to CVL |
| Runtime direction | Medical to CVL |
| Training losses | Adversarial, cycle, identity, and structure |

The GAN is a style adapter, not a text recognizer. TrOCR performs recognition,
and the original crop remains a candidate to reduce destructive GAN changes.

## Datasets

| Dataset | Role | Development inventory |
|---|---|---:|
| RxHandBD | Medical handwriting domain | 5,578 labeled word images |
| CVL cropped database | General handwriting domain | 1,604 page images |
| Bilingual prescription dataset | Future Bangla/English research | 1,000 image pairs |

The bilingual CSV was not treated as exact OCR ground truth because its
descriptions are not reliable word-for-word transcriptions. Dataset files are
excluded from Git. Review each source license before use or redistribution.

See [docs/DATASETS.md](docs/DATASETS.md) and
[docs/TRAINING.md](docs/TRAINING.md) for details.

## Development

~~~bash
pip install -r requirements-dev.txt
python -m py_compile app.py handwriting_gan.py
pytest -q
~~~

Pull requests are checked by GitHub Actions. See
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

## Known Limitations

- The current GAN checkpoint has only 2,000 training updates.
- TrOCR can confidently produce incorrect text for difficult handwriting.
- Generative adaptation can alter letter shapes.
- Printed and handwriting OCR sections can contain duplicate text.
- Bangla OCR is not implemented.
- The project does not yet report CER or WER on a held-out clinical test set.
- CPU inference is slow because every selected line may require two OCR passes.

## Roadmap

- Publish a versioned checkpoint through GitHub Releases or a model registry
- Add held-out character error rate and word error rate evaluation
- Fine-tune OCR on verified prescription transcriptions
- Add medicine-aware review without silently rewriting raw OCR
- Add separately evaluated Bangla OCR
- Add GPU-backed deployment configuration

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Datasets](docs/DATASETS.md)
- [Model checkpoint](models/README.md)
- [Security and medical-data handling](SECURITY.md)

## License

No repository-level license file is currently included. Confirm ownership and
add an appropriate software license before redistribution. Dataset, TrOCR,
Tesseract, and checkpoint licenses must be reviewed separately.
