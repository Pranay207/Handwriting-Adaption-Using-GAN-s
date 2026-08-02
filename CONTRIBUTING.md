# Contributing

## Before Opening a Change

1. Create a focused branch from **main**.
2. Keep datasets, checkpoints, prescriptions, and patient information out of
   Git.
3. Preserve raw OCR output. Do not silently replace uncertain medicine names.
4. Add or update tests for behavioral changes.
5. Update documentation when interfaces or model contracts change.

## Local Checks

~~~bash
pip install -r requirements-dev.txt
python -m py_compile app.py handwriting_gan.py
pytest -q
~~~

## Pull Requests

Describe the problem, implementation, validation, privacy impact, and remaining
limitations. Include screenshots only when they contain no patient-identifying
information.

## Model Changes

Report dataset provenance, licenses, split strategy, training configuration,
checkpoint checksum, CER, WER, and known failure modes. Do not claim medical
accuracy from visual samples alone.
