# Datasets

## Local Inventory

The development workflow used three datasets. All archives, extracted files,
manifests, and processed images remain outside Git.

| Dataset | Development inventory | Use |
|---|---:|---|
| RxHandBD | 5,578 labeled word images | Medical handwriting domain |
| CVL cropped database | 1,604 TIFF page images from 310 writers | General handwriting domain |
| Bilingual prescription dataset | 1,000 prescription image pairs | Future Bangla/English research |

## RxHandBD

The cleaned local inventory contained 4,463 training images and 1,115 test
images. It supplied the medical handwriting domain for unpaired CycleGAN
training.

## CVL

CVL pages were not passed directly to the GAN. Candidate word regions were
detected with thresholding, morphology, contour filtering, and
content-preserving resize/padding.

## Bilingual Prescriptions

The bilingual source contains useful prescription images, but the available CSV
descriptions are inconsistent and should not be treated as exact line-level OCR
ground truth. The dataset was therefore excluded from CycleGAN and OCR
supervision.

Future use requires manually verified Bangla and English transcriptions,
explicit train/validation/test splits, and a separately evaluated Bangla OCR
model.

## Data Layout

Recommended local structure:

~~~text
data/
|-- raw/
|   |-- rxhandbd/
|   |-- cvl/
|   +-- bilingual_prescriptions/
+-- processed/
    |-- rxhandbd/
    |-- cvl/
    +-- bilingual_prescriptions/
~~~

Only this README is tracked. Everything else under **data/** is ignored.

## Privacy and Licensing

- Remove patient identifiers before training or sharing.
- Do not commit prescription photographs.
- Preserve dataset attribution and license files.
- Verify whether redistribution and commercial use are allowed.
- Record checksums and source versions for reproducible research.
