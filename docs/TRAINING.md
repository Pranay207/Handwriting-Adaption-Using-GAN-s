# CycleGAN Training

## Objective

Training learns unpaired translation between:

- Domain A: general CVL handwriting word crops
- Domain B: RxHandBD medical handwriting word images

The runtime uses the B-to-A generator to adapt medical handwriting toward the
general handwriting domain recognized by TrOCR.

## Input Contract

| Property | Value |
|---|---|
| Channels | 1 |
| Height | 64 |
| Width | 256 |
| Range | -1 to 1 |
| Pairing | Unpaired |

Images must be grayscale, aspect-ratio preserving, and padded with white rather
than stretched.

## Architecture

- Two ResNet generators
- Six residual blocks per generator
- Bilinear upsampling followed by convolution
- Two PatchGAN discriminators
- Instance normalization

The exported checkpoint contains:

~~~text
generator_cvl_to_medical
generator_medical_to_cvl
discriminator_cvl
discriminator_medical
completed_steps
image_height
image_width
~~~

## Losses

- Least-squares adversarial loss
- Cycle-consistency L1 loss
- Identity L1 loss
- Sobel-based structure-preservation loss

An image replay pool can be used to stabilize discriminator updates.

## Development Run

The current checkpoint was trained for 2,000 update steps on a Kaggle GPU. That
run proves the pipeline but is not sufficient evidence of production quality.

## Required Evaluation

Before claiming OCR improvement:

1. Freeze a held-out prescription test set.
2. Create verified line-level transcriptions.
3. Measure TrOCR on original crops.
4. Measure TrOCR on GAN-adapted crops.
5. Report character error rate and word error rate.
6. Report how often GAN adaptation changes correct characters.
7. Include confidence intervals and failure examples.

## Export

Export state dictionaries and metadata rather than serializing entire Python
model objects. The local application expects:

~~~text
models/handwriting_cyclegan/handwriting_cyclegan.pt
~~~

Do not commit large checkpoints directly. Publish a versioned release asset or
use a model registry with checksums and license metadata.
