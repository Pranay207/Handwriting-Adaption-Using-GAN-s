# Model Checkpoint

Place the exported checkpoint at:

~~~text
models/handwriting_cyclegan/handwriting_cyclegan.pt
~~~

The checkpoint is ignored by Git. Do not rename it unless **GAN_CHECKPOINT** in
**app.py** is updated.

## Expected Keys

~~~text
generator_cvl_to_medical
generator_medical_to_cvl
discriminator_cvl
discriminator_medical
completed_steps
image_height
image_width
~~~

Expected dimensions are 64 x 256 with one grayscale channel. The current
development checkpoint records 2,000 completed steps.

## Verify

~~~bash
python scripts/verify_setup.py
~~~

Expected output includes:

~~~text
CycleGAN: ready (2000 training steps)
~~~

## Distribution

Publish checkpoints as versioned release assets or through a model registry.
Include a SHA-256 checksum, training configuration, dataset licenses, evaluation
results, and model license. Avoid committing large binary files directly.
