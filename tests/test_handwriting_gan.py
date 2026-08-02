from PIL import Image
import torch

from handwriting_gan import Generator, MODEL_HEIGHT, MODEL_WIDTH, resize_with_padding


def test_generator_preserves_tensor_shape():
    generator = Generator().eval()
    sample = torch.zeros(1, 1, MODEL_HEIGHT, MODEL_WIDTH)

    with torch.inference_mode():
        generated = generator(sample)

    assert generated.shape == sample.shape
    assert torch.isfinite(generated).all()


def test_resize_with_padding_uses_model_dimensions():
    image = Image.new('RGB', (120, 40), color='white')
    resized = resize_with_padding(image)

    assert resized.mode == 'L'
    assert resized.size == (MODEL_WIDTH, MODEL_HEIGHT)
