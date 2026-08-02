from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn


MODEL_HEIGHT = 64
MODEL_WIDTH = 256


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class Generator(nn.Module):
    def __init__(self, residual_blocks: int = 6):
        super().__init__()
        layers = [
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        layers.extend(ResidualBlock(256) for _ in range(residual_blocks))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


def resize_with_padding(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contained = ImageOps.contain(
        grayscale,
        (MODEL_WIDTH, MODEL_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new('L', (MODEL_WIDTH, MODEL_HEIGHT), color=255)
    left = (MODEL_WIDTH - contained.width) // 2
    top = (MODEL_HEIGHT - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


class HandwritingAdapter:
    def __init__(self, checkpoint_path: Path, device: str):
        self.device = torch.device(device)
        self.generator = Generator().to(self.device)

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state = checkpoint['generator_medical_to_cvl']
        self.generator.load_state_dict(state, strict=True)
        self.generator.eval()
        self.completed_steps = int(checkpoint.get('completed_steps', 0))

    def adapt(self, image: Image.Image) -> Image.Image:
        prepared = resize_with_padding(image)
        values = np.asarray(prepared, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            generated = self.generator(tensor)

        output = generated.squeeze(0).squeeze(0).cpu().clamp(-1, 1)
        output = ((output + 1.0) * 127.5).round().byte().numpy()
        return Image.fromarray(output, mode='L').convert('RGB')
