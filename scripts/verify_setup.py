from importlib.util import find_spec
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_MODULES = {
    'cv2': 'opencv-python',
    'gradio': 'gradio',
    'numpy': 'numpy',
    'PIL': 'pillow',
    'pytesseract': 'pytesseract',
    'torch': 'torch',
    'transformers': 'transformers',
}


def main() -> int:
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if find_spec(module) is None
    ]
    if missing:
        print('Missing Python packages: ' + ', '.join(missing))
        print('Run: pip install -r requirements.txt')
        return 1

    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Python: {sys.version.split()[0]}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Device: {device}')

    checkpoint = (
        ROOT
        / 'models'
        / 'handwriting_cyclegan'
        / 'handwriting_cyclegan.pt'
    )
    if checkpoint.exists():
        from handwriting_gan import HandwritingAdapter

        adapter = HandwritingAdapter(checkpoint, 'cpu')
        print(f'CycleGAN: ready ({adapter.completed_steps} training steps)')
    else:
        print(f'CycleGAN: checkpoint missing at {checkpoint}')

    tesseract_candidates = [
        Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe'),
        Path(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'),
    ]
    tesseract_status = (
        'ready' if any(path.exists() for path in tesseract_candidates) else 'check PATH'
    )
    print(f'Tesseract desktop engine: {tesseract_status}')
    print('Setup verification completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
