✍️ Handwriting Style Adaptation Using GANs

Normalize diverse handwritten styles, such as doctor prescriptions, using Generative Adversarial Networks (GANs) to improve OCR recognition accuracy.

📸 Overview

Doctors’ handwriting varies widely, making prescriptions and notes hard to read. This project:

Adapts and normalizes diverse handwriting styles 🖌️

Improves OCR recognition accuracy 🎯

Provides a complete preprocessing pipeline for handwritten text 🖥️

🛠️ Tech Stack

Python 3.x – Core logic and preprocessing 🐍

PyTorch / TensorFlow – GAN model for handwriting adaptation 🔥

OpenCV / NumPy / Matplotlib – Image preprocessing & visualization 🎨

Tesseract OCR / CNN-LSTM – Handwriting recognition 🖋️

Streamlit (optional) – Interactive demo interface 🚀

🧠 How It Works

Prepare Dataset: Collect handwritten images (e.g., prescriptions) and place them in data/raw/.

Train GAN: Normalize handwriting styles across the dataset 🎨.

Apply GAN: Generate normalized versions of handwritten images 🔹.

OCR Recognition: Extract text from normalized images using Tesseract or custom CNN-LSTM 🔹.

Output: Extracted text can be used for further processing or translation.
