✍️ Handwriting Style Adaptation Using GANs

Normalize diverse handwritten styles, such as doctor prescriptions, using GANs to improve OCR recognition accuracy.

📸 Overview

Doctors’ handwriting often varies widely, making prescriptions and notes difficult to read. This project:

Normalizes diverse handwriting styles 🖌️

Improves OCR recognition accuracy 🎯

Provides a simple pipeline for preprocessing handwritten images 🖥️

🚀 Live Demo / Preview

Try it Live Now  https://huggingface.co/spaces/Pranay2007/GAN-For-Prescriptions


How to Use

Collect handwritten images (doctor prescriptions, forms, etc.) 📄

Normalize handwriting styles using the GAN model 🎨

Apply OCR to extract text from normalized images 🔹

Optionally, run the Streamlit demo for interactive testing 🖥️

🛠️ Tech Stack

Python 3.x – Core logic and processing 🐍

PyTorch / TensorFlow – GAN for handwriting style adaptation 🔥

OpenCV / NumPy / Matplotlib – Image preprocessing & visualization 🎨

Tesseract OCR / CNN-LSTM – Handwriting recognition 🖋️

Streamlit (optional) – Web demo interface 🚀

🧠 How It Works

Prepare Dataset: Place handwritten images in data/raw/ 📁

Train GAN: Learn to normalize diverse handwriting styles 🎨

Apply GAN: Generate normalized versions of handwritten images 🔹

OCR Recognition: Extract text from normalized images using Tesseract or CNN-LSTM 🔹

Output: Get accurate, readable text for digital use ✨.
