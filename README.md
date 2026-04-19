# 🩺 RxGAN: Prescription Handwriting Normalization  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-GAN-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>  

## 📸 Overview  
Doctors' handwriting often varies widely, making prescriptions and notes **difficult to read**.  
**RxGAN** tackles this by:  

- 🖌️ Normalizing diverse handwriting styles  
- 🎯 Improving OCR recognition accuracy  
- 🖥️ Providing a simple preprocessing pipeline for handwritten images  

---

## 🚀 Live Demo / Preview  

👉 [Try it Live on Hugging Face Spaces](https://huggingface.co/spaces/Pranay2007/GAN-For-Prescriptions)  


---

## ⚙️ How to Use  

1. 📄 Collect handwritten images (doctor prescriptions, forms, etc.)  
2. 🎨 Normalize handwriting styles using the GAN model  
3. 🔹 Apply OCR to extract text from normalized images  
4. 🖥️ (Optional) Run the Streamlit demo for interactive testing  


## 🛠️ Tech Stack  

| Tool / Library | Purpose |
|----------------|---------|
| 🐍 **Python 3.x** | Core programming language |
| 🔥 **PyTorch / TensorFlow** | GAN-based handwriting style adaptation |
| 🎨 **OpenCV / NumPy / Matplotlib** | Image preprocessing & visualization |
| 🖋️ **Tesseract OCR / CNN-LSTM** | Handwriting recognition |
| 🚀 **Streamlit (optional)** | Interactive web demo interface |

---

## 🧠 How It Works  

1. 📁 **Prepare Dataset** → Place handwritten images in `data/raw/`  
2. 🎨 **Train GAN** → Learn to normalize diverse handwriting styles  
3. 🔹 **Apply GAN** → Generate normalized versions of handwritten images  
4. 🔹 **OCR Recognition** → Extract text using **Tesseract** or **CNN-LSTM**  
5. ✨ **Output** → Get accurate, readable text for digital use  

---

## 📜 License  

This project is licensed under the **MIT License**.  

<p align="center">⚡ Built with AI to make handwriting readable 🩺</p>
