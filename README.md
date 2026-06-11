---
title: Artha ML
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app_hf.py
pinned: false
---

# 🌍 ArthaML

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LikhinMN/artha-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

ArthaML is an open-source, custom-built Neural Machine Translation (NMT) system for translating English to Kannada. Built from scratch using PyTorch and a Sequence-to-Sequence (Seq2Seq) architecture with Bahdanau Attention, it is designed for researchers, developers, and NLP enthusiasts who want an accessible, hackable baseline for Indic language translation.

---

## 🚀 Getting Started

### Prerequisites
- **OS**: Linux, macOS, or Windows (WSL recommended)
- **Runtime**: Python 3.12+
- **Hardware**: CPU or NVIDIA GPU (CUDA supported)

### Installation
Clone the repository and install the dependencies using `uv` (or standard `pip`):

```bash
git clone https://github.com/LikhinMN/artha-ml.git
cd artha-ml

# Install dependencies
uv pip install torch torchvision torchaudio datasets transformers sacrebleu gradio
```

### Quick Start
You can run a local web server instantly to test translations through your browser.

```bash
python app.py
```
*Navigate to `http://127.0.0.1:7860/` to interact with the translation model!*

---

## 💻 Usage & Examples

### Core Workflows

**1. Command Line Interface (CLI)**
Run one-off translations directly in your terminal:
```bash
python scripts/translate.py "The weather is beautiful today."
# Output: ಇಂದು ಹವಾಮಾನ ಸುಂದರವಾಗಿದೆ
```
For continuous translation, use interactive mode:
```bash
python scripts/translate.py --interactive
```

**2. Training the NMT Pipeline from Scratch**
ArthaML provides end-to-end scripts to train the model yourself:
```bash
python scripts/load_data.py   # Downloads 50k samples from Samanantar
python scripts/clean_data.py  # Cleans and normalizes unicode text
python scripts/build_vocab.py # Builds English and Kannada indices
python scripts/train.py       # Trains the Seq2Seq LSTM network
```

### Configuration
- `app.py` and `app_hf.py` handle the Gradio web UI. `app_hf.py` includes additional soft length warnings and OOD token handling optimized for public demonstration.
- Training parameters (like batch size and epochs) are configured directly inside `scripts/train.py`.

---

## 🤝 Support & Contributions

**Support:**
If you encounter bugs, have questions, or need help with setup, please [open an issue](https://github.com/LikhinMN/artha-ml/issues) on GitHub.

**Contributing:**
Contributions are always welcome! 
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License & Credits

**License:** 
Distributed under the MIT License. See `LICENSE` for more information.

**Acknowledgments:**
- The amazing [AI4Bharat Samanantar](https://ai4bharat.iitm.ac.in/samanantar/) dataset which provided the English-Kannada corpus.
- The foundational [Bahdanau Attention mechanism](https://arxiv.org/abs/1409.0473) (2014) paper.
