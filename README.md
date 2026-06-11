---
title: Artha ML
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app_hf.py
pinned: false
---

# ArthaML — Kannada-English Neural Machine Translation

ArthaML is an open-source, custom-built Neural Machine Translation (NMT) system for translating English to Kannada. Built entirely from scratch using PyTorch, the system implements a classic Sequence-to-Sequence (Seq2Seq) architecture enhanced with Bahdanau (Additive) Attention.

## 🚀 Features
- **Custom Architecture**: Seq2Seq LSTM with dynamic Bahdanau Attention mechanism.
- **Trained from Scratch**: No pre-trained foundational models were used. The model learns entirely from a 50k-sample subset of the Samanantar corpus.
- **End-to-End Pipeline**: Includes complete, modular scripts for data fetching, cleaning, vocabulary building, training, evaluation, and inference.
- **Standardized Evaluation**: Uses SacreBLEU for rigorous translation quality tracking over epochs.
- **Interactive Interfaces**: Features both a fast CLI translation engine and a responsive Gradio web application for demonstrations.

## 🧠 Architecture
- **Encoder**: Embedding Layer (21,551 vocab size, 256 dim) $\rightarrow$ 2-Layer LSTM (512 hidden units) with 0.3 Dropout.
- **Attention**: Bahdanau Additive Attention dynamically computing alignment scores over the encoder's hidden states.
- **Decoder**: Embedding Layer (26,802 vocab size, 256 dim) $\rightarrow$ Attention Context $\rightarrow$ 2-Layer LSTM (512 hidden units).
- **Optimization**: Trained using the Adam optimizer with Cross-Entropy Loss (ignoring `<PAD>` tokens) and Gradient Clipping to prevent exploding gradients.

## ⚙️ Setup and Installation
The project uses modern Python packaging via `uv`.

```bash
# Clone the repository
git clone https://github.com/LikhinMN/artha-ml.git
cd artha-ml

# Install dependencies into your environment
uv pip install torch torchvision torchaudio datasets transformers sacrebleu gradio
```

## 💻 Usage

### 1. Web Application Demo
Launch the interactive web UI powered by Gradio:
```bash
python app.py
```
This will start a local server at `http://127.0.0.1:7860/` where you can input English text and see the Kannada translation dynamically generated along with inference timing.

### 2. Command Line Interface
Run a single translation query directly from the terminal:
```bash
python scripts/translate.py "The weather is beautiful today."
```

Or start the interactive CLI loop for continuous translation:
```bash
python scripts/translate.py --interactive
```

## 🏋️ Training Pipeline
If you wish to retrain the model from scratch on your own hardware:

1. **Download and Clean Data**: Fetches 50,000 samples from the `ai4bharat/samanantar` dataset and normalizes the text.
   ```bash
   python scripts/load_data.py
   python scripts/clean_data.py
   ```
2. **Build Vocabulary**: Generates English and Kannada word-to-index mappings.
   ```bash
   python scripts/build_vocab.py
   ```
3. **Train the Model**: Trains the Seq2Seq network, automatically saving the best checkpoint (`best_model.pth`) based on Validation Loss.
   ```bash
   python scripts/train.py
   ```

## 📁 Project Structure
```text
artha-ml/
├── arthaml/
│   ├── data/          # Dataset loaders and Vocabulary classes
│   ├── models/        # PyTorch architecture (Encoder, Decoder, Seq2Seq)
│   ├── evaluation/    # BLEU score logic
│   └── inference.py   # Main Translator class for greedy decoding
├── data/              # Raw and processed corpora + vocab JSONs
├── models/            # Saved weights (.pth checkpoints)
├── scripts/           # Standalone scripts for the ML pipeline
└── app.py             # Gradio web interface
```

## 📄 License
This project is open-source and available under the MIT License. Feel free to fork, experiment, and contribute!
