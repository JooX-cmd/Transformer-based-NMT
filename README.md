# 🌍 Arabic → English Neural Machine Translation

A Transformer-based Neural Machine Translation (NMT) system that translates Arabic text to English using PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![NLP](https://img.shields.io/badge/NLP-Transformers-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 🎯 Project Overview

This project implements a **Neural Machine Translation (NMT)** model from **Arabic to English** using a **Transformer encoder-decoder architecture** in PyTorch. Built as a final project for Pattern Recognition course.

```
Arabic Input:  "مرحبا كيف حالك"
     ↓
[Transformer Model]
     ↓
English Output: "Hello how are you"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER MODEL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                      ┌─────────────┐      │
│  │   ENCODER   │                      │   DECODER   │      │
│  ├─────────────┤                      ├─────────────┤      │
│  │ Multi-Head  │                      │ Masked      │      │
│  │ Self-Attn   │──────────────────────│ Self-Attn   │      │
│  ├─────────────┤      Cross           ├─────────────┤      │
│  │ Feed        │      Attention       │ Cross-Attn  │      │
│  │ Forward     │                      ├─────────────┤      │
│  ├─────────────┤                      │ Feed        │      │
│  │ Add & Norm  │                      │ Forward     │      │
│  └─────────────┘                      └─────────────┘      │
│        ↑                                    ↓              │
│  ┌─────────────┐                      ┌─────────────┐      │
│  │ Positional  │                      │ Linear +    │      │
│  │ Encoding    │                      │ Softmax     │      │
│  └─────────────┘                      └─────────────┘      │
│        ↑                                    ↓              │
│  Arabic Input                         English Output       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔤 **Transformer Architecture** | Full encoder-decoder with multi-head attention |
| 📊 **Positional Encoding** | Sinusoidal encoding from original paper |
| 📚 **Custom Vocabulary** | Word-level tokenization with special tokens |
| 🔄 **Beam Search** | Advanced decoding for better translations |
| 📈 **Training Visualization** | Loss curves and attention maps |

## 📁 Project Structure

```
arabic-english-nmt/
├── README.md
├── requirements.txt
├── notebooks/
│   └── translation_transformer.ipynb
├── data/
│   └── .gitkeep (add ara_.txt here)
├── models/
│   └── .gitkeep (saved model weights)
└── src/
    └── .gitkeep
```

## 🛠️ Technical Details

### Model Components

| Component | Description |
|-----------|-------------|
| **Embedding** | Word embeddings for Arabic & English |
| **Positional Encoding** | Sinusoidal position information |
| **Multi-Head Attention** | Self-attention mechanism |
| **Feed Forward** | Position-wise feed-forward network |
| **Layer Normalization** | Stabilizes training |

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<pad>` | Padding sequences |
| `<bos>` | Beginning of sentence |
| `<eos>` | End of sentence |
| `<unk>` | Unknown words |

### Hyperparameters

```python
# Model Configuration
D_MODEL = 256        # Embedding dimension
N_HEADS = 8          # Attention heads
N_LAYERS = 4         # Encoder/Decoder layers
D_FF = 512           # Feed-forward dimension
DROPOUT = 0.1        # Dropout rate
MAX_LEN = 100        # Maximum sequence length

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 20
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Libraries

```
torch>=1.9.0
numpy
pandas
matplotlib
tqdm
```

### Dataset

The model uses a parallel Arabic-English corpus (`ara_.txt`):
- Tab-separated format
- Column 0: English sentence
- Column 1: Arabic sentence

### Usage

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/arabic-english-nmt.git
cd arabic-english-nmt
```

2. **Add your dataset**
```bash
# Place ara_.txt in the data/ folder
```

3. **Run the notebook**
```bash
jupyter notebook notebooks/translation_transformer.ipynb
```

4. **Train the model**
```python
# Training is handled in the notebook
# Model will be saved to models/ folder
```

5. **Translate text**
```python
# After training
translate("مرحبا بك في مصر")
# Output: "welcome to egypt"
```

## 📊 Training Pipeline

```
1. Load Dataset
      ↓
2. Preprocess Text
      ↓
3. Build Vocabularies
      ↓
4. Numericalize (Tokens → IDs)
      ↓
5. Create DataLoaders
      ↓
6. Initialize Transformer
      ↓
7. Train with Cross-Entropy Loss
      ↓
8. Evaluate with BLEU Score
      ↓
9. Inference with Beam Search
```

## 📈 Results

| Metric | Score |
|--------|-------|
| Training Loss | ~ |
| Validation Loss | ~ |
| BLEU Score | ~ |

*Results depend on dataset size and training duration*

## 🔍 Example Translations

| Arabic | English (Predicted) |
|--------|---------------------|
| مرحبا | hello |
| كيف حالك | how are you |
| أنا طالب | i am a student |
| مصر جميلة | egypt is beautiful |

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual Guide
- PyTorch Documentation

## 🎓 Course Information

- **Course**: Pattern Recognition
- **Project**: Neural Machine Translation
- **University**: Helwan University

## 👨‍💻 Author

**Joox**
- IoT & AI Developer @ VoltX
- CS Student @ Helwan University '27





<p align="center">
  <b>⭐ Star this repo if you find it useful!</b>
</p>
