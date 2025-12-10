# 💬🧠 Twitter Emotion Recognition using Deep Learning & Transformers
### Deep Learning & Applications (UEC642) — Final Project

A complete end-to-end system for emotion classification in tweets using multiple deep learning approaches, ranging from classical LSTMs to modern Transformer-based models.  
This project evaluates three separate models and compares their performance:

- Original LSTM (Random Embeddings)  
- Improved LSTM (GloVe Pre-trained Embeddings)  
- RoBERTa Transformer (State-of-the-Art)  

This repository contains the full code, evaluation pipeline, comparison metrics, visualizations, and model benchmarking.

---

## 👨‍🎓 Submitted To
**Dr. Gaganpreet Kaur**

## 👨‍🎓 Submitted By
- **Kanav Kukreja — 102215145**  
- **Priyanshu — 102215164**  
- **Vinaayak Kumar Puri — 102215165**  
- **Punya Arora — 102215186**

---

# 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Emotion Classes](#-emotion-classes)
- [Model Architectures](#-model-architectures)
- [Training Setup](#-training-setup)
- [Results Summary](#-results-summary)
- [Full Model Comparison](#-full-model-comparison)
- [Visualizations](#-visualizations)
- [Use-Case Recommendations](#-use-case-recommendations)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Project Report](#-project-report)

---

# 🌟 Project Overview
The goal of this project is to classify English tweets into one of six emotion categories using machine learning and deep learning.

The project includes:

✔️ Complete preprocessing pipeline  
✔️ Tokenization & padding  
✔️ LSTM models (baseline & improved)  
✔️ GloVe 100-dimensional embeddings  
✔️ RoBERTa transformer classifier  
✔️ Confusion matrices  
✔️ F1, Precision, Recall benchmarking  
✔️ Model agreement analysis  
✔️ Comprehensive visualizations  

The entire implementation is contained inside:  
📄 **tweet emotion recognition.py**

---

# 📊 Dataset
The project uses the HuggingFace “emotion” dataset:

| Split | Samples |
|---------|----------|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

---

# 🎭 Emotion Classes
The dataset maps emotions using:
{0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise}

---

# 🧠 Model Architectures

## 1️⃣ Original LSTM Model (Baseline)
- Random embedding layer (size 16)  
- Bi-directional LSTM (20 units × 2 layers)  
- Softmax output for 6 emotions  
**Purpose:** Fast baseline for comparison.

---

## 2️⃣ Improved LSTM Model (GloVe)
- GloVe 100-dimensional pretrained embeddings  
- Bi-LSTM with 64 → 32 hidden units  
- Dropout regularization  
- Dense classifier  
**Purpose:** Improved semantic understanding and F1 score.

---

## 3️⃣ RoBERTa Transformer Model
Using model:  
**j-hartmann/emotion-english-distilroberta-base**

- Contextual transformer embeddings  
- SOTA emotion classification  
- Evaluated on full test set using report metrics  
**Purpose:** Highest representational capability and contextual understanding.

---

# ⚙️ Training Setup

| Parameter | Value |
|----------|--------|
| Max sequence length | 50 tokens |
| Vocabulary size | 10,000 words |
| Optimizer | Adam |
| Loss | Sparse Crossentropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau |
| GloVe Embeddings | glove.twitter.27B.100d |

---

# 🏆 Results Summary

## 📌 Original LSTM (Random Embeddings)
- **Accuracy:** 87.75%  
- **Weighted F1 Score:** 0.8780  
- **Precision:** 0.8790  
- **Recall:** 0.8775  

---

## 📌 Improved LSTM (GloVe Embeddings)
- **Accuracy:** 92.70%  
- **Weighted F1 Score:** 0.9264  
- **Precision:** 0.9293  
- **Recall:** 0.9270  
- **Improvement vs Baseline:**  
  - Accuracy: +4.95%  
  - F1 Score: +0.0484 (4.84%)  

---

## 📌 RoBERTa Transformer (SOTA)
- **Accuracy:** 88.89%  
- **Weighted F1 Score:** 0.8672  
- **Precision:** 0.8476  
- **Recall:** 0.8889  

---

# 📈 Full Model Comparison

| Model | Accuracy | Weighted F1 | Precision | Recall |
|--------|-----------|--------------|------------|---------|
| **Original LSTM** | 87.75% | 0.8780 | 0.8790 | 0.8775 |
| **Improved LSTM** | 92.70% | 0.9264 | 0.9293 | 0.9270 |
| **RoBERTa** | 88.89% | 0.8672 | 0.8476 | 0.8889 |

---

# 🎨 Visualizations

The project automatically generates:

- 📌 Accuracy Comparison (all 3 models)  
- 📌 F1 Score Comparison  
- 📌 Confusion Matrices  
- 📌 Model Agreement Pie Chart  
- 📌 Emotion Distribution Heatmap  
- 📌 Confidence Plots  

All visualizations appear directly during script execution.

---

# 🔍 Use-Case Recommendations

### ✔️ Use Original LSTM When:
- Low-latency is required  
- Edge devices  
- Quick inference  

### ✔️ Use Improved LSTM (GloVe) When:
- Balanced speed + accuracy needed  
- Medium-scale apps  
- Highest statistical performance  

### ✔️ Use RoBERTa When:
- Maximum contextual accuracy needed  
- Transformer-level linguistic understanding  
- Server-based inference pipelines  

---

# 📁 Project Structure
Twitter-Emotion-Recognition/
│── tweet emotion recognition.py # Full implementation
└── README.md # Documentation

---

# ▶️ How to Run

### 1. Install dependencies
pip install numpy pandas tensorflow matplotlib seaborn sklearn datasets transformers torch

### 2. Run the Python file
python "tweet emotion recognition.py"

### 3. View Outputs  
The script prints:

- Training logs  
- Validation metrics  
- Test evaluation  
- Confusion matrices  
- Model comparison tables  
- Visualizations  

---

# 📄 Project Report  
Download and view the complete project report here:  
👉 **[Download & View Report](https://docs.google.com/document/d/1uaZi6xD16Hv_5GaH6mTZF2B1FznTtUIq-C40tKcn9Zg/edit?usp=sharing)**

---

# 🎉 Final Note
This project demonstrates a full progression from a classical LSTM model to modern transformer-based deep learning for emotion analysis, including benchmarking, visualizations, embeddings, comparisons, and insights — making it a complete academic and practical submission.

If you liked this project, ⭐ star the repository!
