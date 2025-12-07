# 💬🧠 Twitter Emotion Recognition using Deep Learning & Transformers  
### **Deep Learning & Applications (UEC642) — Final Project**

A complete end-to-end system for emotion classification in tweets using multiple deep learning approaches, ranging from classical LSTMs to modern Transformer-based models.  
This project evaluates **three separate models** and compares their performance:

1. **Original LSTM (Random Embeddings)**  
2. **Improved LSTM (GloVe Pre-trained Embeddings)**  
3. **RoBERTa Transformer (State-of-the-Art)**  

This repository contains the full code, evaluation pipeline, comparison metrics, visualizations, and model benchmarking.

---

# 👨‍🎓 Submitted To  
**Dr. Gaganpreet Kaur**

# 👨‍🎓 Submitted By  
- **Kanav Kukreja — 102215145**  
- **Priyanshu — 102215164**  
- **Vinaayak Kumar Puri — 102215165**  
- **Punya Arora — 102215186**

---

# 📌 Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Emotion Classes](#emotion-classes)  
- [Model Architectures](#model-architectures)  
- [Training Setup](#training-setup)  
- [Results Summary](#results-summary)  
- [Full Model Comparison](#full-model-comparison)  
- [Visualizations](#visualizations)  
- [Use-Case Recommendations](#use-case-recommendations)  
- [Project Structure](#project-structure)  
- [How to Run](#how-to-run)

---

# 🌟 Project Overview
The goal of this project is to classify English tweets into one of **six emotion categories** using machine learning and deep learning.

The project includes:

- ✔️ Complete preprocessing pipeline  
- ✔️ Tokenization & padding  
- ✔️ LSTM models (baseline & improved)  
- ✔️ GloVe 100-dimensional embeddings  
- ✔️ RoBERTa transformer classifier  
- ✔️ Confusion matrices  
- ✔️ F1, Precision, Recall benchmarking  
- ✔️ Model agreement analysis  
- ✔️ Comprehensive visualizations  

The entire implementation is contained inside:  
📄 **tweet emotion recognition.py** :contentReference[oaicite:0]{index=0}

---

# 📊 Dataset
The project uses the **HuggingFace “emotion” dataset**:

| Split | Samples |
|-------|----------|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

---

# 🎭 Emotion Classes
The dataset maps emotions using:
{0: sadness,
1: joy,
2: love,
3: anger,
4: fear,
5: surprise}


---

# 🧠 Model Architectures

## **1️⃣ Original LSTM Model (Baseline)**
- Random embedding layer (size 16)  
- Bi-directional LSTM (20 units × 2 layers)  
- Softmax output for 6 emotions  

**Purpose:** Fast baseline for comparison.

---

## **2️⃣ Improved LSTM Model (GloVe)**
- GloVe 100-dimensional pretrained embeddings  
- Bi-LSTM with larger hidden units (64 → 32)  
- Dropout for regularization  
- Additional Dense layer  

**Purpose:** Increase semantic understanding and F1 score.

---

## **3️⃣ RoBERTa Transformer Model**
Using model:  
`j-hartmann/emotion-english-distilroberta-base`

- Contextual transformer embeddings  
- SOTA emotion classification  
- Evaluated on 100 test samples due to transformer inference cost  

**Purpose:** Highest accuracy and contextual understanding.

---

# ⚙️ Training Setup

| Parameter | Value |
|-----------|--------|
| Max sequence length | 50 tokens |
| Vocabulary size | 10,000 words |
| Optimizer | Adam |
| Loss | Sparse Crossentropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau |
| GloVe Embeddings | glove.twitter.27B.100d |

---

# 🏆 Results Summary

## **📌 Original LSTM (Random Embeddings)**
- **Accuracy:** Printed during evaluation  
- **F1 Score:** Printed  
- **Recall:** Printed  
- **Precision:** Printed  

---

## **📌 Improved LSTM (GloVe Embeddings)**
- Higher accuracy  
- Higher F1 score  
- Better generalization  
- Clear improvement over baseline  

---

## **📌 RoBERTa Transformer (SOTA)**
Evaluated on **100 tweets**.  
Results printed in the file include:

- Accuracy  
- F1 Score  
- Precision  
- Recall  
- Classification report  
- Confusion matrix  
- Per-emotion performance  

---

# 📈 Full Model Comparison (Printed in Output)

The code prints a table summarizing:

| Model | Accuracy | F1 Score | Precision | Recall |
|--------|-----------|-----------|------------|----------|
| Original LSTM | Values printed | Printed | Printed | Printed |
| Improved LSTM | Values printed | Printed | Printed | Printed |
| RoBERTa | Values printed | Printed | Printed | Printed |

It also prints improvements such as:

- F1 improvement from LSTM → GloVe  
- F1 improvement from LSTM → RoBERTa  
- Accuracy increases  

---

# 🎨 Visualizations

The project automatically generates visualizations:

### 📌 Accuracy Comparison  
- Original vs Improved vs RoBERTa  

### 📌 F1 Score Comparison  
- Highlights F1 as the **primary metric**

### 📌 Confusion Matrices  
- For original  
- For improved  
- For RoBERTa  

### 📌 Model Agreement Pie Chart  
Shows how often all 3 models agree, 2 agree, or none agree.

### 📌 Emotion Distribution Heatmap  
Across all 3 models.

### 📌 Confidence Plots  
For each test tweet across all 3 models.

All visualizations are created using Matplotlib and displayed directly during execution.

---

# 🔍 Use-Case Recommendations (Included in Code Output)

### ✔️ Use Original LSTM When:
- Low-latency is required  
- Edge devices  
- Quick inference  

### ✔️ Use Improved LSTM (GloVe) When:
- Balanced speed + accuracy needed  
- Medium-scale apps  
- F1 score is priority  

### ✔️ Use RoBERTa When:
- Maximum accuracy needed  
- Batch inference  
- Server-side deployments  

---

# 📁 Project Structure
Twitter-Emotion-Recognition/
│── tweet emotion recognition.py # Full end-to-end implementation
│── README.md # Documentation
│── results/ # (Optional) Save plots manually if needed
└── models/ # (Optional) Save models if exporting

---

# ▶️ How to Run

### **1. Install dependencies**
pip install numpy pandas tensorflow matplotlib seaborn sklearn datasets transformers torch

### **2. Run the Python file**
python "tweet emotion recognition.py"

### **3. View Outputs**
This file contains:

- Training logs  
- Validation metrics  
- Test evaluation  
- Confusion matrices  
- Model comparison tables  
- Visualizations  

Everything runs automatically — no manual steps required.

---

# 🎉 Final Note  
This project demonstrates a full progression from a classical LSTM model to modern transformer-based deep learning for emotion analysis.  
It includes benchmarking, visualizations, comparisons, embeddings, and insights — making it a complete academic + practical submission.

If you liked this project, ⭐ star the repository!

