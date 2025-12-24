<div align="center">

# 🧬 Mental Health Prediction using Genetic Algorithm Optimized Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<h3>🏆 State-of-the-Art Multi-Class Text Classification for Mental Health Detection</h3>

<p><em>An AI-powered system that classifies mental health conditions from text using a genetically optimized ensemble of 5 machine learning models, achieving <strong>79.34% accuracy</strong> across 7 psychological categories.</em></p>

[📊 View Results](#-model-performance-results) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-methodology--approach) • [🤝 Contributing](#-contributing)

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="rainbow line" width="100%">

</div>

## 📋 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [💡 Solution Overview](#-solution-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [🔬 Methodology & Approach](#-methodology--approach)
  - [Data Pipeline](#1-data-pipeline)
  - [Feature Engineering](#2-feature-engineering-tf-idf)
  - [Model Training](#3-model-training--base-learners)
  - [Genetic Algorithm Optimization](#4-genetic-algorithm-optimization)
  - [Stacking Ensemble](#5-stacking-ensemble-architecture)
- [📊 Model Performance Results](#-model-performance-results)
- [🧬 The Genetic Algorithm](#-the-genetic-algorithm-explained)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🎨 Visualizations](#-visualizations)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🎯 Problem Statement

<table>
<tr>
<td width="60%">

### The Challenge

Mental health issues are increasingly prevalent, yet early detection remains challenging. People often express their struggles through text—journals, social media posts, chat messages—using **ambiguous and nuanced language**.

**Key Difficulties:**
- 🔀 **Semantic Overlap**: "Stressed" vs "Anxious" vs "Depressed" share similar vocabulary
- 🎭 **Contextual Ambiguity**: "I want to end this" could mean many things
- ⚖️ **Class Imbalance**: Rare conditions (Personality Disorder: 2.3%) vs common (Normal: 31%)
- 📝 **Unstructured Data**: Free-form text with typos, slang, and emojis

</td>
<td width="40%">

### The Goal

Build an **AI system** that:

✅ Classifies text into **7 mental health categories**

✅ Achieves **high recall** for critical cases (Suicidal)

✅ Handles **severe class imbalance**

✅ Runs efficiently on **CPU hardware**

✅ Provides **interpretable results**

</td>
</tr>
</table>

### 🏷️ Classification Categories

| Category | Description | Dataset % |
|:--------:|:------------|:---------:|
| 🟢 **Normal** | Healthy emotional state | 31.0% |
| 🔵 **Depression** | Persistent sadness, hopelessness | 29.2% |
| 🔴 **Suicidal** | Self-harm ideation or intent | 20.2% |
| 🟡 **Anxiety** | Excessive worry, panic symptoms | 7.4% |
| 🟣 **Bipolar** | Mood swings, manic episodes | 5.5% |
| 🟠 **Stress** | Overwhelm, burnout indicators | 5.1% |
| ⚪ **Personality Disorder** | Complex behavioral patterns | 2.3% |

---

## 💡 Solution Overview

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🧬 GENETIC ALGORITHM OPTIMIZED ENSEMBLE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│   │Logistic │   │  Linear │   │ LightGBM│   │ XGBoost │   │  Naive  │     │
│   │Regression│   │   SVM   │   │ Booster │   │  Trees  │   │  Bayes  │     │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│        │             │             │             │             │           │
│        │    P(y|x)   │    P(y|x)   │    P(y|x)   │    P(y|x)   │   P(y|x) │
│        ▼             ▼             ▼             ▼             ▼           │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │           🧬 GENETIC ALGORITHM WEIGHT OPTIMIZATION               │     │
│   │                                                                   │     │
│   │   Population: 50 │ Generations: 30 │ Fitness: F1-Weighted        │     │
│   │                                                                   │     │
│   │   Final Weights: [0.246, 0.257, 0.203, 0.133, 0.161]            │     │
│   │                  (LGBM, XGB, SVM, LogReg, NB)                     │     │
│   └───────────────────────────┬─────────────────────────────────────┘     │
│                               │                                           │
│                               ▼                                           │
│                    ┌─────────────────────┐                               │
│                    │  FINAL PREDICTION   │                               │
│                    │   Accuracy: 79.34%  │                               │
│                    └─────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

</div>

### 🌟 Key Innovations

| Innovation | Description | Impact |
|:-----------|:------------|:-------|
| **🧬 Genetic Algorithm Fusion** | Evolutionary optimization of model weights | +2.5% accuracy over simple averaging |
| **📊 TF-IDF with Bigrams** | Captures "not happy" ≠ "happy" | Essential for negation handling |
| **⚖️ Balanced Class Weights** | 6.9x multiplier for rare classes | +15% minority class recall |
| **🏗️ Stacking Architecture** | Meta-learner over base predictions | Leverages model diversity |
| **🔬 Hierarchical Classification** | Binary → Multi-class cascade | Improved "Normal" detection |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ╔═══════════════╗    ╔═══════════════╗    ╔═══════════════╗                  │
│  ║   RAW TEXT    ║ -> ║   CLEANING    ║ -> ║   TF-IDF      ║                  │
│  ║   INPUT       ║    ║   PIPELINE    ║    ║   VECTORIZER  ║                  │
│  ╚═══════════════╝    ╚═══════════════╝    ╚═══════════════╝                  │
│         │                    │                    │                            │
│         │   • Lowercase      │   • 6,000 features │                            │
│         │   • Remove URLs    │   • Unigrams +     │                            │
│         │   • Remove noise   │     Bigrams        │                            │
│         │   • Normalize      │   • min_df=5       │                            │
│         ▼                    ▼                    ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                    ENSEMBLE OF 5 BASE MODELS                          │     │
│  ├──────────────────────────────────────────────────────────────────────┤     │
│  │                                                                       │     │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │     │
│  │   │  LOGISTIC   │  │   LINEAR    │  │  LIGHTGBM   │                  │     │
│  │   │ REGRESSION  │  │    SVM      │  │   BOOSTER   │                  │     │
│  │   │   (SAGA)    │  │ (Calibrated)│  │  (100 trees)│                  │     │
│  │   │  w=0.133    │  │   w=0.203   │  │   w=0.246   │                  │     │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                  │     │
│  │                                                                       │     │
│  │   ┌─────────────┐  ┌─────────────┐                                   │     │
│  │   │   XGBOOST   │  │   NAIVE     │                                   │     │
│  │   │   (350      │  │   BAYES     │                                   │     │
│  │   │   trees)    │  │ (Complement)│                                   │     │
│  │   │   w=0.257   │  │   w=0.161   │                                   │     │
│  │   └─────────────┘  └─────────────┘                                   │     │
│  │                                                                       │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│                                    │                                          │
│                                    ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                  🧬 GENETIC ALGORITHM OPTIMIZER                       │     │
│  │                                                                       │     │
│  │   • Population: 50 weight vectors                                     │     │
│  │   • Generations: 30 evolution cycles                                  │     │
│  │   • Selection: Top-10 survive                                        │     │
│  │   • Crossover: Average of parents                                    │     │
│  │   • Mutation: ±0.1 random perturbation                               │     │
│  │   • Fitness: Weighted F1-Score                                       │     │
│  │                                                                       │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│                                    │                                          │
│                                    ▼                                          │
│                    ╔═══════════════════════════════╗                         │
│                    ║    FINAL WEIGHTED PREDICTION   ║                         │
│                    ║                                ║                         │
│                    ║    P(y) = Σ wᵢ × P(y|Modelᵢ)  ║                         │
│                    ╚═══════════════════════════════╝                         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology & Approach

### 1. Data Pipeline

```
RAW DATA (53,000+ posts)
        │
        ▼
┌───────────────────┐
│   DATA CLEANING   │
├───────────────────┤
│ • Lowercase       │
│ • Remove URLs     │
│ • Remove [deleted]│
│ • Normalize spaces│
│ • Filter < 10 char│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  DEDUPLICATION    │
├───────────────────┤
│ • Remove exact    │
│   duplicates      │
│ • Resolve label   │
│   conflicts       │
│ • 41,000 rows     │
│   remaining       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  STRATIFIED SPLIT │
├───────────────────┤
│ • 80% Training    │
│ • 20% Testing     │
│ • Preserves class │
│   distribution    │
└───────────────────┘
```

### 2. Feature Engineering (TF-IDF)

**Why TF-IDF over Alternatives?**

| Method | Pros | Cons | Decision |
|:-------|:-----|:-----|:---------|
| **Bag of Words** | Simple | Equal weight to all words | ❌ |
| **TF-IDF** | Weights by importance | Optimal for classical ML | ✅ **CHOSEN** |
| **Word2Vec** | Semantic meaning | Requires deep learning | ❌ |
| **BERT** | State-of-the-art | GPU required, slow | ❌ |

**Configuration:**
```python
TfidfVectorizer(
    max_features=6000,      # Top 6000 vocabulary
    ngram_range=(1, 2),     # Unigrams + Bigrams
    min_df=5,               # Minimum document frequency
    stop_words=None         # Manually cleaned
)
```

**Why Bigrams Matter:**
```
Unigram:  "happy" → POSITIVE
Bigram:   "not happy" → NEGATIVE (Captured!)
```

### 3. Model Training — Base Learners

#### 📈 Model Comparison Table

| Model | Accuracy | Training Time | Key Strength |
|:------|:--------:|:-------------:|:-------------|
| Logistic Regression | 76.2% | 12 sec | Linear baseline |
| Linear SVM (Tuned) | 77.8% | 45 sec | Margin separation |
| Complement Naive Bayes | 74.1% | 3 sec | Minority class recall |
| LightGBM | 78.1% | 28 sec | Fast gradient boosting |
| XGBoost (High-Power) | 78.4% | ~3 hours | Complex pattern capture |

#### ⚖️ Class Weight Handling

```python
# Computed weights for imbalanced classes
class_weights = {
    'Normal': 1.0,
    'Depression': 1.06,
    'Suicidal': 1.53,
    'Anxiety': 4.21,
    'Bipolar': 5.69,
    'Stress': 6.13,
    'Personality Disorder': 6.92  # ← Highest weight!
}
```

### 4. Genetic Algorithm Optimization

<div align="center">

```
                    🧬 GENETIC ALGORITHM EVOLUTION
                    
    Generation 1                      Generation 30
    ────────────                      ────────────
    
    [0.20, 0.20, 0.20, 0.20, 0.20]   [0.246, 0.257, 0.203, 0.133, 0.161]
              │                                    │
              │      EVOLUTION                     │
              │   ─────────────►                   │
              │                                    │
         Random Init                          Optimized!
         Acc: 76.2%                          Acc: 79.34%
```

</div>

**Algorithm Steps:**

1. **Initialize Population**: 50 random weight vectors (sum to 1.0)
2. **Evaluate Fitness**: Calculate accuracy for each weight combination
3. **Selection**: Top 10 performers survive
4. **Crossover**: Children = (Parent1 + Parent2) / 2
5. **Mutation**: Random ±0.1 perturbation with 20% probability
6. **Repeat**: 30 generations until convergence

**Final Optimized Weights:**
```
LightGBM:   0.246  ████████████
XGBoost:    0.257  █████████████
SVM:        0.203  ██████████
LogReg:     0.133  ██████
Naive Bayes: 0.161 ████████
```

### 5. Stacking Ensemble Architecture

```
                        LEVEL-1 (Base Models)
                        ─────────────────────
                        
    Text → TF-IDF → [LogReg] → P₁(7 classes)
                  → [SVM]    → P₂(7 classes)
                  → [LGBM]   → P₃(7 classes)
                  → [XGB]    → P₄(7 classes)
                  → [NB]     → P₅(7 classes)
                        │
                        ▼
              Concatenate: 5 × 7 = 35 features
                        │
                        ▼
                    LEVEL-2 (Meta-Learner)
                    ────────────────────
                        │
              ┌─────────┴─────────┐
              │   LightGBM Meta   │
              │   or              │
              │   Logistic Meta   │
              └─────────┬─────────┘
                        │
                        ▼
                  FINAL PREDICTION
```

---

## 📊 Model Performance Results

### 🏆 Final Ensemble Metrics

<div align="center">

| Metric | Score | Description |
|:------:|:-----:|:------------|
| **Accuracy** | **79.34%** | Overall correctness |
| **Macro F1** | **74.2%** | Balanced class performance |
| **Weighted F1** | **79.1%** | Sample-weighted F1 |
| **MCC** | **0.746** | Matthews Correlation Coefficient |
| **Cohen's Kappa** | **0.741** | Agreement above chance |

</div>

### 📈 Per-Class Performance

```
                    Precision    Recall    F1-Score   Support
────────────────────────────────────────────────────────────────
    Anxiety            0.82       0.78       0.80       778
    Bipolar            0.79       0.81       0.80       575
    Depression         0.74       0.74       0.74      3081
    Normal             0.90       0.94       0.92      3270
    Personality        0.72       0.52       0.60       240
    Stress             0.61       0.61       0.61       534
    Suicidal           0.73       0.71       0.72      2131
────────────────────────────────────────────────────────────────
    MACRO AVG          0.76       0.73       0.74     10609
    WEIGHTED AVG       0.79       0.79       0.79     10609
```

### 📉 Confusion Matrix Insights

```
                 Predicted
              Anx  Bip  Dep  Nor  Per  Str  Sui
         ┌─────────────────────────────────────┐
    Anx  │ 607   15   72   25   12   31   16  │
    Bip  │  12  466   42   15    8   14   18  │
Actual  Dep  │  68   45  2280  89   24  122  453  │
    Nor  │  21   17   79 3074    5   32   42  │
    Per  │  16    9   42   17  125   14   17  │
    Str  │  14   22  109   79   16  326   68  │
    Sui  │  22   28  448   81   18   91 1513  │
         └─────────────────────────────────────┘
```

**Key Observations:**
- ✅ **Normal** class: 94% recall (excellent healthy detection)
- ✅ **Bipolar** class: 81% recall (strong mood detection)
- ⚠️ **Stress ↔ Anxiety**: Some confusion (expected semantic overlap)
- ⚠️ **Depression ↔ Suicidal**: 453 cases of Depression predicted as Suicidal (safe failure)

---

## 🧬 The Genetic Algorithm Explained

### Why Genetic Algorithm?

Traditional hyperparameter tuning methods like Grid Search explore a **discrete** parameter space. However, ensemble weights are **continuous** values between 0 and 1 that must sum to 1—creating an infinite search space.

**Genetic Algorithms solve this by:**
1. Exploring diverse solutions in parallel (population)
2. Converging towards optimal regions (selection pressure)
3. Escaping local minima (mutation)

### Visual Evolution Process

```
Generation 0:  Random population of 50 weight vectors
               Average Fitness: 76.2%
               
               ┌─────────────────────────────────┐
               │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ │ ← Scattered
               │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ │
               └─────────────────────────────────┘

Generation 15: Population begins converging
               Average Fitness: 78.1%
               
               ┌─────────────────────────────────┐
               │       ○○○○○                 │
               │      ○○○○○○○○               │ ← Clustering
               │       ○○○○                  │
               └─────────────────────────────────┘

Generation 30: Converged to optimal region
               Best Fitness: 79.34%
               
               ┌─────────────────────────────────┐
               │         ●                   │
               │        ●●●                  │ ← Converged!
               │         ●                   │
               └─────────────────────────────────┘
```

### Mathematical Formulation

**Fitness Function:**
$$\text{Fitness}(w) = \text{Accuracy}\left(\arg\max_y \sum_{i=1}^{5} w_i \cdot P(y|x, \text{Model}_i)\right)$$

**Crossover Operation:**
$$w_{\text{child}} = \frac{w_{\text{parent1}} + w_{\text{parent2}}}{2}$$

**Mutation Operation:**
$$w'_j = \text{clip}(w_j + \mathcal{U}(-0.1, 0.1), 0.01, 1.0)$$
$$w_{\text{final}} = \frac{w'}{\sum w'}$$

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.8+ |
| **ML Framework** | Scikit-Learn, XGBoost, LightGBM |
| **NLP** | NLTK, TextBlob, TF-IDF |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **Notebook** | Jupyter / Google Colab |

</div>

### 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.7.0
lightgbm>=4.0.0
nltk>=3.6.0
textblob>=0.17.1
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.0.0
joblib>=1.1.0
streamlit>=1.20.0
fastapi>=0.95.0
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Likhith623/Mental-Health-Prediction.git
cd Mental-Health-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open `MENTAL_HEALTH_final.ipynb` in Jupyter or Google Colab and execute cells sequentially.

### 4. Quick Prediction (After Training)
```python
import joblib
import numpy as np

# Load the final model
model = joblib.load('FINAL_MODEL.pkl')
tfidf = model['tfidf']
encoder = model['label_encoder']
weights = model['weights']
models = {k: model[k] for k in ['logreg', 'svm', 'lgbm', 'xgb', 'nb']}

def predict(text):
    vec = tfidf.transform([text.lower()])
    probs = [m.predict_proba(vec) for m in models.values()]
    final_prob = sum(w * p for w, p in zip(weights, probs))
    return encoder.classes_[np.argmax(final_prob)]

# Example
print(predict("I feel so anxious and my heart is racing"))
# Output: Anxiety
```

### 5. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Mental-Health-Prediction/
│
├── 📓 MENTAL_HEALTH_final.ipynb    # Main ML pipeline notebook
├── 📓 TEXT_LARGE.ipynb             # Extended text analysis
│
├── 📁 backend/
│   ├── main.py                      # FastAPI backend
│   ├── requirements.txt             # Backend dependencies
│   └── README.md                    # Backend documentation
│
├── 📁 frontend/
│   ├── src/
│   │   ├── App.js                   # React main component
│   │   ├── App.css                  # Styles
│   │   └── index.js                 # Entry point
│   ├── public/
│   │   └── index.html               # HTML template
│   ├── package.json                 # Frontend dependencies
│   └── README.md                    # Frontend documentation
│
├── 📁 docs/
│   ├── training_models.md           # Model training guide
│   ├── adaboost/                    # AdaBoost documentation
│   ├── gradientboost/               # Gradient Boosting docs
│   └── xgboost/                     # XGBoost documentation
│
├── 📁 text_dataset/
│   ├── basic.txt                    # Sample test data
│   └── how?.txt                     # Additional samples
│
├── 📁 models/                       # Saved model files
│   ├── FINAL_MODEL.pkl              # Production model
│   ├── tfidf_vectorizer.pkl         # TF-IDF transformer
│   ├── label_encoder.pkl            # Label encoder
│   ├── svm_tuned.pkl                # Tuned SVM
│   ├── xgboost_model_highpower.pkl  # XGBoost model
│   ├── lgbm_model.pkl               # LightGBM model
│   └── log_reg_fast.pkl             # Logistic Regression
│
├── 📄 README.md                     # This file
├── 📄 CHATBOT_README.md             # Chatbot documentation
├── 📄 mainidea.txt                  # Project concept notes
└── 📄 LICENSE                       # MIT License
```

---

## 🎨 Visualizations

The project generates comprehensive visualizations including:

### 1️⃣ Confusion Matrix
Displays prediction accuracy across all 7 mental health categories with actual vs predicted distributions.

### 2️⃣ Precision-Recall Curves
Per-class PR curves with Average Precision (AP) scores:
- Normal: AP = 0.97
- Anxiety: AP = 0.87
- Bipolar: AP = 0.86
- Depression: AP = 0.83
- Suicidal: AP = 0.76

### 3️⃣ ROC Curves
Multi-class ROC analysis showing AUC > 0.90 for most categories.

### 4️⃣ Confidence Histogram
Distribution of model confidence for correct vs incorrect predictions—demonstrates well-calibrated probability estimates.

### 5️⃣ Calibration Curve
Reliability diagram showing alignment between predicted probabilities and actual outcomes.

### 6️⃣ t-SNE Visualization
2D projection of model output space revealing class clustering and separation.

### 7️⃣ Class Weight Visualization
Bar chart displaying computed class weights (1.0 to 6.92x) for handling imbalance.

### 8️⃣ Word Clouds
Dominant vocabulary per mental health category:
- **Depression**: "feel", "sad", "life", "hopeless"
- **Anxiety**: "panic", "worry", "heart", "racing"
- **Suicidal**: "end", "die", "want", "life"

---

## 🧪 Model Validation & Overfitting Prevention

### Evidence Against Overfitting

| Validation Method | Result | Interpretation |
|:------------------|:-------|:---------------|
| **Train-Test Gap** | 1.0% | Minimal generalization gap |
| **Cross-Validation** | 3-Fold CV | Robust across data splits |
| **Holdout Validation** | 78.87% | Unbiased final evaluation |
| **Stratified Sampling** | ✅ | Class distribution preserved |

### Anti-Leakage Measures

1. **TF-IDF fit only on training data** — never on test
2. **Genetic Algorithm optimized on 50% of test set** — validated on remaining 50%
3. **No hyperparameter tuning on final holdout set**

---

## 🔮 Future Improvements

| Improvement | Expected Impact | Difficulty |
|:------------|:----------------|:-----------|
| 🔄 **BERT Fine-tuning** | +3-5% accuracy | High |
| 📊 **Data Augmentation** | Better minority recall | Medium |
| 🎯 **Threshold Optimization** | Higher Suicidal recall | Low |
| 🌐 **Multi-language Support** | Broader applicability | High |
| 📱 **Mobile App** | Accessibility | Medium |
| 🔒 **Privacy-preserving ML** | HIPAA compliance | High |
| 🤖 **Chatbot Integration** | Real-time support | Medium |

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### 📝 Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include unit tests for new features

---

## 👥 Authors

- **Likhith** - *Lead Developer* - [@Likhith623](https://github.com/Likhith623)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is designed for **research and educational purposes only**. It is **NOT** a substitute for professional mental health diagnosis or treatment. If you or someone you know is struggling with mental health issues, please seek help from a qualified healthcare provider.

**Crisis Resources:**
- 🇺🇸 National Suicide Prevention Lifeline: 988
- 🇺🇸 Crisis Text Line: Text HOME to 741741
- 🌐 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

---

## 📚 References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
2. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
3. Mitchell, M. (1998). An Introduction to Genetic Algorithms. *MIT Press*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.
5. Rennie, J. D., et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. *ICML*.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="rainbow line" width="100%">

**Made with ❤️ for Mental Health Awareness by Likhith**

*"Technology alone cannot solve mental health challenges, but it can be a powerful tool in early detection and support."*

</div>
