# 🧠 Multimodal Mental Health Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Stacking-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-79%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> An advanced AI-powered mental health classification system using state-of-the-art ensemble machine learning techniques to detect and classify mental health conditions from text data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results & Visualizations](#results--visualizations)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project presents a **Multimodal Mental Health Monitoring System** that analyzes a user's psychological state using three inputs: **text, facial expressions, and voice**. The current implementation focuses on the **text classification component** using advanced machine learning techniques.

When the user begins a session, the system:
- Requests a 6-second camera check-in to capture baseline facial emotions
- Engages in text conversation with continuous sentiment analysis
- Evaluates cognitive distortions, stress cues, suicide risk, and conversation tone
- Optionally processes voice input for speech rate, vocal stress, tone stability, and fatigue analysis

All modalities are fused to generate detailed mental-health insights including depression level, anxiety level, stress level, emotion state, risk score, and personalized recommendations.

---

## 🔴 Problem Statement

### The Challenge

Mental health issues are increasingly prevalent, yet early detection remains difficult. People often express their struggles through text (journals, social media, chats) using ambiguous language. Distinguishing between a user who is experiencing "Stress" versus someone exhibiting "Suicidal" tendencies requires capturing subtle semantic patterns and contextual nuances.

### The Goal

Build a robust Machine Learning system that takes a user's text input and accurately classifies it into one of **7 mental health categories**:

1. **Normal** - Healthy mental state
2. **Anxiety** - Persistent worry and nervousness
3. **Depression** - Prolonged sadness and loss of interest
4. **Suicidal** - Active suicidal ideation (HIGH RISK)
5. **Stress** - Overwhelming pressure and tension
6. **Bipolar** - Extreme mood swings
7. **Personality Disorder** - Persistent patterns of behavior

### The Solution

A **Stacking Ensemble Classifier** that combines the strengths of:
- **Gradient Boosting** (LightGBM, XGBoost) - Pattern recognition
- **Geometry-based models** (SVM) - Linear separation
- **Probability models** (Logistic Regression, Naive Bayes) - Statistical inference

This ensemble achieves maximum accuracy on CPU hardware while maintaining interpretability and reliability.

---

## ✨ Key Features

### 🎯 High Accuracy Classification
- **79% overall accuracy** across 7 mental health categories
- Specialized optimization for high-risk classes (Suicidal detection)
- Robust handling of class imbalance

### 🧮 Advanced NLP Techniques
- **TF-IDF Vectorization** with bigrams (6,000 features)
- Custom text preprocessing pipeline
- N-gram analysis for context preservation

### 🏆 Ensemble Architecture
- **5-model stacking ensemble**:
  - Logistic Regression (Baseline)
  - LinearSVC (Geometry-based)
  - LightGBM (Gradient Boosting)
  - XGBoost (Enhanced Boosting)
  - Complement Naive Bayes (Probabilistic)
- **Genetic Algorithm** for optimal weight tuning
- Meta-learner for final predictions

### ⚖️ Class Imbalance Handling
- Custom class weighting (e.g., 6.9x for Personality Disorder)
- Sample-weight integration for tree-based models
- Stratified train-test splitting

### 📊 Comprehensive Evaluation
- Multi-metric analysis (Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa)
- ROC-AUC curves for all classes
- Precision-Recall curves (critical for imbalanced data)
- Calibration curves for probability reliability
- Confusion matrix analysis

### 🎨 Interactive Deployment
- **Streamlit web application** with gradient UI
- Real-time predictions with confidence scores
- Visual confidence breakdown using Plotly
- Public access via ngrok tunneling

---

## 📊 Dataset

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 53,043 |
| **Features** | 2 (text statement + status label) |
| **Train Set** | 42,434 samples (80%) |
| **Test Set** | 10,609 samples (20%) |
| **Vocabulary Size** | 6,000 unique tokens |
| **Average Text Length** | ~100 characters |

### Class Distribution

| Mental Health Category | Count | Percentage |
|------------------------|-------|------------|
| **Normal** | 16,351 | 30.8% |
| **Depression** | 15,404 | 29.0% |
| **Suicidal** | 10,653 | 20.1% |
| **Anxiety** | 3,888 | 7.3% |
| **Bipolar** | 2,877 | 5.4% |
| **Stress** | 2,669 | 5.0% |
| **Personality Disorder** | 1,201 | 2.3% |

**Challenge**: Significant class imbalance (14:1 ratio between largest and smallest class)

---

## 🏗️ Technical Architecture

### System Pipeline

```
Raw Text Input
    ↓
Text Cleaning & Preprocessing
    ↓
TF-IDF Vectorization (6000 features)
    ↓
┌──────────────────────────────────┐
│     Base Model Layer (Level-1)   │
├──────────────────────────────────┤
│  • Logistic Regression           │
│  • Linear SVM                    │
│  • LightGBM                      │
│  • XGBoost                       │
│  • Complement Naive Bayes        │
└──────────────────────────────────┘
    ↓ (Probability Predictions)
┌──────────────────────────────────┐
│   Meta-Learner (Level-2)         │
│   Genetic Algorithm Optimization  │
└──────────────────────────────────┘
    ↓
Final Weighted Ensemble Prediction
    ↓
Output: Class + Confidence Score
```

### Technology Stack

**Core Libraries:**
- `Python 3.8+`
- `scikit-learn` - ML algorithms and preprocessing
- `pandas` & `numpy` - Data manipulation
- `XGBoost` & `LightGBM` - Gradient boosting
- `NLTK` & `re` - Text preprocessing
- `joblib` - Model serialization

**Visualization:**
- `matplotlib` & `seaborn` - Static plots
- `plotly` - Interactive charts

**Deployment:**
- `streamlit` - Web application framework
- `pyngrok` - Public URL tunneling

---

## 📈 Model Performance

### Overall Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 79.0% | Overall correctness |
| **Top-2 Accuracy** | 92.3% | Correct label in top 2 predictions |
| **Matthews Correlation Coefficient (MCC)** | 0.74 | Strong correlation (robust for imbalance) |
| **Cohen's Kappa** | 0.72 | Substantial agreement beyond chance |
| **Log Loss** | 0.58 | Good probability calibration |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 0.83 | 0.86 | 0.84 | 3,270 |
| **Depression** | 0.77 | 0.79 | 0.78 | 3,081 |
| **Suicidal** | 0.74 | 0.76 | 0.75 | 2,131 |
| **Anxiety** | 0.82 | 0.68 | 0.74 | 778 |
| **Bipolar** | 0.79 | 0.73 | 0.76 | 575 |
| **Stress** | 0.71 | 0.58 | 0.64 | 534 |
| **Personality Disorder** | 0.75 | 0.62 | 0.68 | 240 |

### ROC-AUC Scores

| Class | AUC |
|-------|-----|
| **Normal** | 0.95 |
| **Depression** | 0.92 |
| **Anxiety** | 0.91 |
| **Suicidal** | 0.89 |
| **Bipolar** | 0.93 |
| **Stress** | 0.87 |
| **Personality Disorder** | 0.90 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- CPU sufficient (no GPU required)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Likhith623/Multimodal-Mental-Health-Prediction.git
cd Multimodal-Mental-Health-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
nltk>=3.6
joblib>=1.1.0
streamlit>=1.12.0
pyngrok>=5.1.0
```

---

## 💻 Usage

### 1. Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
df = pd.read_csv('Combined Data.csv')

# Run the complete training pipeline
# (See MENTAL_HEALTH_final.ipynb for full code)

# The trained model will be saved as:
# - FINAL_MODEL.pkl (complete ensemble)
# - tfidf_vectorizer.pkl
# - label_encoder.pkl
```

### 2. Making Predictions

```python
import joblib
import numpy as np

# Load the model
model = joblib.load('FINAL_MODEL.pkl')

# Extract components
tfidf = model['tfidf']
label_encoder = model['label_encoder']
weights = model['weights']

# Prepare text
text = "I feel overwhelmed and anxious all the time"
text_vec = tfidf.transform([text])

# Get predictions from all models
predictions = []
for model_name in ['lgbm', 'xgb', 'svm', 'logreg', 'nb']:
    pred = model[model_name].predict_proba(text_vec)
    predictions.append(pred)

# Weighted ensemble
final_proba = sum(w * p for w, p in zip(weights, predictions))
prediction = label_encoder.inverse_transform([np.argmax(final_proba)])[0]
confidence = np.max(final_proba)

print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
```

### 3. Running the Web Application

```bash
# Start Streamlit app
streamlit run app.py

# For Google Colab (with ngrok)
python deploy_colab.py
# Access via the generated public URL
```

---

## 📁 Project Structure

```
Multimodal-Mental-Health-Prediction/
│
├── MENTAL_HEALTH_final.ipynb    # Main training notebook
├── TEXT_LARGE.ipynb             # Text analysis experiments
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── backend/
│   ├── main.py                  # FastAPI backend
│   ├── README.md
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js               # React main component
│   │   ├── App.css
│   │   └── index.js
│   ├── public/
│   │   └── index.html
│   └── package.json
│
├── docs/
│   ├── training_models.md
│   ├── adaboost/
│   ├── gradientboost/
│   └── xgboost/
│
├── models/                       # Saved trained models
│   ├── FINAL_MODEL.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── logistic_base.pkl
│   ├── svm_tuned.pkl
│   ├── lgbm_model.pkl
│   └── xgboost_model_highpower.pkl
│
└── text_dataset/
    ├── basic.txt
    └── how?.txt
```

---

## 🔬 Methodology

### 1. Data Preprocessing

#### Text Cleaning Pipeline
```python
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

**Steps:**
1. Lowercasing
2. Special character removal
3. Whitespace normalization
4. Duplicate removal

### 2. Feature Engineering

#### TF-IDF Vectorization
- **max_features**: 6,000 (optimal balance)
- **ngram_range**: (1, 2) - Unigrams + Bigrams
- **min_df**: 5 - Minimum document frequency
- **stop_words**: None (handled in preprocessing)

**Why TF-IDF?**
- Highlights meaningful, rare mental-health keywords
- Automatically down-weights common words
- Works excellently with classical ML algorithms
- Fast and interpretable

**Why 6,000 features?**
- Covers 95% of meaningful vocabulary
- Eliminates noise (typos, rare terms)
- Prevents overfitting
- Optimizes training speed

### 3. Class Imbalance Handling

#### Computed Class Weights

| Class | Weight Multiplier |
|-------|-------------------|
| Personality Disorder | 6.90x |
| Stress | 3.10x |
| Bipolar | 2.88x |
| Anxiety | 2.13x |
| Suicidal | 0.78x |
| Depression | 0.54x |
| Normal | 0.51x |

**Why Class Weighting > SMOTE?**

| Method | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Class Weighting** | ✅ No synthetic data<br>✅ Memory efficient<br>✅ Preserves semantic meaning | ⚠️ May increase false positives | ✅ **CHOSEN** |
| **SMOTE** | ✅ Balances distribution | ❌ Creates fake emotional text<br>❌ Memory intensive<br>❌ Risk of overfitting | ❌ Not suitable for NLP |
| **Undersampling** | ✅ Simple | ❌ Data loss | ❌ Wasteful |

### 4. Model Training

#### Base Models (Level-1)

**1. Logistic Regression**
```python
LogisticRegression(
    class_weight=class_weight_dict,
    max_iter=1000,
    solver='saga',
    n_jobs=-1
)
```
- **Role**: Baseline linear classifier
- **Strength**: Fast, interpretable
- **Accuracy**: ~75%

**2. Linear SVM**
```python
LinearSVC(
    class_weight=class_weight_dict,
    dual=False,  # For n_samples > n_features
    max_iter=2000
)
```
- **Role**: Geometric separator
- **Strength**: Excellent for high-dimensional sparse data
- **Accuracy**: ~77%

**3. LightGBM**
```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=40,
    class_weight=class_weight_dict
)
```
- **Role**: Efficient gradient boosting
- **Strength**: Fast training, high accuracy
- **Accuracy**: ~78%

**4. XGBoost**
```python
XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    tree_method='hist'  # CPU-optimized
)
```
- **Role**: Powerful gradient boosting
- **Strength**: State-of-the-art performance
- **Accuracy**: ~78.5%

**5. Complement Naive Bayes**
```python
ComplementNB()
```
- **Role**: Probabilistic classifier
- **Strength**: Designed for imbalanced text
- **Accuracy**: ~74%

#### Meta-Learner (Level-2)

**Genetic Algorithm Optimization**

```python
# Population-based weight search
population_size = 50
generations = 30

# Evolves optimal weights for ensemble
# Final weights (example):
weights = [0.24, 0.22, 0.21, 0.18, 0.15]  # LightGBM, XGBoost, SVM, LogReg, NB
```

**Process:**
1. Initialize random weight populations
2. Evaluate fitness (F1-weighted score)
3. Select top performers
4. Crossover and mutation
5. Repeat for 30 generations
6. Validate on holdout set

---

## 📊 Results & Visualizations

### 1. Confusion Matrix

The normalized confusion matrix reveals:
- **Normal**: 86% correctly identified
- **Depression**: 79% accuracy (some overlap with Suicidal)
- **Suicidal**: 76% recall (critical for safety)
- **Anxiety**: 68% (confusion with Stress)
- **Personality Disorder**: 62% (limited training data)

### 2. ROC Curves

All classes achieve AUC > 0.87, indicating excellent separation:
- **Top performers**: Normal (0.95), Bipolar (0.93), Depression (0.92)
- Strong curves hug the top-left corner
- Minimal false positive rates at high sensitivity

### 3. Precision-Recall Curves

Critical for imbalanced classes:
- **Normal**: AP = 0.97 (excellent)
- **Anxiety**: AP = 0.87
- **Bipolar**: AP = 0.86
- **Depression**: AP = 0.83
- **Suicidal**: AP = 0.76 (acceptable for high-risk detection)
- **Personality Disorder**: AP = 0.70 (limited by sample size)

### 4. Calibration Curve

Model probabilities are well-calibrated:
- Predictions closely follow the diagonal
- Slight under-confidence at 0.7-0.9 range (safer than over-confidence)
- High-confidence predictions (>0.8) are highly reliable

### 5. Confidence Distribution

- Correct predictions cluster at 0.80-0.95 confidence
- Incorrect predictions peak at 0.40-0.60
- Clear separation indicates good calibration
- Very few high-confidence errors

### Error Analysis

**Top Confusions:**
1. **Depression ↔ Suicidal** (semantic overlap with self-harm language)
2. **Anxiety ↔ Stress** (similar symptom descriptions)
3. **Bipolar ↔ Depression** (depressive episodes)

**Optimal Threshold for Suicidal Detection:**
- Standard: 0.50 (default)
- Optimized: **0.37** (maximizes F1-score)
- **Recommendation**: Lower threshold to 0.37 for production to prioritize safety over precision

---

## 🌐 Deployment

### Streamlit Web Application

**Features:**
- Gradient animated title (Red/Pink/Orange theme)
- Real-time text input processing
- Confidence breakdown visualization
- Responsive design
- Public URL via ngrok

**Demo Interface:**
```
╔════════════════════════════════════════╗
║     Mental Health AI 🧠                ║
║  Genetically Optimized Ensemble        ║
╠════════════════════════════════════════╣
║                                        ║
║  [Text Input Area]                     ║
║                                        ║
║        [✨ Analyze Emotion ✨]         ║
║                                        ║
║  ┌──────────────────────────────────┐ ║
║  │ Diagnosis: Anxiety               │ ║
║  │ Confidence: 84.2%                │ ║
║  └──────────────────────────────────┘ ║
║                                        ║
║  📊 Confidence Breakdown:              ║
║  ████████ Anxiety        84%          ║
║  ███ Stress              8%           ║
║  ██ Depression           5%           ║
║  ░ Others                3%           ║
╚════════════════════════════════════════╝
```

### Running Locally

```bash
streamlit run app.py --server.port 8501
```

### Deployment on Google Colab

```python
from pyngrok import ngrok

# Set auth token
ngrok.set_auth_token("YOUR_TOKEN")

# Create tunnel
public_url = ngrok.connect(8501)
print(f"🚀 App live at: {public_url}")

# Run Streamlit
!streamlit run app.py &
```

---

## 🔮 Future Enhancements

### Short-term (3-6 months)
- [ ] Integrate facial emotion recognition (CNN-based)
- [ ] Add voice sentiment analysis (audio processing)
- [ ] Implement user authentication & history tracking
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add multilingual support

### Medium-term (6-12 months)
- [ ] Fine-tune BERT/RoBERTa for improved accuracy
- [ ] Real-time conversation context tracking
- [ ] Integration with mental health resources API
- [ ] Mobile application (React Native)
- [ ] Therapist dashboard for monitoring

### Long-term (1-2 years)
- [ ] Federated learning for privacy-preserving training
- [ ] Explainable AI (LIME/SHAP) for transparency
- [ ] Clinical validation studies
- [ ] Integration with wearable devices
- [ ] Predictive analytics for early intervention

---

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Model improvements
- New visualization techniques
- Documentation enhancements
- Bug fixes
- Performance optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Combined mental health text dataset (53,000+ samples)
- **Libraries**: scikit-learn, XGBoost, LightGBM, Streamlit
- **Inspiration**: The critical need for accessible mental health monitoring tools
- **Community**: Open-source ML community for tools and techniques

---

## 📧 Contact

**Project Maintainer**: Likhith
- GitHub: [@Likhith623](https://github.com/Likhith623)
- Repository: [Multimodal-Mental-Health-Prediction](https://github.com/Likhith623/Multimodal-Mental-Health-Prediction)

---

## ⚠️ Disclaimer

**IMPORTANT**: This system is designed for **research and educational purposes only**. It should **NOT** be used as a substitute for professional mental health diagnosis or treatment. 

- Always consult qualified mental health professionals for diagnosis and treatment
- If you or someone you know is experiencing a mental health crisis, contact:
  - **National Suicide Prevention Lifeline**: 1-800-273-8255 (US)
  - **Crisis Text Line**: Text HOME to 741741 (US)
  - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

---

## 📚 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Documentation: https://xgboost.readthedocs.io/
3. LightGBM Documentation: https://lightgbm.readthedocs.io/
4. Ensemble Methods in Machine Learning: T. Dietterich (2000)
5. TF-IDF and Text Classification: Manning et al., Introduction to Information Retrieval

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for mental health awareness

</div>
