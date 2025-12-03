# Mental Health Chatbot - Full Stack Application

A compassionate AI-powered chatbot that provides mental health support while predicting emotional states from user messages.

## 🎯 Features

- **Real-time Emotion Detection**: ML model predicts emotions (depression, anxiety, stress, anger, happiness, sadness, fear, neutral)
- **AI-Powered Responses**: Integration with Grok API for empathetic, context-aware responses
- **Fallback Support**: Smart fallback responses when API is unavailable
- **Beautiful UI**: Modern, responsive chat interface designed for mental health support
- **Privacy-Focused**: Optional API key configuration, no data persistence

## 🏗️ Architecture

```
├── backend/           # FastAPI server
│   ├── main.py       # Main API endpoints
│   └── requirements.txt
├── frontend/         # React application
│   ├── src/
│   │   ├── App.js   # Main chat component
│   │   └── App.css  # Styling
│   └── package.json
└── MODELS/           # ML models
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    └── stacking_model.pkl
```

## 🚀 Quick Start

### Backend Setup

1. Navigate to backend folder:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The API will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend folder:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## 🔑 Grok API Configuration (Optional)

1. Get your API key from [https://console.x.ai/](https://console.x.ai/)
2. In the chat interface, click the ⚙️ settings icon
3. Enter your API key

**Note**: The app works without an API key using empathetic fallback responses.

## 📊 How It Works

### Emotion Prediction Pipeline

1. **User Input** → Raw text message
2. **TF-IDF Vectorization** → Convert text to numerical features
3. **Stacking Model** → Predict emotion label
4. **Label Decoding** → Convert to emotion name

```python
# Behind the scenes
text → TF-IDF Vector → Stacking Model → Emotion Label → "depression"
```

### Chat Response Flow

1. User sends message
2. Backend predicts emotion from text
3. Backend calls Grok API with emotion context
4. Grok returns empathetic response
5. Frontend displays response + emotion badge

## 🎨 Emotion Categories

| Emotion | Color | Emoji |
|---------|-------|-------|
| Depression | Purple | 😔 |
| Anxiety | Pink | 😰 |
| Stress | Yellow | 😫 |
| Anger | Red | 😠 |
| Happiness | Green | 😊 |
| Sadness | Blue | 😢 |
| Fear | Light Purple | 😨 |
| Neutral | Gray | 😐 |

## 🔧 API Endpoints

### POST /chat
Send a message and receive bot response with emotion prediction.

**Request**:
```json
{
  "message": "I feel overwhelmed with work",
  "grok_api_key": "optional_key"
}
```

**Response**:
```json
{
  "bot_response": "I understand that work can be overwhelming...",
  "predicted_emotion": "stress"
}
```

### GET /health
Check API health and model loading status.

### GET /docs
Interactive API documentation (Swagger UI).

## 🛠️ Tech Stack

### Backend
- FastAPI
- Scikit-learn
- Joblib
- Httpx
- Uvicorn

### Frontend
- React 18
- Axios
- CSS3

### Machine Learning
- TF-IDF Vectorizer
- Stacking Ensemble Model
- Label Encoder

## ⚠️ Important Notes

- This is a **support tool**, not a replacement for professional mental health care
- The chatbot provides empathetic responses but is not a licensed therapist
- For crisis situations, please contact appropriate emergency services

## 📝 License

This project is for educational and support purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure any changes maintain the empathetic and supportive nature of the application.

## 📧 Support

If you need help setting up or using the application, please refer to the individual README files in the `backend/` and `frontend/` folders.
