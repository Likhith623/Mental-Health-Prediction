# Mental Health Chatbot - Backend

FastAPI backend for the Mental Health Prediction Chatbot.

## Features

- **Emotion Prediction**: Uses TF-IDF vectorizer and stacking model to predict emotions from text
- **Grok AI Integration**: Generates empathetic chatbot responses using Grok API
- **Fallback Responses**: Provides empathetic responses when Grok API is unavailable
- **CORS Enabled**: Allows requests from React frontend

## Setup

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure API Key** (optional but recommended):
- Copy `.env.example` to `.env`
- Add your Grok API key from https://console.x.ai/

3. **Run the server**:
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /chat
Send a chat message and get emotion prediction + bot response.

**Request Body**:
```json
{
  "message": "I feel so empty and depressed lately.",
  "grok_api_key": "optional_api_key_here"
}
```

**Response**:
```json
{
  "bot_response": "I hear you, and I want you to know...",
  "predicted_emotion": "depression"
}
```

### GET /health
Check if the API and models are loaded correctly.

### GET /docs
Interactive API documentation (Swagger UI).

## Model Files Required

Ensure these files exist in `../MODELS/`:
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `stacking_model.pkl`

## Notes

- The API runs on `http://localhost:8000` by default
- Grok API key can be provided per-request or configured in environment
- If no Grok API key is provided, the system uses empathetic fallback responses
