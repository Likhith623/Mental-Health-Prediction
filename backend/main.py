from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from typing import Optional
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Mental Health Chatbot API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
MODEL_PATH = "../MODELS"
tfidf_vectorizer = None
label_encoder = None
stacking_model = None

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.on_event("startup")
async def load_models():
    global tfidf_vectorizer, label_encoder, stacking_model
    try:
        tfidf_vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
        stacking_model = joblib.load(os.path.join(MODEL_PATH, "stacking_model.pkl"))
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e


class ChatMessage(BaseModel):
    message: str
    gemini_api_key: Optional[str] = None


class ChatResponse(BaseModel):
    bot_response: str
    predicted_emotion: str
    confidence: Optional[str] = None


def predict_emotion(text: str) -> str:
    """Predict emotion from text using the stacking model"""
    try:
        # Transform text to TF-IDF vector
        text_vec = tfidf_vectorizer.transform([text])
        
        # Predict encoded label
        pred = stacking_model.predict(text_vec)
        
        # Decode numeric label to emotion name
        emotion = label_encoder.inverse_transform(pred)[0]
        
        return emotion
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "unknown"


async def get_gemini_response(message: str, emotion: str, api_key: Optional[str] = None) -> str:
    """Get chatbot response from Gemini API"""
    
    # Use provided API key or fall back to environment variable
    api_key = api_key or GEMINI_API_KEY
    
    if not api_key:
        # Fallback response if no API key provided
        print("⚠️ No Gemini API key found. Using fallback responses.")
        return generate_fallback_response(emotion)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # System prompt to guide Gemini's empathetic responses
            system_instruction = f"""You are a compassionate mental health support chatbot. 
The user's message shows signs of {emotion}. Respond with empathy, understanding, and support.
Keep responses warm, supportive, and conversational (2-3 sentences max). Avoid being clinical or overly formal.
If the emotion indicates distress, gently encourage seeking professional help when appropriate."""
            
            prompt = f"{system_instruction}\n\nUser message: {message}\n\nRespond with empathy:"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 150,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return generate_fallback_response(emotion)
                
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return generate_fallback_response(emotion)


def generate_fallback_response(emotion: str) -> str:
    """Generate empathetic fallback responses based on detected emotion"""
    responses = {
        "depression": "I hear you, and I want you to know that what you're feeling is valid. It's okay to not be okay sometimes. Have you considered talking to someone you trust about how you've been feeling?",
        "anxiety": "It sounds like you're going through a tough time. Anxiety can be overwhelming, but you're not alone in this. Taking things one step at a time can help. Is there anything specific that's been on your mind?",
        "stress": "I understand that you're feeling stressed. Remember to take care of yourself - even small breaks can make a difference. What's been weighing on you lately?",
        "anger": "I can sense you're feeling frustrated or upset. Those feelings are completely valid. Would you like to talk about what's bothering you?",
        "happiness": "It's wonderful to hear positive energy in your message! I'm here to chat whenever you need. What's been going well for you?",
        "sadness": "I'm sorry you're feeling down. It's important to acknowledge these feelings. Remember, it's okay to reach out for support. I'm here to listen.",
        "fear": "It sounds like you're feeling worried or scared about something. Those feelings are natural. Would you like to share what's concerning you?",
        "neutral": "I'm here to listen and support you. How are you doing today? Feel free to share whatever's on your mind."
    }
    
    return responses.get(emotion.lower(), "Thank you for sharing. I'm here to listen and support you. How can I help you today?")


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint that:
    1. Predicts emotion from user message
    2. Gets chatbot response from Grok API (or fallback)
    """
    try:
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Predict emotion from the message
        predicted_emotion = predict_emotion(message.message)
        
        # Get bot response from Gemini API
        bot_response = await get_gemini_response(
            message.message, 
            predicted_emotion,
            message.gemini_api_key
        )
        
        return ChatResponse(
            bot_response=bot_response,
            predicted_emotion=predicted_emotion
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([
            tfidf_vectorizer is not None,
            label_encoder is not None,
            stacking_model is not None
        ])
    }


@app.get("/")
async def root():
    return {
        "message": "Mental Health Chatbot API",
        "endpoints": {
            "/chat": "POST - Send a chat message",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
