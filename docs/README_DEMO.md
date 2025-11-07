# Demo README — AI Emotion Recognition (Speech + Facial Expressions)

This file contains full instructions to run the demo backend and frontend (FastAPI + Vite/React).

## Run locally (macOS)

### Backend

1. Create a Python virtual environment and activate it (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Run the backend (from repo root):

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000`.

### Frontend

1. Change to the frontend folder and install dependencies (requires Node.js >= 16):

```bash
cd frontend
npm install
npm run dev
```

2. Open `http://localhost:5173` in your browser.

### Example curl

You can test the API directly with curl (replace files):

```bash
curl -X POST "http://localhost:8000/predict" -F "image=@face.jpg" -F "audio=@voice.wav"
```

## Notes

- The backend uses simple heuristics as a demo. Replace `backend/app/predict.py` with real ML model code for production.
- If you want a single-command local run, I can add Docker and docker-compose files next.
