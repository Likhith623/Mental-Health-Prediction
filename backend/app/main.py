from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .predict import analyze_files
import uvicorn

app = FastAPI(title="AI Emotion Recognition (demo)", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
    if image is None and audio is None:
        raise HTTPException(status_code=400, detail="At least one of image or audio must be provided")

    try:
        result = await analyze_files(image, audio)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
