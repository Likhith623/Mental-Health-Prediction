import cv2
import numpy as np
import tempfile
import soundfile as sf
import librosa
from typing import Optional
from fastapi import UploadFile


async def analyze_files(image: Optional[UploadFile], audio: Optional[UploadFile]):
    face_res = None
    speech_res = None

    if image is not None:
        image_bytes = await image.read()
        face_res = _predict_face_emotion(image_bytes)

    if audio is not None:
        audio_bytes = await audio.read()
        speech_res = _predict_speech_emotion(audio_bytes)

    combined = _combine_emotions(face_res, speech_res)

    return {"face": face_res, "speech": speech_res, "combined": combined}


# --- Face heuristic ---
def _predict_face_emotion(image_bytes: bytes) -> dict:
    # decode image bytes to numpy
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"label": "unknown", "confidence": 0.0, "notes": "unable to decode image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return {"label": "no_face_detected", "confidence": 0.0}

    # look for smile inside face regions
    smiles_found = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) > 0:
            smiles_found += 1

    if smiles_found >= 1:
        return {"label": "happy", "confidence": min(0.9, 0.5 + 0.4 * smiles_found)}

    # fallback: neutral
    return {"label": "neutral", "confidence": 0.6}


# --- Speech heuristic ---

def _predict_speech_emotion(audio_bytes: bytes) -> dict:
    # write bytes to a temp file and use librosa to analyze
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            y, sr = librosa.load(tmp.name, sr=22050, mono=True)
        except Exception:
            # try soundfile read as fallback
            try:
                y, sr = sf.read(tmp.name)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
            except Exception:
                return {"label": "unknown", "confidence": 0.0, "notes": "unable to decode audio"}

    # simple features
    energy = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y, sr=sr)

    # heuristics thresholds (demo only)
    if energy > 0.05 and tempo > 120:
        label = 'excited'
        conf = 0.75
    elif energy > 0.04:
        label = 'angry'
        conf = 0.7
    elif energy < 0.01:
        label = 'sad'
        conf = 0.7
    else:
        label = 'neutral'
        conf = 0.6

    # adjust using zcr (higher -> more active)
    conf = min(0.95, conf + (zcr * 0.1))

    return {"label": label, "confidence": float(conf), "energy": float(energy), "zcr": float(zcr), "tempo": float(tempo)}


def _combine_emotions(face: Optional[dict], speech: Optional[dict]) -> dict:
    # If both exist and match, return that. If they disagree, return both with simple rule.
    if face is None and speech is None:
        return {"label": "none", "confidence": 0.0}

    if face is None:
        return {"label": speech['label'], "source": 'speech', "confidence": speech.get('confidence', 0.6)}

    if speech is None:
        return {"label": face['label'], "source": 'face', "confidence": face.get('confidence', 0.6)}

    if face['label'] == speech['label']:
        # average confidences
        avg_conf = (face.get('confidence', 0.6) + speech.get('confidence', 0.6)) / 2.0
        return {"label": face['label'], "source": 'both', "confidence": float(avg_conf)}

    # heuristics mapping between face and speech labels to a combined label
    # Example: if face happy and speech excited -> positive
    positive = set(['happy', 'excited'])
    negative = set(['sad', 'angry'])

    if face['label'] in positive and speech['label'] in positive:
        label = 'positive'
    elif face['label'] in negative and speech['label'] in negative:
        label = 'negative'
    else:
        label = 'mixed'

    avg_conf = (face.get('confidence', 0.6) + speech.get('confidence', 0.6)) / 2.0
    return {"label": label, "source": 'combined', "confidence": float(avg_conf), "components": {"face": face, "speech": speech}}
