import React, { useState } from 'react'
import UploadForm from './components/UploadForm'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="container">
      <header>
        <h1>AI Emotion Recognition — Demo</h1>
        <p>Upload a face image and/or a short audio clip. This demo uses simple heuristics; replace backend logic with real ML models for production.</p>
      </header>

      <UploadForm onResult={setResult} />

      {result && (
        <section className="results">
          <h2>Prediction</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </section>
      )}

      <footer>
        <small>Backend: FastAPI (http://localhost:8000). For better results, plug in real models.</small>
      </footer>
    </div>
  )
}
