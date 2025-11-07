import React, { useState } from 'react'

export default function UploadForm({ onResult }) {
  const [imageFile, setImageFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function handleImageChange(e) {
    const f = e.target.files[0]
    setImageFile(f)
    if (f) setPreviewUrl(URL.createObjectURL(f))
  }

  function handleAudioChange(e) {
    const f = e.target.files[0]
    setAudioFile(f)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    onResult(null)

    if (!imageFile && !audioFile) {
      setError('Please select an image or audio file to upload.')
      return
    }

    setLoading(true)
    const fd = new FormData()
    if (imageFile) fd.append('image', imageFile)
    if (audioFile) fd.append('audio', audioFile)

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: fd,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Request failed')
      }

      const data = await res.json()
      onResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <div className="field">
        <label>Face image</label>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        {previewUrl && <img src={previewUrl} alt="preview" className="preview" />}
      </div>

      <div className="field">
        <label>Audio clip (wav/mp3)</label>
        <input type="file" accept="audio/*" onChange={handleAudioChange} />
        {audioFile && <audio controls src={URL.createObjectURL(audioFile)} />}
      </div>

      <div className="actions">
        <button type="submit" disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
      </div>

      {error && <div className="error">{error}</div>}
    </form>
  )
}
