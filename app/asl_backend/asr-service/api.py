import os, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from asr import transcribe

app = FastAPI(title="ASR Service")

ALLOWED = {".mp3",".wav",".m4a",".flac",".ogg",".webm"}

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: str = "base"
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported format: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path, model_size=model_size)
        if result.startswith("[ERROR]"):
            raise HTTPException(500, result)
        return JSONResponse({"transcript": result})
    finally:
        os.unlink(tmp_path)  # always clean up temp file

@app.get("/health")
def health(): return {"status": "ok"}