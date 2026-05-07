import json
import os
import threading
import torch
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

MODEL_DIR      = Path("/tmp/smartride-model")
MAX_IN_LEN     = 128
MAX_NEW_TOKENS = 60

tokenizer = None
model     = None
model_ready = False

BASE_MODEL = "google/flan-t5-small"

RELEASE_BASE = "https://github.com/habibfourati/smartride-infer/releases/download/v1.0-model"
MODEL_FILES  = ["model.safetensors", "config.json", "generation_config.json"]

def download_model_files():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MODEL_FILES:
        dest = MODEL_DIR / fname
        if dest.exists():
            continue
        url = f"{RELEASE_BASE}/{fname}"
        print(f"Téléchargement {fname}...")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(f"{fname} OK ({dest.stat().st_size // 1024} KB)")

def load_model():
    global tokenizer, model, model_ready
    print("Chargement du modèle en arrière-plan...")
    download_model_files()
    torch.set_num_threads(4)
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    model.eval()
    model_ready = True
    print("Modèle prêt.")

threading.Thread(target=load_model, daemon=True).start()

class InferRequest(BaseModel):
    ocr: str

def fix_json(s: str) -> str:
    s = s.strip()
    if not s.startswith('{'):
        s = '{' + s
    if not s.endswith('}'):
        s = s + '}'
    return s

@app.get("/health")
def health():
    if not model_ready:
        return {"status": "loading"}
    return {"status": "ok"}

@app.post("/infer")
def infer(req: InferRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")

    ocr = req.ocr.strip()
    if not ocr or len(ocr) < 5:
        raise HTTPException(status_code=400, detail="ocr trop court")
    if len(ocr) > 2000:
        ocr = ocr[:2000]

    prompt  = f"extrait: {ocr}"
    inputs  = tokenizer(prompt, max_length=MAX_IN_LEN, truncation=True, return_tensors="pt")

    with torch.no_grad():
        out_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    raw   = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    fixed = fix_json(raw)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"JSON invalide: {raw}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
