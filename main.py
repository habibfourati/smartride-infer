import json
import os
import threading
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

MODEL_DIR   = Path(__file__).parent / "model"
MAX_IN_LEN  = 256
MAX_OUT_LEN = 128

tokenizer = None
model     = None
model_ready = False

BASE_MODEL = "google/flan-t5-small"

def load_model():
    global tokenizer, model, model_ready
    print("Chargement du modèle en arrière-plan...")
    # Tokenizer depuis HuggingFace (évite le problème LFS avec spiece.model)
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    # Poids fine-tunés depuis le dossier local
    model     = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    model.eval()
    model_ready = True
    print("Modèle prêt.")

# Chargement en arrière-plan au démarrage — le serveur répond immédiatement
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
            max_length=MAX_OUT_LEN,
            num_beams=4,
            early_stopping=True
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
