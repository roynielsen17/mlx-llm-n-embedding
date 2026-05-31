#!/usr/bin/env python3

from typing import Union, List

from fastapi import FastAPI
from pydantic import BaseModel
from mlx_embeddings.utils import load
import uvicorn

# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

print(f"Loading embedding model: {MODEL_NAME}")
model, tokenizer = load(MODEL_NAME)
print("Embedding model loaded")

app = FastAPI()


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

EmbeddingInput = Union[
    str,
    List[str],
    List[int],
    List[List[int]]
]

class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: EmbeddingInput


# --------------------------------------------------
# Helper: Generate Embeddings
# --------------------------------------------------

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="mlx"
    )

    outputs = model(**inputs)
    return outputs.text_embeds.tolist()


# --------------------------------------------------
# Embeddings Endpoint (OpenAI Compatible)
# --------------------------------------------------

@app.post("/v1/embeddings")
def create_embedding(req: EmbeddingRequest):

    raw_input = req.input
    print("\n=== /v1/embeddings CALLED ===")
    print("RAW INPUT TYPE:", type(raw_input))
    print("RAW INPUT VALUE:", repr(raw_input))

    # -------------------------------
    # Normalize input into List[str]
    # -------------------------------

    if isinstance(raw_input, str):
        texts = [raw_input]

    elif isinstance(raw_input, list) and len(raw_input) > 0:

        if isinstance(raw_input[0], str):
            texts = raw_input

        elif isinstance(raw_input[0], int):
            # Single tokenized document
            texts = [tokenizer.decode(raw_input)]

        elif isinstance(raw_input[0], list):
            # Batch of tokenized documents
            texts = [tokenizer.decode(tokens) for tokens in raw_input]

        else:
            return {"error": f"Unsupported list element type: {type(raw_input[0])}"}

    else:
        return {"error": f"Unsupported input type: {type(raw_input)}"}

    print("NORMALIZED TEXT COUNT:", len(texts))

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    print("GENERATED EMBEDDINGS:", len(embeddings))

    # Build OpenAI-style response
    data = []
    total_tokens = 0

    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        try:
            token_count = len(tokenizer.encode(text))
        except Exception:
            token_count = 0

        total_tokens += token_count

        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": embedding
        })

    return {
        "object": "list",
        "data": data,
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


# --------------------------------------------------
# Models Endpoint
# --------------------------------------------------

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "mlx"
            }
        ]
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8898
    )

