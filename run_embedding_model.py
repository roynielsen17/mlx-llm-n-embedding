#!/usr/bin/env python3

import argparse
from typing import Union, List

from fastapi import FastAPI
from pydantic import BaseModel
from mlx_embeddings.utils import load
import uvicorn


# --------------------------------------------------
# CLI ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser(description="MLX Embedding Server")
parser.add_argument("--model", type=str, required=True,
                    help="Embedding model name (HuggingFace or MLX repo)")
parser.add_argument("--host", type=str, default="127.0.0.1",
                    help="Host to bind the server")
parser.add_argument("--port", type=int, default=8898,
                    help="Port to bind the server")
args = parser.parse_args()

MODEL_NAME = args.model


# --------------------------------------------------
# Load Model
# --------------------------------------------------

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
# Input Normalization
# --------------------------------------------------

def normalize_to_text_list(raw_input) -> List[str]:
    """
    Normalize OpenAI-style embedding input into List[str].

    Supports:
    - "hello"
    - ["hello", "world"]
    - [123, 456]  (tokenized single sequence)
    - [[123, 456], [789]] (batch of tokenized sequences)
    """

    # Single string
    if isinstance(raw_input, str):
        return [raw_input]

    # List input
    if isinstance(raw_input, list) and len(raw_input) > 0:

        # List[str]
        if isinstance(raw_input[0], str):
            return raw_input

        # List[int] → decode single tokenized sequence
        if isinstance(raw_input[0], int):
            return [tokenizer.decode(raw_input)]

        # List[List[int]] → decode batch of tokenized sequences
        if isinstance(raw_input[0], list):
            return [tokenizer.decode(tokens) for tokens in raw_input]

        raise ValueError(f"Unsupported list element type: {type(raw_input[0])}")

    raise ValueError(f"Unsupported input type: {type(raw_input)}")


# --------------------------------------------------
# Embedding Generation
# --------------------------------------------------

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.
    """
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

    try:
        texts = normalize_to_text_list(req.input)
    except Exception as e:
        return {"error": str(e)}

    embeddings = generate_embeddings(texts)

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
    print(f"Starting MLX Embedding server on http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )


