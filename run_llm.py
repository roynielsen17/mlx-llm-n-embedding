#!/usr/bin/env python3
import argparse
import json
from typing import List, Optional, Generator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_lm import load, generate


# ---------------- CLI args ----------------

parser = argparse.ArgumentParser(description="MLX LLM Streaming Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8899)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

print(f"Loading model: {args.model}")
model, tokenizer = load(args.model)

app = FastAPI()


# ---------------- Schemas ----------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


# ---------------- Helpers ----------------

def build_prompt(messages: List[ChatMessage]) -> str:
    """
    If the user already sends a Qwen-formatted prompt (<|im_start|>system ...),
    pass it through unchanged.

    Otherwise, concatenate system + user messages.
    """
    # Pass-through for preformatted Qwen prompt
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user and "<|im_start|>" in last_user.content:
        return last_user.content

    # Fallback: simple concatenation
    prompt = ""
    for m in messages:
        if m.role in ("system", "user"):
            prompt += m.content + "\n"
    return prompt


def generate_text(req: ChatRequest) -> str:
    """
    Generate text using MLX. No unsupported kwargs.
    """
    prompt = build_prompt(req.messages)

    gen_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": req.max_tokens or args.max_tokens,
    }

    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p

    return generate(**gen_kwargs)


def sse_stream(req: ChatRequest) -> Generator[bytes, None, None]:
    """
    Stream the generated text in OpenAI-compatible SSE format.
    """
    full_text = generate_text(req)

    chunk_size = 50
    idx = 0

    while idx < len(full_text):
        piece = full_text[idx:idx + chunk_size]
        idx += chunk_size

        data = {
            "choices": [
                {
                    "delta": {
                        "content": piece
                    }
                }
            ]
        }

        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

    yield b"data: [DONE]\n\n"


# ---------------- Route ----------------

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    # Merge extra_body (OpenAI client quirk)
    extra = body.get("extra_body", {})
    if isinstance(extra, dict):
        body.update(extra)

    req = ChatRequest(**body)
    stream_flag = body.get("stream", False)

    if stream_flag:
        return StreamingResponse(
            sse_stream(req),
            media_type="text/event-stream",
        )

    # Non-streaming mode
    text = generate_text(req)
    return {
        "id": "mlx-chatcmpl",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ]
    }


# ---------------- Main ----------------

if __name__ == "__main__":
    print(f"Starting streaming MLX LLM server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


