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


# ---------------- Helpers ----------------

def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Build a prompt using ONLY system + user messages.
    Assistant messages are ignored.
    No role tags are added.
    """
    prompt = ""
    for m in messages:
        if m.role in ("system", "user"):
            prompt += m.content + "\n"
    return prompt

"""
def generate_text(req: ChatRequest) -> str:
    ""
    Generate full text in one pass.
    Do NOT pass temperature/top_p to avoid MLX streaming path.
    ""
    prompt = build_prompt(req.messages)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=req.max_tokens or args.max_tokens,
    )
"""
def generate_text(req: ChatRequest) -> str:
    """
    Generate full text in one pass.
    Chain-of-thought suppressed using bad_words_ids.
    """
    prompt = build_prompt(req.messages)

    # Block Qwen reasoning tokens
    bad_words_ids = [
        [tokenizer.convert_tokens_to_ids("<think>")],
        [tokenizer.convert_tokens_to_ids("</think>")],
        [tokenizer.convert_tokens_to_ids("1.")],   # blocks numbered reasoning
        [tokenizer.convert_tokens_to_ids("2.")],
        [tokenizer.convert_tokens_to_ids("3.")],
    ]

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=req.max_tokens or args.max_tokens,
        bad_words_ids=bad_words_ids,
        stop=["<think>", "</think>", "1.", "2.", "3."],
    )


def sse_stream(req: ChatRequest) -> Generator[bytes, None, None]:
    """
    Stream ONLY the generated text in chunks.
    No assistant role.
    No wrapper.
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

    # ⭐ Merge extra_body into top-level JSON (required for OpenAI client)
    extra = body.get("extra_body", {})
    if isinstance(extra, dict):
        body.update(extra)

    # Parse into ChatRequest
    req = ChatRequest(**body)

    # Read stream flag from merged JSON
    stream_flag = body.get("stream", False)

    if stream_flag:
        return StreamingResponse(
            sse_stream(req),
            media_type="text/event-stream",
        )

    # Non-streaming mode: return ONLY the generated text
    text = generate_text(req)
    return text


# ---------------- Main ----------------

if __name__ == "__main__":
    print(f"Starting streaming MLX LLM server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

