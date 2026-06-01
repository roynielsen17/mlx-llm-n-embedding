#!/usr/bin/env python3
import argparse
import json
import re
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
    content: str  # we normalize to str before validation


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None


# ---------------- Helpers ----------------

def normalize_messages_in_body(body: dict) -> None:
    """
    In-place normalization of body["messages"]:
    - If content is a list, join into a single string.
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return

    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            # Join list elements into a single string
            m["content"] = " ".join(str(x) for x in content)


def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert OpenAI-style messages into a single Qwen-style prompt.

    - Use the last system message (if any) as system.
    - Use the last user message as user.
    - Ignore assistant messages.
    """
    system = ""
    user = ""

    for m in messages:
        content = m.content
        if m.role == "system":
            system = content
        elif m.role == "user":
            user = content

    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_text(req: ChatRequest) -> str:
    """
    Generate full text in one pass.
    Do NOT pass temperature/top_p to avoid MLX streaming path.
    Also strip Qwen <think> tags.
    """
    prompt = build_prompt(req.messages)

    full_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=req.max_tokens or args.max_tokens,
    )

    # Strip chain-of-thought tags if present
    full_text = re.sub(r"<think>.*</think>", "", full_text, flags=re.DOTALL)
    full_text = re.sub(r"<think>", "", full_text)
    full_text = re.sub(r"</think>\n", "", full_text)
    full_text.strip()
 
    return full_text


def sse_stream(req: ChatRequest) -> Generator[bytes, None, None]:
    """
    Stream ONLY the generated text in chunks.
    No assistant role.
    No wrapper beyond OpenAI-like delta format.
    """
    full_text = generate_text(req)

    chunk_size = 50
    idx = 0

    while idx < len(full_text):
        piece = full_text[idx:idx + chunk_size]
        idx += chunk_size

        # Safety: strip think tags in case they appear mid-chunk
        #piece = piece.replace("<think>", "").replace("</think>", "")

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

    # Merge extra_body into top-level JSON (for OpenAI-compatible clients)
    extra = body.get("extra_body", {})
    if isinstance(extra, dict):
        body.update(extra)

    # Normalize messages so Pydantic sees content as str, not list
    normalize_messages_in_body(body)

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


