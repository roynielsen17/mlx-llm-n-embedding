#!/usr/bin/env python3

import argparse
import json
import re
from typing import List, Optional, Generator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# --------------------------------------------------
# CLI Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(description="MLX OpenAI-Compatible Server")

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="MLX model path"
)

parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1"
)

parser.add_argument(
    "--port",
    type=int,
    default=8899
)

parser.add_argument(
    "--max-tokens",
    type=int,
    default=2048
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.7
)

parser.add_argument(
    "--top-p",
    type=float,
    default=0.95
)

parser.add_argument(
    "--top-k",
    type=int,
    default=50
)

args = parser.parse_args()


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print(f"Loading model: {args.model}")

model, tokenizer = load(args.model)

print("Model loaded.")

app = FastAPI()


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str  # we normalize to str before validation


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    max_tokens: Optional[int] = None

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_messages_in_body(body: dict) -> None:
    """
    Convert OpenAI message content lists into strings.
    """

    messages = body.get("messages", [])

    if not isinstance(messages, list):
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")

        if isinstance(content, list):
            msg["content"] = " ".join(
                str(x)
                for x in content
            )


def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert OpenAI messages into a Qwen chat prompt.
    """

    system = ""
    conversation = []

    for m in messages:

        if m.role == "system":
            system = m.content

        elif m.role == "user":
            conversation.append(
                f"<|im_start|>user\n{m.content}\n<|im_end|>"
            )

        elif m.role == "assistant":
            conversation.append(
                f"<|im_start|>assistant\n{m.content}\n<|im_end|>"
            )

    prompt = (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        + "\n".join(conversation)
        + "\n<|im_start|>assistant\n"
    )

    return prompt


def strip_think_tags(text: str) -> str:

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    )

    text = text.replace("<think>", "")
    text = text.replace("</think>", "")

    return text.strip()


def generate_text(req: ChatRequest) -> str:

    prompt = build_prompt(req.messages)

    temperature = (
        req.temperature
        if req.temperature is not None
        else args.temperature
    )

    top_p = (
        req.top_p
        if req.top_p is not None
        else args.top_p
    )

    top_k = (
        req.top_k
        if req.top_k is not None
        else args.top_k
    )

    max_tokens = (
        req.max_tokens
        if req.max_tokens is not None
        else args.max_tokens
    )

    sampler = make_sampler(
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # Strip chain-of-thought tags if present
    #full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    #full_text = re.sub(r"<think>", "", full_text)
    #full_text = re.sub(r"</think>\n", "", full_text)
    #full_text.strip()

    return strip_think_tags(text)


# --------------------------------------------------
# SSE Streaming
# --------------------------------------------------

def sse_stream(
    req: ChatRequest
) -> Generator[bytes, None, None]:

    text = generate_text(req)

    chunk_size = 50

    for i in range(0, len(text), chunk_size):

        piece = text[i:i + chunk_size]

        payload = {
            "choices": [
                {
                    "delta": {
                        "content": piece
                    }
                }
            ]
        }

        yield (
            f"data: {json.dumps(payload)}\n\n"
        ).encode("utf-8")

    yield b"data: [DONE]\n\n"


# --------------------------------------------------
# Route
# --------------------------------------------------

@app.post("/v1/chat/completions")
async def chat(request: Request):

    body = await request.json()

    extra = body.get("extra_body")

    if isinstance(extra, dict):
        body.update(extra)

    # Normalize messages so Pydantic sees content as str, not list
    normalize_messages_in_body(body)

    # Parse into ChatRequest
    req = ChatRequest(**body)

    stream = body.get("stream", False)

    if stream:

        return StreamingResponse(
            sse_stream(req),
            media_type="text/event-stream",
        )

    # Non-streaming mode: return ONLY the generated text
    text = generate_text(req)
    return text

    return JSONResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                    }
                }
            ]
        }
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Starting streaming MLX LLM server on http://{args.host}:{args.port}")

    print(
        f"Starting MLX server on "
        f"http://{args.host}:{args.port}"
    )

    print(
        f"Defaults: "
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"top_k={args.top_k}"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )

