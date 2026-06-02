#!/usr/bin/env python3
"""
End-to-end local RAG pipeline using:

- MLX Qwen LLM server (chat/completions) on http://127.0.0.1:8899
- MLX Qwen embedding server (embeddings) on http://127.0.0.1:8898
- LangChain (LLM wrapper, embeddings wrapper, prompts, chains)
- Chroma as the vector store
- A single PDF as the knowledge source

This file is designed to be:

- Drop-in
- Deterministic
- Easy to extend
- Easy to debug
"""

import json
import requests
from typing import List

# LangChain core
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# LangChain community integrations
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ============================================================
# 1. Embedding wrapper for your MLX embedding server
# ============================================================

class MLXQwenEmbedding(Embeddings):
    """
    LangChain-compatible embedding class that talks to your
    MLX embedding server at /v1/embeddings.

    Assumes your server implements an OpenAI-style response:

    {
      "object": "list",
      "data": [
        {
          "object": "embedding",
          "index": 0,
          "embedding": [ ... floats ... ]
        }
      ],
      "model": "...",
      "usage": {...}
    }
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8898/v1/embeddings"):
        self.base_url = base_url

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        resp = requests.post(
            self.base_url,
            json={"input": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        For simplicity and determinism, we call the server once per text.
        If you want speed, you can batch them and send as a list.
        """
        return [self.embed_query(t) for t in texts]


# ============================================================
# 2. LLM wrapper for your MLX Qwen chat server
# ============================================================

class MLXQwenChat(LLM):
    """
    LangChain-compatible LLM that talks to your MLX Qwen LLM server
    at /v1/chat/completions.

    Your server:

    - Accepts OpenAI-style chat payloads
    - Internally converts messages → Qwen prompt
    - Strips <think> tags
    - Returns plain text (non-OpenAI JSON) in non-streaming mode
    """

    base_url: str = "http://127.0.0.1:8899/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "mlx-qwen-chat"

    def _call(self, prompt: str, stop=None) -> str:
        """
        Send a single user message with the given prompt.

        We keep this minimal and deterministic:
        - No streaming
        - No temperature / top_p
        """
        body = {
            "model": "mlx",  # your server ignores or uses this as a label
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }

        resp = requests.post(self.base_url, json=body, timeout=120)
        resp.raise_for_status()

        # Your server returns raw text (not JSON) in non-streaming mode.
        return resp.text


# ============================================================
# 3. Qwen prompt formatting helper
# ============================================================

def format_qwen_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Format a prompt in Qwen chat template style.

    NOTE: In this pipeline, we actually let the server do the
    system/user handling internally, but this helper is useful
    if you want to push a fully formatted Qwen prompt yourself.

    For now, we use it only conceptually; the LLMChain prompt
    is plain text, and your server builds the Qwen prompt.
    """
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ============================================================
# 4. Build vector store (PDF → chunks → embeddings → Chroma)
# ============================================================

def build_vectorstore(
    pdf_path: str,
    persist_dir: str = "article_db",
) -> Chroma:
    """
    Load a PDF, split into chunks, embed with MLXQwenEmbedding,
    and store in a persistent Chroma DB.
    """

    # 1) Load PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # 2) Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # 3) Embedding model (your MLX embedding server)
    embedding_model = MLXQwenEmbedding()

    # 4) Build / persist Chroma vector store
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=persist_dir,
    )

    return vectorstore


# ============================================================
# 5. Build RAG chain (retriever + prompt + LLM)
# ============================================================

def build_rag_chain(vectorstore: Chroma):

    system_prompt = (
        "You are an academic research assistant. "
        "Use ONLY the retrieved context. "
        "Do NOT reveal chain-of-thought. "
        "Answer concisely and factually."
    )

    template = """
{system}

Context:
{context}

Question:
{question}

Answer:
""".strip()

    prompt = PromptTemplate(
        input_variables=["system", "context", "question"],
        template=template,
    )

    llm = MLXQwenChat()

    chain = prompt | llm

    # attach custom state
    chain._vectorstore = vectorstore
    chain._system_prompt = system_prompt

    return chain


# ============================================================
# 6. RAG answer function
# ============================================================

def rag_answer(chain: LLMChain, question: str, k: int = 10) -> str:
    """
    High-level RAG function:

    - Uses the vectorstore attached to the chain
    - Retrieves top-k relevant chunks
    - Calls the LLMChain with system/context/question
    - Returns the model's answer as a string
    """

    vectorstore: Chroma = chain._vectorstore
    system_prompt: str = chain._system_prompt

    # 1) Build retriever from vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 2) Retrieve relevant documents
    docs = retriever.invoke(question)

    # 3) Concatenate their content into a single context string
    context = "\n\n".join(d.page_content for d in docs)

    # 4) Run the LLMChain
    result = chain.invoke(
        {
            "system": system_prompt,
            "context": context,
            "question": question,
        }
    )

    answer = result.encode().decode("unicode_escape")
    return answer.strip('"').strip()


# ============================================================
# 7. Main: build everything and ask a few questions
# ============================================================

if __name__ == "__main__":
    # 1) Build or load vectorstore from your PDF
    #    Change "article.pdf" to your actual document.
    pdf_path = "article.pdf"
    vectorstore = build_vectorstore(pdf_path, persist_dir="article_db")

    # 2) Build RAG chain
    rag_chain = build_rag_chain(vectorstore)

    # 3) Ask questions
    questions = [
        "What is this article about?",
        "Who are the authors?",
        "What are the main conclusions?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {q}")
        print("-" * 80)
        ans = rag_answer(rag_chain, q)
        print(ans)


