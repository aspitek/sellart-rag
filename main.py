import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import *
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage
from qdrant_client import QdrantClient
from openai import OpenAI as OpenAIClient
import redis
import json
import asyncio
from collections.abc import AsyncIterator
import traceback
from typing import AsyncGenerator, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# === Redis setup ===
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# === FastAPI ===
app = FastAPI()

# === CORS setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === RAG setup ===
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(persist_dir="./storage", vector_store=vector_store)
index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

# Récupérer plus de documents
retriever = index.as_retriever(similarity_top_k=7)

openai_client = OpenAIClient(api_key=OPENAI_API_KEY)



print("Loading reranker model...")
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
reranker_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker_model.to(device)
print(f"Reranker loaded on device {device}")

def rerank(query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    docs: list of (doc_id, doc_text)
    Returns docs sorted by relevance descending.
    """
    pairs = [(query, doc_text) for _, doc_text in docs]
    inputs = reranker_tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = reranker_model(**inputs).logits[:, 1].cpu().numpy()  # score for class 1 (relevant)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in scored_docs]



# === Endpoint streaming avec routage ===
@app.post("/chat/stream")
async def chat_stream(input: ChatInput):
    return StreamingResponse(
        stream_chat_response(input), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# === Lancement de l'API ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)