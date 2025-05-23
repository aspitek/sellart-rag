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
import redis
import json
import asyncio
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

AFRO_CENTRIC_SYSTEM_PROMPT = """
You are an AI curator for an online African art gallery, deeply knowledgeable about African art, culture, and history.
Use *only* the context from the retrieved documents to answer questions. Do not invent facts.
Focus on African art, its cultural significance, historical context, and contemporary expressions.
Respond warmly.
Use the user's language when possible, and be grammatically precise.
For questions about evolution or history, provide a response with scientific rigor with proof like statistics, dates, and events etc.
"""

# === Cross-encoder reranker setup ===
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

# === Mémoire utilisateur ===
def get_user_memory(user_id: str) -> ChatMemoryBuffer:
    key = f"user_memory:{user_id}"
    raw = redis_client.get(key)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    if raw:
        try:
            messages = json.loads(raw)
            memory.reset()
            memory.set([ChatMessage(role=m["role"], content=m["content"]) for m in messages])
        except Exception as e:
            print(f"Failed to load memory for user {user_id}: {e}")
    return memory

def save_user_memory(user_id: str, memory: ChatMemoryBuffer):
    key = f"user_memory:{user_id}"
    messages = [{"role": m.role, "content": m.content} for m in memory.get_all()]
    redis_client.setex(key, 3600, json.dumps(messages))  # TTL 1h
    print(f"Memory saved to Redis for user {user_id}: {messages}")

# === Résumé automatique ===
async def summarize_memory(memory: ChatMemoryBuffer, llm: OpenAI) -> str:
    """
    Utilise le LLM pour résumer l'historique complet en une version plus courte.
    """
    convo_text = "\n".join([f"{m.role}: {m.content}" for m in memory.get_all()])
    prompt = (
        "Résumé l'historique de la conversation suivante de manière concise "
        "en gardant les informations importantes utiles pour le contexte futur. "
        "Garde le style professionnel et concis.\n\n"
        f"Conversation:\n{convo_text}\n\nRésumé:"
    )
    try:
        response = llm.chat(prompt)
        return str(response).strip()
    except Exception as e:
        print(f"Erreur lors du résumé: {e}")
        return convo_text  # fallback

def update_memory_with_summary(memory: ChatMemoryBuffer, summary: str):
    """
    Remplace la mémoire complète par un résumé unique (role=system).
    """
    memory.reset()
    memory.put(ChatMessage(role="system", content="Résumé de la conversation précédente : " + summary))

# === Création moteur chat avec reranking intégré ===
def create_chat_engine(memory: ChatMemoryBuffer, streaming: bool = False) -> ContextChatEngine:
    # Wrapper autour du retriever pour appliquer reranking via cross-encoder
    original_retrieve = retriever.retrieve

    def reranked_retrieve(query: str):
        docs = original_retrieve(query)
        # docs is a list of Document objects: convert to (id, text)
        docs_tuples = [(str(i), d.get_content()) for i, d in enumerate(docs)]
        ranked = rerank(query, docs_tuples)
        # Return Documents objects in ranked order
        return [docs[int(idx)] for idx, _ in ranked]

    # Patch temporairement le retriever
    retriever.retrieve = reranked_retrieve

    llm_instance = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=streaming,
    )

    engine = ContextChatEngine.from_defaults(
        llm=llm_instance,
        retriever=retriever,
        memory=memory,
        system_prompt=AFRO_CENTRIC_SYSTEM_PROMPT,
    )

    # Restore original retrieve to avoid side effects elsewhere
    retriever.retrieve = original_retrieve

    return engine

# === Modèle Pydantic ===
class ChatInput(BaseModel):
    message: str
    user_id: str

# === Streaming chat response avec résumé automatique & logs ===
# ... [imports identiques jusqu'à ici]

# === Fonction de routage via prompt OpenAI ===
async def determine_routing_via_prompt(query: str, llm: OpenAI) -> str:
    prompt = (
        "Tu es un assistant intelligent pour un moteur RAG spécialisé en art africain.\n"
        "On te donne une question utilisateur. Tu dois dire si elle peut être répondue avec des documents de la base de données (réponds 'rag'),\n"
        "ou si elle nécessite une recherche sur internet (réponds 'websearch').\n"
        "Sois conservateur — si tu n'es pas sûr, choisis 'websearch'.\n"
        f"Question: {query}\nRéponse (juste 'rag' ou 'websearch'):"
    )
    try:
        result = llm.chat(prompt)
        answer = str(result).strip().lower()
        if "web" in answer:
            return "websearch"
        return "rag"
    except Exception as e:
        print(f"Erreur routage prompt: {e}")
        return "rag"  # fallback

# === Web search fallback simple (dummy) ===
async def fallback_web_search_answer(query: str, llm: OpenAI) -> str:
    prompt = (
        "Tu es un assistant intelligent qui répond à la question suivante en simulant une recherche web.\n"
        "Base ta réponse sur des faits disponibles publiquement jusqu'à 2024 et donne des informations crédibles.\n"
        f"Question: {query}\nRéponse:"
    )
    try:
        result = llm.chat(prompt)
        return str(result).strip()
    except Exception as e:
        print(f"Erreur web search fallback: {e}")
        return "Je suis désolé, je ne peux pas répondre pour le moment."

# === Streaming chat response avec routage ===
async def stream_chat_response(input: ChatInput) -> AsyncGenerator[str, None]:
    try:
        memory = get_user_memory(input.user_id)
        print(f"Memory size for user {input.user_id}: {len(memory.get_all())} messages")

        if len(memory.get_all()) > 20:
            llm_summary = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
            summary = await summarize_memory(memory, llm_summary)
            update_memory_with_summary(memory, summary)

        llm_router = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
        routing = await determine_routing_via_prompt(input.message, llm_router)
        print(f"Routing decision for user {input.user_id}: {routing}")

        if routing == "websearch":
            answer = await fallback_web_search_answer(input.message, llm_router)
            yield json.dumps({"text": answer})
            memory.put(ChatMessage(role="user", content=input.message))
            memory.put(ChatMessage(role="assistant", content=answer))
            save_user_memory(input.user_id, memory)
            yield "\n[END_OF_RESPONSE]"
            return

        chat_engine = create_chat_engine(memory, streaming=True)
        streaming_response = chat_engine.stream_chat(input.message)
        complete_response = ""

        if hasattr(streaming_response, 'response_gen'):
            async for token in streaming_response.response_gen:
                if token:
                    complete_response += token
                    yield f"data: {json.dumps({'text': token})}\n\n"
        else:
            response_text = str(streaming_response)
            complete_response = response_text
            yield f"data: {json.dumps({'text': response_text})}\n\n"

        memory.put(ChatMessage(role="user", content=input.message))
        memory.put(ChatMessage(role="assistant", content=complete_response))
        save_user_memory(input.user_id, memory)
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = f"Streaming chat failed: {str(e)}"
        print(traceback.format_exc())
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# === Endpoint streaming avec routage ===
@app.post("/chat/stream")
async def chat_stream(input: ChatInput):
    return StreamingResponse(stream_chat_response(input), media_type="text/event-stream")

# === Endpoint non-streaming avec routage ===
@app.post("/chat")
async def chat(input: ChatInput):
    try:
        memory = get_user_memory(input.user_id)

        if len(memory.get_all()) > 20:
            llm_summary = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
            summary = await summarize_memory(memory, llm_summary)
            update_memory_with_summary(memory, summary)

        llm_router = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
        routing = await determine_routing_via_prompt(input.message, llm_router)
        print(f"Routing decision: {routing}")

        if routing == "websearch":
            answer = await fallback_web_search_answer(input.message, llm_router)
            memory.put(ChatMessage(role="user", content=input.message))
            memory.put(ChatMessage(role="assistant", content=answer))
            save_user_memory(input.user_id, memory)
            return {
                "response": answer,
                "history": [m.content for m in memory.get_all()]
            }

        chat_engine = create_chat_engine(memory, streaming=False)
        response = chat_engine.chat(input.message)
        memory.put(ChatMessage(role="user", content=input.message))
        memory.put(ChatMessage(role="assistant", content=str(response)))
        save_user_memory(input.user_id, memory)

        return {
            "response": str(response),
            "history": [m.content for m in memory.get_all()]
        }

    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

# === Lancement de l'API ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
