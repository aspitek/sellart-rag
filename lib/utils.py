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
        max_tokens=800,
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

    retriever.retrieve = original_retrieve

    return engine



# === Fonction de routage via prompt OpenAI ===
async def determine_routing_via_prompt(memory: ChatMemoryBuffer, new_question: str, llm: OpenAI) -> str:
    """
    Utilise l'historique de conversation + la question actuelle pour déterminer si on route vers RAG ou WebSearch.
    """
    context = "\n".join([f"{m.role}: {m.content}" for m in memory.get_all()])
    full_prompt = (
        "Tu es un assistant intelligent pour un moteur RAG spécialisé en art africain.\n"
        "Voici l'historique de la conversation entre un utilisateur et l'assistant :\n"
        f"{context}\n\n"
        "L'utilisateur pose maintenant la question suivante :\n"
        f"{new_question}\n\n"
        "Dois-tu utiliser la base de documents (réponds 'rag') ou effectuer une recherche sur Internet (réponds 'websearch') ?\n"
        "Sois conservateur — si tu n'es pas sûr que la base de documents suffise, choisis 'websearch'.\n"
        "Réponse (juste 'rag' ou 'websearch') :"
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        if "web" in answer:
            return "websearch"
        return "rag"
    except Exception as e:
        print(f"Erreur routage prompt: {e}")
        return "rag"  # fallback

# === Web search fallback simple ===
async def fallback_web_search_answer(query: str, llm: OpenAI) -> AsyncGenerator[str, None]:
    """
    Version streaming de la recherche web fallback
    """
    prompt = (
        "Tu es un assistant intelligent qui répond à la question suivante en simulant une recherche web.\n"
        "Base ta réponse sur des faits disponibles publiquement jusqu'à 2024 et donne des informations crédibles.\n"
        "retire les liens de ta reponse mais insere les references tout en gardant une reponse professionelle.\n"
        "ramene la reponse de façon complète\n"
        f"Question: {query}\nRéponse:"
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Erreur web search fallback: {e}")
        yield "Je suis désolé, je ne peux pas répondre pour le moment."

# === Streaming chat response avec routage UNIFIÉ ===
async def stream_chat_response(input: ChatInput) -> AsyncGenerator[str, None]:
    try:
        memory = get_user_memory(input.user_id)
        print(f"Memory size for user {input.user_id}: {len(memory.get_all())} messages")

        # Résumé automatique si nécessaire
        if len(memory.get_all()) > 20:
            llm_summary = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
            summary = await summarize_memory(memory, llm_summary)
            update_memory_with_summary(memory, summary)

        # Déterminer le routage
        llm_router = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
        routing = await determine_routing_via_prompt(memory, input.message, llm_router)
        print(f"Routing decision for user {input.user_id}: {routing}")

        complete_response = ""

        # === FORMAT UNIFIÉ POUR TOUS LES TYPES DE RÉPONSE ===
        if routing == "websearch":
            # Streaming web search
            async for token in fallback_web_search_answer(input.message, llm_router):
                if token:
                    complete_response += token
                    yield f"data: {json.dumps({'text': token})}\n\n"
        else:
            # Streaming RAG
            chat_engine = create_chat_engine(memory, streaming=True)
            streaming_response = chat_engine.stream_chat(input.message)

            if hasattr(streaming_response, 'response_gen'):
                if isinstance(streaming_response.response_gen, AsyncIterator):
                    async for token in streaming_response.response_gen:
                        if token:
                            complete_response += token
                            yield f"data: {json.dumps({'text': token})}\n\n"
                else:
                    for token in streaming_response.response_gen:
                        if token:
                            complete_response += token
                            yield f"data: {json.dumps({'text': token})}\n\n"
            else:
                # Fallback si pas de streaming
                response_text = str(streaming_response)
                complete_response = response_text
                # Simuler le streaming en envoyant par chunks
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.05)  # Petit délai pour simuler le streaming

        # Sauvegarder la conversation
        memory.put(ChatMessage(role="user", content=input.message))
        memory.put(ChatMessage(role="assistant", content=complete_response))
        save_user_memory(input.user_id, memory)
        
        # Signal de fin unifié
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = f"Streaming chat failed: {str(e)}"
        print(traceback.format_exc())
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
