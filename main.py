from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import *
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI  # CHANGED: Import OpenAI instead of Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage
from qdrant_client import QdrantClient
import redis
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
import traceback

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

# === Moteur RAG ===
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(persist_dir="./storage", vector_store=vector_store)
index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=5)

# CHANGED: Initialize OpenAI LLM instead of Ollama
llm = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)  # e.g., OPENAI_MODEL = "gpt-3.5-turbo"

# === Fonctions mémoire Redis ===
def get_user_memory(user_id: str) -> ChatMemoryBuffer:
    key = f"user_memory:{user_id}"
    raw = redis_client.get(key)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    if raw:
        messages = json.loads(raw)
        memory.reset()
        memory.set([
            ChatMessage(role=m["role"], content=m["content"]) for m in messages
        ])
    return memory

def save_user_memory(user_id: str, memory: ChatMemoryBuffer):
    key = f"user_memory:{user_id}"
    messages = [{"role": m.role, "content": m.content} for m in memory.get_all()]
    redis_client.setex(key, 300, json.dumps(messages))  # TTL = 300 sec = 5 min
    print(f"Memory saved to Redis for user {user_id}: {messages}")

# === Pydantic Model ===
class ChatInput(BaseModel):
    message: str
    user_id: str

# === Streaming functions ===
async def stream_chat_response(input: ChatInput) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    try:
        memory = get_user_memory(input.user_id)
        print(f"Memory for user {input.user_id}: {memory.get_all()}")
        
        # CHANGED: Configure OpenAI for streaming
        streaming_llm = OpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, stream=True)
        
        chat_engine = ContextChatEngine.from_defaults(
            llm=streaming_llm, retriever=retriever, memory=memory
        )
        print(f"Streaming chat engine initialized for user {input.user_id}")
        
        # Get the streaming response object
        streaming_response = chat_engine.stream_chat(input.message)
        
        # Debug info about the streaming response object
        print(f"Type of streaming response: {type(streaming_response)}")
        print(f"Dir of streaming response: {dir(streaming_response)}")
        
        # Track complete response for memory
        complete_response = ""
        
        # Use proper method to get the actual streaming content
        if hasattr(streaming_response, 'response_gen'):
            # This is the case for newer LlamaIndex versions
            response_iter = streaming_response.response_gen
            for token in response_iter:
                if token:
                    complete_response += token
                    yield token
                    await asyncio.sleep(0.01)
        else:
            # Fallback for other response types
            response_text = str(streaming_response)
            complete_response = response_text
            yield response_text
        
        # After streaming completes, update memory with the full conversation
        memory.put(ChatMessage(role="user", content=input.message))
        memory.put(ChatMessage(role="assistant", content=complete_response))
        save_user_memory(input.user_id, memory)
        print(f"Memory after chat for user {input.user_id}: {memory.get_all()}")
        
        # Send a special marker to indicate end of streaming
        yield "\n[END_OF_RESPONSE]"
        
    except Exception as e:
        error_msg = f"Streaming chat failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())  # Print detailed stack trace
        yield json.dumps({"error": error_msg})

# === Endpoint Chat Streaming ===
@app.post("/chat/stream")
async def chat_stream(input: ChatInput):
    """Endpoint that returns a streaming response."""
    
    async def event_generator():
        async for chunk in stream_chat_response(input):
            if chunk == "\n[END_OF_RESPONSE]":
                # Send a completion event
                yield f"data: [DONE]\n\n"
            else:
                # Format as SSE event
                yield f"data: {json.dumps({'text': chunk})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# === Non-streaming endpoint ===
@app.post("/chat")
async def chat(input: ChatInput):
    try:
        memory = get_user_memory(input.user_id)
        print(f"Memory for user {input.user_id}: {memory.get_all()}")
        
        chat_engine = ContextChatEngine.from_defaults(
            llm=llm, retriever=retriever, memory=memory
        )
        print(f"Chat engine initialized for user {input.user_id}")
        
        response = chat_engine.chat(input.message)
        print(f"Response for user {input.user_id}: {response}")
        
        save_user_memory(input.user_id, memory)
        print(f"Memory after chat for user {input.user_id}: {memory.get_all()}")
        
        return {
            "response": str(response),
            "history": [m.content for m in memory.get_all()]
        }
    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")