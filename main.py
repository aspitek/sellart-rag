from fastapi import FastAPI, Request
from pydantic import BaseModel
from config import *
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage
from qdrant_client import QdrantClient
import redis
import json

# === Redis setup ===
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# === FastAPI ===
app = FastAPI()

# === CORS setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines
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
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

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

# === Endpoint Chat ===
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
            "history": [m.content for m in memory.get_all()]  # Changed m.content to m["content"]
        }

    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")