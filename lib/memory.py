from config import *
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage


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
        "Résume l'historique de la conversation suivante de manière concise "
        "en gardant les informations importantes utiles pour le contexte futur. "
        "Garde le style professionnel et concis.\n\n"
        f"Conversation:\n{convo_text}\n\nRésumé:"
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Erreur lors du résumé: {e}")
        return convo_text  # fallback

def update_memory_with_summary(memory: ChatMemoryBuffer, summary: str):
    """
    Remplace la mémoire complète par un résumé unique (role=system).
    """
    memory.reset()
    memory.put(ChatMessage(role="system", content="Résumé de la conversation précédente : " + summary))
