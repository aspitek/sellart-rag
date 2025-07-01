# === Modèle Pydantic ===
from pydantic import BaseModel

class ChatInput(BaseModel):
    message: str
    user_id: str