from langchain_groq import ChatGroq
from config import settings
from pydantic import SecretStr

llm = ChatGroq(
    model=settings.LLM_MODEL_NAME,
    api_key=SecretStr(settings.GROQ_API_KEY),
    temperature=0.2
)