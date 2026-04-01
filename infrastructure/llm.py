from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

# -------------------------------
# GROQ (kept for reference)
# -------------------------------
from langchain_groq import ChatGroq
llm = ChatGroq(
    model=settings.LLM_MODEL_NAME,
    api_key=settings.GROQ_API_KEY,
    temperature=0.2,
)

# -------------------------------
# GEMINI (active)
# -------------------------------
# llm = ChatGoogleGenerativeAI(
#     model=settings.LLM_MODEL_NAME,
#     google_api_key=settings.GOOGLE_API_KEY,
#     temperature=0.2,
# )