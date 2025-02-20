import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AIRAConfig:
    # API Keys and Authentication
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "FAISS"  # Options: FAISS, Weaviate, ChromaDB
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    VECTOR_STORE_PATH: str = "vector_stores"
    
    # LLM Settings
    LLM_MODEL: str = "gpt-4"  # Options: gpt-4, gemini-pro, llama-3
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    
    # Data Sources
    ARXIV_CATEGORIES: List[str] = ["cs.AI", "cs.CL", "cs.LG"]
    MAX_PAPERS_PER_QUERY: int = 100
    UPDATE_FREQUENCY_HOURS: int = 24
    
    # Agent Settings
    AGENT_TYPES: List[str] = ["retrieval", "summarization", "knowledge_graph", "execution"]
    MEMORY_TYPE: str = "buffer"  # Options: buffer, summary, conversational
    MEMORY_KEY: str = "chat_history"
    
    # Database Settings
    DB_CONNECTION: str = "sqlite:///aira.db"
    
    # Feedback Settings
    MIN_FEEDBACK_THRESHOLD: float = 0.7
    FEEDBACK_BATCH_SIZE: int = 50

    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            SEMANTIC_SCHOLAR_API_KEY=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            GOOGLE_CLOUD_PROJECT=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        )

# Create default config instance
config = AIRAConfig()
