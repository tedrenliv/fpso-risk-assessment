import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# Vector store
CHROMA_PERSIST_DIR = "C:/Multiagent/chroma_db"
COLLECTION_NAME = "risk_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Skill repository
SKILLS_DIR = "C:/Multiagent/skills"

# Human-in-the-loop timeouts (seconds, None = wait indefinitely)
HITL_TIMEOUT = None
