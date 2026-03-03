"""Configuration for the mock interview system."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    base: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    def __post_init__(self) -> None:
        self.rag_server: Path = self.base / "RAG-MCP-SERVER"
        self.data:       Path = self.base / "data"
        self.uploads:    Path = self.data / "uploads"
        self.output:     Path = self.data / "output"


@dataclass
class ModelConfig:
    base_url:    str   = field(default_factory=lambda: os.getenv("VLLM_BASE_URL",   "http://localhost:8000/v1/"))
    name:        str   = field(default_factory=lambda: os.getenv("VLLM_MODEL_NAME", "Qwen3-8B"))
    api_key:     str   = field(default_factory=lambda: os.getenv("VLLM_API_KEY",    "not-needed"))
    temperature: float = 0.7
    max_tokens:  int   = 512
    top_p:       float = 0.9


@dataclass
class MCPConfig:
    command: str
    env:     dict


@dataclass
class InterviewConfig:
    num_questions:  int = 5
    max_follow_ups: int = 1


@dataclass
class RAGConfig:
    resume:          str = "resume"
    job_description: str = "job_description"
    default:         str = "default"


# ─── Singleton instances ───────────────────────────────────────────────────────
paths    = PathConfig()
model    = ModelConfig()
mcp      = MCPConfig(
    command = str(paths.base / "scripts" / "start_rag_server.sh"),
    env     = {**os.environ, "PYTHONPATH": str(paths.rag_server)},
)
interview = InterviewConfig()
rag       = RAGConfig()
