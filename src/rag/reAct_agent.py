"""
ReAct agent construction.

The agent is built **per user and per index version**. The previous
implementation created a single agent at import time, permanently binding it
to the placeholder index that existed before any upload; documents uploaded
later were never reachable through it.

Executors are cached so the hot query path does not rebuild an agent on every
request, and the cache key includes the index version so an upload
invalidates it immediately.
"""

import threading
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_llm
from src.rag import vector_store

logger = get_logger(__name__)

config = Config()

_lock = threading.RLock()
# user_id -> (index_version, executor)
_executor_cache: dict[str, tuple[int, AgentExecutor]] = {}


def _build_executor(user_id: str) -> Optional[AgentExecutor]:
    """Construct a fresh executor bound to the user's current index."""
    retriever_tool = vector_store.get_retriever_tool(user_id)
    if retriever_tool is None:
        return None

    tools = [retriever_tool]
    prompt = PromptTemplate.from_template(config.prompt("system_prompt"))

    agent = create_react_agent(get_llm(), tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        early_stopping_method="force",
        return_intermediate_steps=True,
        verbose=False,
    )


def get_agent_executor(user_id: str) -> Optional[AgentExecutor]:
    """
    Return an agent executor bound to the user's current documents.

    Args:
        user_id: The owning user.

    Returns:
        A cached or newly built :class:`AgentExecutor`, or None when the user
        has not uploaded any documents.
    """
    version = vector_store.get_version(user_id)
    if version == 0:
        return None

    with _lock:
        cached = _executor_cache.get(user_id)
        if cached is not None and cached[0] == version:
            return cached[1]

        executor = _build_executor(user_id)
        if executor is None:
            _executor_cache.pop(user_id, None)
            return None

        logger.info("Built ReAct agent for index version %d", version)
        _executor_cache[user_id] = (version, executor)
        return executor


def reset_cache(user_id: Optional[str] = None) -> None:
    """
    Drop cached executors.

    Args:
        user_id: Drop only this user's executor; drop all when None.
    """
    with _lock:
        if user_id is None:
            _executor_cache.clear()
        else:
            _executor_cache.pop(user_id, None)
