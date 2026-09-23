"""
Live integration probes for the Gemini provider.

These require a real GEMINI_API_KEY and are excluded from the default test
run (see tests_live/conftest.py). They verify the things a unit test cannot:
the model names in settings are actually served by the current Google account,
responses carry usage metadata, embeddings are 3072-dimensional floats, and
structured output round-trips through function calling.

Each test constructs providers from the same factories the app uses.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.core.config import settings
from src.llms.gemini import get_answer_llm, get_embeddings, get_llm

__all__ = ["_Answer"], get_embeddings, get_llm


class _Answer(BaseModel):
    answer: str
    confidence: float


def test_chat_model_in_settings_is_reachable(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Reply with exactly: pong")])
    assert response.content
    assert response.usage_metadata is not None


def test_system_message_is_obeyed(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content="Always answer with the single word orange."),
            HumanMessage(content="What color is a banana?"),
        ]
    )
    assert "orange" in response.content.lower()


def test_embeddings_are_3072_dimensional_floats(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    embeddings = get_embeddings()
    vector = embeddings.embed_query("what is adaptive retrieval")
    assert len(vector) == 3072
    assert all(isinstance(x, float) for x in vector)


def test_embed_documents_returns_parallel_vectors(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(
        ["first document", "second document", "a third, longer document"]
    )
    assert len(vectors) == 3
    assert all(len(v) == 3072 for v in vectors)


def test_structured_output_round_trips_through_tool_calling(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    llm = get_answer_llm().with_structured_output(_Answer)
    out = llm.invoke("Paris, 90% sure")
    assert isinstance(out, _Answer)
    assert out.answer and 0 <= out.confidence <= 1


def test_chat_model_lives_in_settings_model_name(gemini_api_key):
    settings.GEMINI_API_KEY = gemini_api_key
    llm = get_llm()
    assert llm.model == f"models/{settings.GEMINI_MODEL}"
