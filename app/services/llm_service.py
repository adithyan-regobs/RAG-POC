from functools import lru_cache

from openai import OpenAI

from app.config import settings

GROUNDED_SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer the user's question using ONLY "
    "the information in the CONTEXT section. Do NOT use any prior or external "
    "knowledge. If the context does not contain enough information to answer, "
    'reply exactly: "I don\'t know based on the provided context." '
    "When you do answer, cite supporting sources inline using the [id=...] "
    "markers that appear in the context blocks. Be concise and precise."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful legal assistant. The current message does not require "
    "consulting legal reference material. Respond conversationally and briefly. "
    "If the user asks a substantive legal question, encourage them to rephrase "
    "so you can look it up."
)

NO_CONTEXT_FALLBACK = "I don't know based on the provided context."


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def generate_answer(question: str, context: str) -> str:
    """Generate a grounded answer restricted to the supplied context."""
    if not context.strip():
        return NO_CONTEXT_FALLBACK

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    resp = _client().chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_direct_answer(question: str) -> str:
    """Answer without retrieval — used for small-talk / general queries."""
    resp = _client().chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
