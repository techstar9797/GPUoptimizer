"""OpenAI chat calls, with token accounting for the ledger."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SETTINGS


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


_client = None


def complete(system: str, user: str, *, temperature: float = 0.8, max_tokens: int = 4000) -> Reply:
    global _client
    if not SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env in the repo root.")
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=SETTINGS.openai_api_key, max_retries=5, timeout=120)
    kwargs = dict(model=SETTINGS.openai_model,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    # Reasoning models reject temperature and use max_completion_tokens.
    if re.match(r"^(o\d|gpt-5)", SETTINGS.openai_model):
        kwargs["max_completion_tokens"] = max_tokens * 4
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    r = _client.chat.completions.create(**kwargs)
    u = r.usage
    return Reply(text=r.choices[0].message.content or "", prompt_tokens=u.prompt_tokens if u else 0,
                 completion_tokens=u.completion_tokens if u else 0, model=r.model)


def extract_code(text: str) -> str:
    """Pull the python block out of a model reply."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        with_score = [b for b in blocks if "def score" in b]
        return (with_score or blocks)[-1].strip()
    return text.strip()


def extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text
