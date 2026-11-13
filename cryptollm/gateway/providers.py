"""Minimal LLM provider adapters (stdlib only).

* ``DryRunProvider``      - sends nothing; echoes what the provider *would* receive.
* ``OpenAICompatible``    - any /v1/chat/completions endpoint: OpenAI, a local Ollama /
                            llama.cpp / vLLM server, or a confidential-computing service that
                            exposes an OpenAI-style API (see docs/REPORT.md, section 4).
* ``Anthropic``           - the Anthropic Messages API.
"""

from __future__ import annotations

import json
import os
import urllib.request


class DryRunProvider:
    def complete(self, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        return f"(dry run - nothing was sent.) I would have received: {last}"


class OpenAICompatible:
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")

    def complete(self, messages: list[dict]) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": self.model, "messages": messages}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"]


class Anthropic:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 2048):
        self.model, self.max_tokens = model, max_tokens
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, messages: list[dict]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"])


SYSTEM_HINT = (
    "Some values in the user's message were replaced by placeholders such as [PERSON_1] or "
    "[EMAIL_2]. Treat each placeholder as an opaque real value, keep it exactly as written "
    "(including brackets) wherever you refer to it, and do not guess what it stands for."
)
