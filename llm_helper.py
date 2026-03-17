from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


class LLMHelper:
    """
    Wrapper around ChatGroq with simple disk cache keyed by concept_id.
    If the same concept_id is requested again, cached text is returned and
    no API call is made.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        cache_path: str = "anything/llm_cache.json",
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        self.cache_path = self._resolve_path(cache_path)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [Path.cwd() / p, Path(__file__).resolve().parent.parent / p]
        return next((c for c in candidates if c.exists()), candidates[0])

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8", errors="replace") as f:
            return json.load(f)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def invoke_for_concept(
        self,
        concept_id: str,
        prompt: str,
        namespace: str = "default",
        force_refresh: bool = False,
    ) -> str:
        """
        Cache key format: <namespace>::<concept_id>
        Use namespace if you run multiple prompt types for same concept.
        """
        cache_key = f"{namespace}::{concept_id}"

        if not force_refresh and cache_key in self.cache:
            return str(self.cache[cache_key]["response"])

        answer = self.llm.invoke(prompt)
        content = answer.content if hasattr(answer, "content") else str(answer)

        self.cache[cache_key] = {
            "concept_id": concept_id,
            "namespace": namespace,
            "response": content,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_cache()
        return content

    def clear_cache(self, concept_id: Optional[str] = None, namespace: Optional[str] = None) -> int:
        """
        Remove cache entries and return number of removed items.
        """
        keys = list(self.cache.keys())
        removed = 0
        for k in keys:
            item = self.cache.get(k, {})
            if concept_id and item.get("concept_id") != concept_id:
                continue
            if namespace and item.get("namespace") != namespace:
                continue
            del self.cache[k]
            removed += 1
        if removed:
            self._save_cache()
        return removed


if __name__ == "__main__":
    helper = LLMHelper()
    text = helper.invoke_for_concept(
        concept_id="demo-cid",
        namespace="sanity",
        prompt="What is the capital of India? Answer in one line.",
    )
    print(text)
