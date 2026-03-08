from __future__ import annotations

import dataclasses
import difflib
import os
import time
from typing import Any, Dict, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "pass_through"   # "pass_through" | "gemini" | "openai" | "groq"
    prompt_version: str = "v1"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLMConfig":
        return LLMConfig(
            enabled=bool(d.get("enabled", False)),
            backend=str(d.get("backend", "pass_through")),
            prompt_version=str(d.get("prompt_version", "v1")),
            gemini_api_key=str(d.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", ""))),
            gemini_model=str(d.get("gemini_model", "gemini-2.0-flash-lite")),
            openai_api_key=str(d.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))),
            openai_model=str(d.get("openai_model", "gpt-4o-mini")),
            groq_api_key=str(d.get("groq_api_key", os.environ.get("GROQ_API_KEY", ""))),
            groq_model=str(d.get("groq_model", "llama-3.3-70b-versatile")),
        )


def prompt_template(version: str = "v1") -> str:
    if version != "v1":
        raise ValueError(f"Unknown prompt_version: {version}")
    return (
        "You are correcting OCR output from an early modern printed Spanish document "
        "(16th–17th century).\n"
        "Task: Correct OCR recognition errors ONLY.\n"
        "Rules:\n"
        "- Preserve historical spelling and orthography exactly; do NOT modernize.\n"
        "- Do NOT translate, summarize, or rewrite.\n"
        "- Preserve all line breaks and paragraph breaks.\n"
        "- Only fix clear OCR mistakes (wrong characters, garbled words).\n"
        "- Output ONLY the corrected text, nothing else.\n"
    )


def make_diff(raw_text: str, corrected_text: str) -> str:
    diff = difflib.unified_diff(
        raw_text.splitlines(keepends=True),
        corrected_text.splitlines(keepends=True),
        fromfile="ocr_raw", tofile="llm_corrected", lineterm="",
    )
    return "".join(diff)


def _correct_gemini(raw_text: str, cfg: LLMConfig) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=cfg.gemini_api_key)
        sdk = "new"
    except ImportError:
        import google.generativeai as genai_old
        genai_old.configure(api_key=cfg.gemini_api_key)
        sdk = "old"

    full_prompt = f"{prompt_template(cfg.prompt_version)}\n\nOCR text to correct:\n{raw_text}"
    for attempt in range(3):
        try:
            if sdk == "new":
                response = client.models.generate_content(
                    model=cfg.gemini_model, contents=full_prompt,
                )
                return response.text.strip() if response.text else raw_text
            else:
                model = genai_old.GenerativeModel(cfg.gemini_model)
                response = model.generate_content(full_prompt)
                return response.text.strip() if response.text else raw_text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"  [LLM/Gemini] Quota error, waiting {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    print(f"  [LLM/Gemini] Retries exhausted: {e}")
                    return raw_text
            else:
                print(f"  [LLM/Gemini] Error: {e}")
                return raw_text
    return raw_text


def _correct_openai(raw_text: str, cfg: LLMConfig) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install with: pip install openai")

    client = OpenAI(api_key=cfg.openai_api_key)
    full_prompt = f"{prompt_template(cfg.prompt_version)}\n\nOCR text to correct:\n{raw_text}"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=cfg.openai_model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip() or raw_text
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 20 * (attempt + 1)
                print(f"  [LLM/OpenAI] Rate limit, waiting {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    print(f"  [LLM/OpenAI] Retries exhausted: {e}")
                    return raw_text
            else:
                print(f"  [LLM/OpenAI] Error: {e}")
                return raw_text
    return raw_text


def _correct_groq(raw_text: str, cfg: LLMConfig) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install with: pip install groq")

    client = Groq(api_key=cfg.groq_api_key)
    full_prompt = f"{prompt_template(cfg.prompt_version)}\n\nOCR text to correct:\n{raw_text}"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=4096,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip() or raw_text
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 20 * (attempt + 1)
                print(f"  [LLM/Groq] Rate limit, waiting {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    print(f"  [LLM/Groq] Retries exhausted: {e}")
                    return raw_text
            else:
                print(f"  [LLM/Groq] Error: {e}")
                return raw_text
    return raw_text


def correct_text(
    raw_text: str,
    cfg: LLMConfig,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    metadata = dict(metadata or {})

    if not cfg.enabled or cfg.backend == "pass_through":
        return raw_text, {**metadata, "enabled": bool(cfg.enabled),
                          "backend": cfg.backend, "skipped": True,
                          "reason": "LLM disabled or pass-through."}

    if cfg.backend == "gemini":
        if not cfg.gemini_api_key:
            return raw_text, {**metadata, "skipped": True, "reason": "No Gemini key."}
        corrected = _correct_gemini(raw_text, cfg)
        return corrected, {**metadata, "enabled": True, "backend": "gemini",
                           "model": cfg.gemini_model, "skipped": False}

    if cfg.backend == "openai":
        if not cfg.openai_api_key:
            return raw_text, {**metadata, "skipped": True, "reason": "No OpenAI key."}
        corrected = _correct_openai(raw_text, cfg)
        return corrected, {**metadata, "enabled": True, "backend": "openai",
                           "model": cfg.openai_model, "skipped": False}

    if cfg.backend == "groq":
        if not cfg.groq_api_key:
            return raw_text, {**metadata, "skipped": True, "reason": "No Groq key."}
        corrected = _correct_groq(raw_text, cfg)
        return corrected, {**metadata, "enabled": True, "backend": "groq",
                           "model": cfg.groq_model, "skipped": False}

    raise RuntimeError(f"Unknown LLM backend: '{cfg.backend}'")