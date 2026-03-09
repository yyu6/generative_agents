"""LLM client wrappers for Gemini/OpenAI JSON decisions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


class AgentLLMClient:
  def __init__(
    self,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: int,
  ):
    self.provider = (provider or "").strip().lower() or "gemini"
    self.model = (model or "").strip() or (
      "gemini-1.5-flash" if self.provider == "gemini" else "gpt-4o-mini"
    )
    self.temperature = float(temperature)
    self.max_output_tokens = int(max_output_tokens)
    self.timeout_sec = int(timeout_sec)

    self.session = requests.Session()
    self.api_key = self._resolve_api_key(self.provider)
    self.available = bool(self.api_key)
    self.unavailable_reason = ""
    if not self.available:
      if self.provider == "gemini":
        self.unavailable_reason = "Missing GOOGLE_API_KEY (or GEMINI_API_KEY)."
      elif self.provider == "openai":
        self.unavailable_reason = "Missing OPENAI_API_KEY."
      else:
        self.unavailable_reason = f"Unsupported provider '{self.provider}'."

  def _resolve_api_key(self, provider: str) -> str:
    if provider == "gemini":
      return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if provider == "openai":
      return os.getenv("OPENAI_API_KEY", "")
    return ""

  def _extract_json_str(self, text: str) -> str:
    raw = (text or "").strip()
    if not raw:
      raise ValueError("LLM returned empty content.")

    if raw.startswith("```"):
      lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
      raw = "\n".join(lines).strip()

    # Fast-path if full payload is already valid JSON.
    try:
      json.loads(raw)
      return raw
    except Exception:
      pass

    # Extract first likely JSON object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
      candidate = raw[start : end + 1]
      json.loads(candidate)
      return candidate

    raise ValueError("Could not extract JSON object from LLM response.")

  def _post_gemini(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = (
      f"https://generativelanguage.googleapis.com/v1beta/models/"
      f"{self.model}:generateContent?key={self.api_key}"
    )
    payload = {
      "systemInstruction": {
        "parts": [{"text": system_prompt}],
      },
      "contents": [
        {
          "role": "user",
          "parts": [{"text": user_prompt}],
        }
      ],
      "generationConfig": {
        "temperature": self.temperature,
        "maxOutputTokens": self.max_output_tokens,
        "responseMimeType": "application/json",
      },
    }
    resp = self.session.post(url, json=payload, timeout=self.timeout_sec)
    if resp.status_code >= 400:
      raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:260]}")
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
      raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()
    usage = data.get("usageMetadata", {})
    return {
      "text": text,
      "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
      "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
      "raw": data,
    }

  def _post_openai(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
      "Authorization": f"Bearer {self.api_key}",
      "Content-Type": "application/json",
    }
    payload = {
      "model": self.model,
      "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      "temperature": self.temperature,
      "max_tokens": self.max_output_tokens,
      "response_format": {"type": "json_object"},
    }
    resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
    if resp.status_code >= 400:
      raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:260]}")
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
      raise RuntimeError("OpenAI returned no choices.")
    text = choices[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return {
      "text": text,
      "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
      "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
      "raw": data,
    }

  def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    if not self.available:
      return {
        "ok": False,
        "error": self.unavailable_reason,
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    try:
      if self.provider == "gemini":
        api_result = self._post_gemini(system_prompt, user_prompt)
      elif self.provider == "openai":
        api_result = self._post_openai(system_prompt, user_prompt)
      else:
        raise RuntimeError(f"Unsupported provider '{self.provider}'.")

      text = api_result.get("text", "")
      json_str = self._extract_json_str(text)
      payload = json.loads(json_str)

      return {
        "ok": True,
        "payload": payload,
        "raw_text": text,
        "prompt_tokens": int(api_result.get("prompt_tokens", 0)),
        "completion_tokens": int(api_result.get("completion_tokens", 0)),
      }
    except Exception as e:
      return {
        "ok": False,
        "error": f"{type(e).__name__}: {str(e)}",
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }
