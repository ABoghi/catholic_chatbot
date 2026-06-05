import json
import os
import subprocess
from pathlib import Path

import requests
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from langchain.llms import Ollama as LangChainOllama
    from langchain.llms import HuggingFacePipeline as LangChainHuggingFacePipeline
except Exception:
    LangChainOllama = None
    LangChainHuggingFacePipeline = None

from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_PORT


class LangChainOllamaClient:
    def __init__(self, host=OLLAMA_HOST, port=OLLAMA_PORT, model=OLLAMA_MODEL):
        if LangChainOllama is None:
            raise RuntimeError("LangChain is not installed. Install `langchain` to use Ollama through LangChain.")
        self.client = LangChainOllama(model=model, base_url=f"http://{host}:{port}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        return self.client(prompt)


class LangChainLocalTransformersLLM:
    def __init__(self, model: str = "gpt2"):
        if pipeline is None or LangChainHuggingFacePipeline is None:
            raise RuntimeError("Install `langchain`, `transformers`, and `torch` to use a local transformers LLM through LangChain.")
        hf_pipeline = pipeline("text-generation", model=model, device_map="auto")
        self.client = LangChainHuggingFacePipeline(pipeline=hf_pipeline)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return self.client(prompt)


class OllamaClient:
    def __init__(self, host=OLLAMA_HOST, port=OLLAMA_PORT, model=OLLAMA_MODEL):
        self.host = host
        self.port = port
        self.model = model
        self.api_url = f"http://{self.host}:{self.port}/completions"

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "text" in data:
                return data["text"].strip()
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                return choice.get("text", choice.get("message", "")).strip()
            return json.dumps(data, indent=2)
        except Exception as exc:
            return self._fallback_cli(prompt, exc)

    def _fallback_cli(self, prompt: str, previous_exception: Exception) -> str:
        try:
            completed = subprocess.run(
                ["ollama", "chat", self.model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=120,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
            raise RuntimeError(
                f"Ollama CLI error: {completed.returncode}\n{completed.stderr}"
            )
        except Exception as cli_exc:
            raise RuntimeError(
                f"Ollama API call failed: {previous_exception}\nOllama CLI fallback failed: {cli_exc}"
            )


class LocalTransformersLLM:
    """Simple local LLM client using HuggingFace `transformers` pipeline.

    This is best-effort: large models will require sufficient GPU/CPU and disk space.
    """

    def __init__(self, model: str = "gpt2"):
        if pipeline is None:
            raise RuntimeError("transformers pipeline is not available. Install `transformers` and `torch`.")
        self.model = model
        self._pipe = pipeline("text-generation", model=self.model, device_map="auto")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        # Truncate prompt to fit within model's max length (approx 800 tokens for GPT2's 1024 limit with generation headroom)
        tokenizer = self._pipe.tokenizer
        max_input_tokens = 512
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        if len(encoded) > max_input_tokens:
            # Truncate to fit and re-decode
            encoded = encoded[:max_input_tokens]
            prompt = tokenizer.decode(encoded, skip_special_tokens=False)
        
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "truncation": True,
            "max_length": max_input_tokens + max_tokens,
        }
        try:
            out = self._pipe(prompt, **params)
            if out and isinstance(out, list):
                generated_text = out[0].get("generated_text", "").strip()
                # Remove input prompt from output if it's repeated at the start
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
        except Exception as e:
            return f"Error generating response: {str(e)}"
        return ""


def create_llm_client(provider: str | None, model: str | None):
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    model = model or os.getenv("OLLAMA_MODEL") or OLLAMA_MODEL
    if provider in {"ollama", "ola", "oll"}:
        if LangChainOllama is not None:
            return LangChainOllamaClient(model=model)
        return OllamaClient(model=model)
    if provider in {"local", "transformers", "hf"}:
        if LangChainHuggingFacePipeline is not None:
            return LangChainLocalTransformersLLM(model=model)
        return LocalTransformersLLM(model=model)
    if LangChainOllama is not None:
        return LangChainOllamaClient(model=model)
    return OllamaClient(model=model)
