"""Hugging Face Hub integration.

Uses stdlib urllib only — no `requests` dependency.  Runs the fetch on a
QThread so the UI stays responsive.
"""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

HF_API_BASE = "https://huggingface.co/api/models"
HF_RAW_BASE = "https://huggingface.co"
TIMEOUT_SEC = 15

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def parse_model_id(url_or_id: str) -> str:
    """Extract ``org/repo`` from any HF URL form or bare model ID."""
    text = url_or_id.strip().rstrip("/")
    if not text:
        raise ValueError("Please enter a model URL or ID.")

    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)", text)
    if m:
        candidate = m.group(1)
        for token in ("/tree/", "/blob/", "/discussions", "/commit/"):
            if token in candidate:
                candidate = candidate.split(token)[0]
    elif "/" in text and text.count("/") == 1 and not text.startswith("http"):
        candidate = text
    else:
        raise ValueError("Invalid model URL — expected huggingface.co/<org>/<repo>.")

    if not _MODEL_ID_RE.match(candidate):
        raise ValueError(
            "Invalid model ID — only letters, digits, '.', '_', '-' allowed in "
            "<org>/<repo>.",
        )
    return candidate


def _encode_id(model_id: str) -> str:
    """URL-encode an already-validated model id for safe path concatenation."""
    return quote(model_id, safe="/")


def _get_json(url: str, token: str | None) -> dict:
    headers = {"User-Agent": "llm-calculator/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_gguf_repo(api_data: dict) -> bool:
    tags = api_data.get("tags", []) or []
    if "gguf" in tags:
        return True
    siblings = api_data.get("siblings") or []
    names = [s.get("rfilename", "") for s in siblings]
    return any(n.endswith(".gguf") for n in names) and not any(
        n == "config.json" for n in names
    )


def _base_model_hint(api_data: dict) -> str | None:
    card = api_data.get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, list) and base:
        return base[0]
    if isinstance(base, str):
        return base
    for tag in api_data.get("tags", []) or []:
        if tag.startswith("base_model:") and not tag.startswith("base_model:adapter:"):
            return tag.split(":", 1)[1]
    return None


_KNOWN_GGUF_QUANTS = [
    "Q2_K",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
    "F16", "FP16", "BF16", "F32", "FP32",
]


def gguf_quants_in_repo(api_data: dict) -> list[str]:
    """Return the quantization labels present as .gguf files in the repo.

    Normalizes half-precision labels (F16/BF16/FP16) to "FP16" to match the
    calculator's BPP table. Skips mmproj/projector files.
    """
    found: list[str] = []
    siblings = api_data.get("siblings") or []
    for s in siblings:
        raw = s.get("rfilename") or ""
        if not raw.lower().endswith(".gguf"):
            continue
        if "mmproj" in raw.lower() or "projector" in raw.lower():
            continue
        upper = raw.upper()
        for q in _KNOWN_GGUF_QUANTS:
            if f".{q}." in upper or f"-{q}." in upper or f"_{q}." in upper:
                normalized = "FP16" if q in ("F16", "FP16", "BF16", "F32", "FP32") else q
                if normalized not in found:
                    found.append(normalized)
                break
    return found


class ModelFetcher(QThread):
    """Fetches HF model metadata + config.json off the UI thread."""

    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, url_or_id: str, token: str | None) -> None:
        super().__init__()
        self._url_or_id = url_or_id
        self._token = token or None

    def run(self) -> None:
        try:
            model_id = parse_model_id(self._url_or_id)
        except ValueError as e:
            self.failed.emit(str(e))
            return

        encoded = _encode_id(model_id)
        try:
            api_data = _get_json(f"{HF_API_BASE}/{encoded}", self._token)
        except HTTPError as e:
            self.failed.emit(self._http_message(e, "model metadata"))
            return
        except URLError as e:
            self.failed.emit(f"Network error: {e.reason}")
            return

        base_model_id: str | None = None
        # Capture GGUF files in the original repo before we may swap api_data
        # for the base model's metadata.
        gguf_quants = gguf_quants_in_repo(api_data)

        try:
            config = _get_json(
                f"{HF_RAW_BASE}/{encoded}/raw/main/config.json", self._token,
            )
        except HTTPError as e:
            if e.code == 404 and _is_gguf_repo(api_data):
                hint = _base_model_hint(api_data)
                if not hint or not _MODEL_ID_RE.match(hint):
                    self.failed.emit(
                        "GGUF-only repository with no usable base_model "
                        "reference. Paste the base model URL directly.",
                    )
                    return
                hint_encoded = _encode_id(hint)
                try:
                    base_api = _get_json(f"{HF_API_BASE}/{hint_encoded}", self._token)
                    config = _get_json(
                        f"{HF_RAW_BASE}/{hint_encoded}/raw/main/config.json",
                        self._token,
                    )
                except HTTPError as sub_e:
                    self.failed.emit(
                        f"GGUF repo references base model '{hint}', but "
                        f"{self._http_message(sub_e, 'its config')}",
                    )
                    return
                except URLError as sub_e:
                    self.failed.emit(
                        f"Network error fetching base model '{hint}': {sub_e.reason}",
                    )
                    return
                base_model_id = hint
                # Swap to base model's safetensors for accurate param count.
                api_data = base_api
            else:
                self.failed.emit(self._http_message(e, "config.json"))
                return
        except URLError as e:
            self.failed.emit(f"Network error: {e.reason}")
            return

        # If the config.json was present inline in a GGUF repo, but safetensors
        # metadata isn't (the repo only ships .gguf weights), try the base model
        # referenced by the config or tags for an accurate param count.
        if gguf_quants and not (api_data.get("safetensors") or {}).get("total"):
            hint = _base_model_hint(api_data)
            if hint and hint != model_id and _MODEL_ID_RE.match(hint):
                try:
                    base_api = _get_json(
                        f"{HF_API_BASE}/{_encode_id(hint)}", self._token,
                    )
                    if (base_api.get("safetensors") or {}).get("total"):
                        api_data = base_api
                        base_model_id = hint
                except (HTTPError, URLError):
                    pass  # Fall through to dimension-based param estimate.

        self.finished_ok.emit({
            "model_id": model_id,
            "base_model_id": base_model_id,
            "api": api_data,
            "config": config,
            "gguf_quants": gguf_quants,
        })

    def _http_message(self, e: HTTPError, what: str) -> str:
        if e.code == 401:
            return (
                f"Gated model — {what} requires authentication. "
                "Paste an HF token in the token field and retry."
            )
        if e.code == 403:
            return (
                f"Forbidden — your token may lack access to this model's {what}."
            )
        if e.code == 404:
            return f"Not found — {what} doesn't exist for this repository."
        return f"HTTP {e.code} while fetching {what}: {e.reason}"
