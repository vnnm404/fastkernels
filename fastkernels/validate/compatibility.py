"""Cheap reference/config checks; never initialize an engine or download weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def inherit_hf_auth():
    # Resolve before the runner redirects HF_HOME into a disposable row cache.
    # Environment credentials propagate to Ray and both reference workers.
    from huggingface_hub import get_token

    token = get_token()
    if token:
        os.environ["HF_TOKEN"] = token


def hub_retry(call, attempts=4):
    """Retry transient Hub failures only; access denials need user action."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            connection_error = isinstance(exc, (TimeoutError, ConnectionError))
            try:
                import httpx

                connection_error |= isinstance(
                    exc,
                    (
                        httpx.TimeoutException,
                        httpx.NetworkError,
                        httpx.RemoteProtocolError,
                    ),
                )
            except ImportError:
                pass
            try:
                import requests

                connection_error |= isinstance(
                    exc,
                    (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
                )
            except ImportError:
                pass
            if (
                status not in (429, 500, 502, 503, 504) and not connection_error
            ) or attempt == attempts - 1:
                raise
            try:
                delay = float(
                    getattr(response, "headers", {}).get(
                        "Retry-After", 2 ** (attempt + 1)
                    )
                )
            except (TypeError, ValueError):
                delay = 2 ** (attempt + 1)
            time.sleep(max(0, min(delay, 180)))


def access_problem(exc):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return {
            "status": "blocked",
            "reason": f"Hugging Face HTTP {status}: verify HF_TOKEN/HF_TOKEN_PATH and "
            "request/accept gated-model access with the token owner account.",
        }
    return None


def probe(config_dir):
    """Called in each installed environment. Support is necessary, not sufficient."""
    import importlib
    import importlib.metadata

    result = {"status": "ok", "packages": {}}
    config = json.loads((Path(config_dir) / "config.json").read_text())
    dependencies = [
        "vllm",
        "transformers",
        "torch",
        "datasets",
        "pyarrow",
        "fastsafetensors",
    ]
    if config.get("model_type") == "whisper" or any(
        k in config for k in ("vision_config", "audio_config", "thinker_config")
    ):
        dependencies.append("av")
    for name in dependencies:
        try:
            importlib.import_module(name)
            result["packages"][name] = importlib.metadata.version(name)
        except Exception as exc:
            return dict(
                result,
                status="environment-error",
                reason=f"{name} import failed: {type(exc).__name__}: {exc}",
            )
    import torch

    try:
        if not torch.zeros(1, pin_memory=True).is_pinned():
            raise RuntimeError("allocation is not pinned")
    except Exception as exc:
        return dict(
            result,
            status="environment-error",
            reason=f"Pinned host memory unavailable: {exc}",
        )
    from vllm.model_executor.models import ModelRegistry

    architectures = config.get("architectures", [])
    supported = ModelRegistry.get_supported_archs()
    native = any(a in supported for a in architectures)
    try:
        from vllm.transformers_utils.config import get_config

        get_config(str(config_dir), trust_remote_code=False, config_format="hf")
    except Exception as exc:
        if architectures and not native:
            return dict(
                result,
                status="unsupported",
                reason=f"vLLM {result['packages']['vllm']} has no native implementation for {architectures}, "
                f"and its installed config loader failed: {type(exc).__name__}: {exc}",
            )
        return dict(
            result,
            status="environment-error",
            reason=f"Config cannot load in this reference: {type(exc).__name__}: {exc}",
        )
    if architectures and not native:
        # Do not preempt vLLM's normal auto fallback when the pinned config
        # loader supports the model. The actual validate run must establish fit.
        result["warning"] = (
            f"No native registry entry for {architectures}; vLLM auto backend must resolve this model at initialization."
        )
    return result


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(probe(sys.argv[1]), indent=2))
