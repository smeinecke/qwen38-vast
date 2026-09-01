"""HTTP/HTTPS client for the local llama-server API (via SSH tunnel)."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
import urllib3

from hostai.config import Config
from hostai.state import State

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _parse_prom(text: Optional[str]) -> Dict[str, float]:
    """Parse llama-server /metrics Prometheus text into a simple dict.

    Handles the common llama.cpp metrics format plus the occasional JSON-wrapped
    string that some revisions return.
    """
    if not text:
        return {}

    stripped = text.strip()
    if stripped.startswith('"'):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, str):
                text = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            out[name] = float(parts[-1])
        except ValueError:
            pass
    return out


class LlamaClient:
    """Minimal requests-based client for a llama-server behind an SSH tunnel."""

    def __init__(self, config: Config, state: State) -> None:
        self.config = config
        self.state = state
        self._api_key = state.api_key or ""

        scheme = "http" if state.unsecure else "https"
        self.base_url = f"{scheme}://127.0.0.1:{state.local_port}"

        if state.unsecure or not state.tls_ca or not state.tls_ca.exists():
            self._verify: Union[bool, str] = False
        else:
            self._verify = str(state.tls_ca)

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> bool:
        """Return True when /health reports ok."""
        try:
            response = requests.get(
                self._url("/health"),
                headers=self._headers,
                verify=self._verify,
                timeout=(2, 5),
            )
        except requests.RequestException:
            return False

        if response.status_code != 200:
            return False

        text = response.text.strip()
        if text.lower() == "ok":
            return True

        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "ok":
                return True
        except (json.JSONDecodeError, ValueError):
            pass

        return "ok" in text.lower()

    def wait_for_health(self, timeout: float = 1200, quiet: bool = False) -> bool:
        """Poll health with 1s initial interval and exponential backoff up to 5s."""
        if timeout <= 0:
            return self.health()

        start = time.monotonic()
        interval = 1.0
        last_log = start

        while True:
            if self.health():
                return True

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return False

            if not quiet:
                now = time.monotonic()
                if now - last_log >= 15:
                    last_log = now
                    print(
                        f"[api] waiting for llama-server health ({int(elapsed)}s / {int(timeout)}s)",
                        flush=True,
                    )

            time.sleep(interval)
            interval = min(interval * 2, 5.0)

    def get_metrics_text(self) -> Optional[str]:
        """GET /metrics and return the raw Prometheus text."""
        try:
            response = requests.get(
                self._url("/metrics"),
                headers=self._headers,
                verify=self._verify,
                timeout=(2, 10),
            )
        except requests.RequestException:
            return None

        if response.status_code != 200:
            return None

        return response.text

    def get_metrics(self) -> Dict[str, float]:
        """GET /metrics and parse it into {metric_name: value}."""
        text = self.get_metrics_text()
        if text is None:
            return {}
        return _parse_prom(text)

    def slots(self) -> List[Dict[str, Any]]:
        """GET /slots. Returns a list of slot objects or [] on failure."""
        try:
            response = requests.get(
                self._url("/slots"),
                headers=self._headers,
                verify=self._verify,
                timeout=(2, 10),
            )
        except requests.RequestException:
            return []

        if response.status_code != 200:
            return []

        try:
            payload = response.json()
            if isinstance(payload, list):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def slot_save(self, slot: int) -> bool:
        """POST /slots/<slot>?action=save."""
        return self._slot_action(slot, "save")

    def slot_restore(self, slot: int) -> bool:
        """POST /slots/<slot>?action=restore."""
        return self._slot_action(slot, "restore")

    def _slot_action(self, slot: int, action: str) -> bool:
        try:
            response = requests.post(
                self._url(f"/slots/{slot}?action={action}"),
                headers=self._headers,
                json={"filename": "current.bin"},
                verify=self._verify,
                timeout=(5, 1800),
            )
        except requests.RequestException:
            return False

        # llama.cpp may return 200 or 202 for accepted async actions.
        return 200 <= response.status_code < 300

    def _stream(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Yield parsed Server-Sent Events from a streaming chat response."""
        for raw in response.iter_lines(decode_unicode=True):
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line:
                continue
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue

            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue

            if isinstance(obj, dict):
                yield obj

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        timeout: float = 900,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """POST /v1/chat/completions. Returns a dict or a streaming generator."""
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

        payload: Dict[str, Any] = {
            "model": self.config.model.model or "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        try:
            response = requests.post(
                self._url("/v1/chat/completions"),
                headers=self._headers,
                json=payload,
                stream=stream,
                verify=self._verify,
                timeout=(5, timeout),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise requests.RequestException(f"chat request failed: {exc}") from exc

        if stream:
            return self._stream(response)

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise requests.RequestException(f"invalid JSON in chat response: {exc}") from exc


def is_api_ready(config: Config, state: State) -> bool:
    """Standalone one-shot health check."""
    return LlamaClient(config, state).health()


def wait_for_api(config: Config, state: State, timeout: float) -> bool:
    """Standalone blocking wait for llama-server /health."""
    return LlamaClient(config, state).wait_for_health(timeout, quiet=False)
