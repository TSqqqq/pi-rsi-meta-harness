from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def now_ts() -> float:
    return time.time()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(base: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def run_capture(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=check)


def split_model(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if not spec or "/" not in spec:
        raise ValueError(f"Expected model as provider/model, got: {spec!r}")
    return tuple(spec.split("/", 1))  # type: ignore[return-value]


def parse_last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in command output")


def _json_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S | re.I)
    sources = blocks + [cleaned]
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in starts:
        depth = 0
        quoted = False
        escaped = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if quoted:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = False
                continue
            if ch == '"':
                quoted = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    sources.append(cleaned[start:i + 1])
                    break
    return sources


def parse_json_from_agent_text(text: str, required_key: str | None = None) -> Any:
    """Parse strict, fenced, think-wrapped, or lightly malformed JSON from an agent."""
    errors: list[str] = []
    for source in _json_candidates(text):
        candidate = re.sub(r",\s*([}\]])", r"\1", source.strip())
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if required_key is None or (isinstance(obj, dict) and required_key in obj):
            return obj
    suffix = f" containing {required_key!r}" if required_key else ""
    raise ValueError(f"Agent response was not valid JSON{suffix}: {errors[-1] if errors else 'no object found'}")


def shell_template(template: str, **values: Any) -> str:
    safe = {k: str(v) for k, v in values.items()}
    return template.format(**safe)


def tail_lines(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def atomic_write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def batched(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i:i+n]
