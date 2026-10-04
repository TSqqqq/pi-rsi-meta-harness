from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


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


def parse_json_from_agent_text(text: str) -> Any:
    """Accept strict JSON or one fenced JSON block; avoid fragile broad extraction."""
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.S | re.I)
    if m:
        return json.loads(m.group(1))
    raise ValueError("Agent response was not valid JSON")


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
