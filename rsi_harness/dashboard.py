from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Config
from .db import ResearchDB
from .util import tail_lines


class DashboardServer:
    """Zero-dependency research console.

    The dashboard is intentionally a thin control/observability surface over the
    durable SQLite state. It never owns the research lifecycle: POST actions are
    recorded in SQLite and the controller consumes them asynchronously.
    """

    def __init__(self, cfg: Config, db: ResearchDB, html_path: Path):
        self.cfg = cfg
        self.db = db
        self.html_path = html_path
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def _audit(self, type_: str, message: str, payload: dict | None = None, level: str = "INFO") -> None:
        self.db.execute(
            "INSERT INTO events(ts,level,type,source,message,payload_json) VALUES(?,?,?,?,?,?)",
            (time.time(), level, type_, "DASHBOARD", message, json.dumps(payload or {}, ensure_ascii=False)),
        )

    def snapshot(self) -> dict:
        exps = self.db.query(
            "SELECT id,created_at,updated_at,status,stage,title,parent_primary,metric,metric_std,delta_parent,delta_baseline,"
            "confidence,novelty,gpu_hours,pinned,validated FROM experiments ORDER BY created_at ASC"
        )
        edges = self.db.query("SELECT parent_id,child_id,edge_type FROM edges")
        events = self.db.query("SELECT id,ts,level,type,source,message FROM events ORDER BY id DESC LIMIT 240")
        events.reverse()
        best = self.db.best(self.cfg.direction)
        baseline = self.db.experiment("baseline")
        sums = self.db.one(
            "SELECT COALESCE(SUM(gpu_hours),0) AS gpu_hours, COUNT(*) AS experiments "
            "FROM experiments WHERE id!='baseline'"
        ) or {}
        toks = self.db.one(
            "SELECT COALESCE(SUM(total_tokens),0) AS tokens, COALESCE(SUM(cost),0) AS cost FROM agent_runs"
        ) or {}
        active_agents = self.db.query(
            "SELECT id,ts_start,agent_name,task_type,session_id,provider,model_id,thinking,experiment_id "
            "FROM agent_runs WHERE ts_end IS NULL ORDER BY ts_start"
        )
        agent_runs = self.db.query(
            "SELECT id,ts_start,ts_end,agent_name,task_type,experiment_id,provider,model_id,total_tokens,tool_calls,success "
            "FROM agent_runs ORDER BY id DESC LIMIT 200"
        )
        active_experiments = self.db.query(
            "SELECT id,title,stage,status,updated_at,metric,delta_parent FROM experiments "
            "WHERE status IN ('running','implemented','planned','repairing') ORDER BY updated_at DESC"
        )
        model_usage = self.db.query(
            "SELECT provider,model_id,COUNT(*) AS runs,COALESCE(SUM(total_tokens),0) AS tokens,"
            "COALESCE(SUM(cost),0) AS cost FROM agent_runs GROUP BY provider,model_id ORDER BY tokens DESC"
        )
        status_counts = self.db.query(
            "SELECT status,COUNT(*) AS n FROM experiments WHERE id!='baseline' GROUP BY status ORDER BY n DESC"
        )
        guidance = self.db.recent_guidance(20)
        deps = self.db.query("SELECT * FROM dependency_requests WHERE status='requested' ORDER BY ts DESC")
        metric_history = self.db.query(
            "SELECT id,created_at,stage,status,metric,delta_baseline,validated FROM experiments "
            "WHERE metric IS NOT NULL ORDER BY created_at"
        )
        note = self.db.get_note()
        ideas = self.db.list_ideas(limit=100)
        return {
            "state": self.db.get_meta("state", "NEW"),
            "pause_reason": self.db.get_meta("pause_reason", ""),
            "objective": {
                "metric": self.cfg.get("objective", "metric"),
                "direction": self.cfg.direction,
                "target": self.cfg.target,
                "min_meaningful_delta": self.cfg.get("objective", "min_meaningful_delta", 0.0),
            },
            "baseline": baseline,
            "best": best,
            "experiments": exps,
            "edges": edges,
            "events": events,
            "metric_history": metric_history,
            "gpu_hours": sums.get("gpu_hours", 0),
            "experiment_count": sums.get("experiments", 0),
            "agent_tokens": toks.get("tokens", 0),
            "agent_cost": toks.get("cost", 0),
            "active_agents": active_agents,
            "agent_runs": agent_runs,
            "active_experiments": active_experiments,
            "model_usage": model_usage,
            "status_counts": status_counts,
            "guidance": guidance,
            "dependency_requests": deps,
            "active_policy": self.db.get_meta("active_policy", {}),
            "plateau_report": self.db.get_meta("plateau_report", {}),
            "human_note": note,
            "idea_queue": ideas,
            "started_at": self.db.get_meta("started_at", None),
            "budgets": {
                "max_gpu_hours": self.cfg.get("budget", "max_gpu_hours", None),
                "max_agent_tokens": self.cfg.get("budget", "max_agent_tokens", None),
                "max_experiments": self.cfg.get("budget", "max_experiments", None),
                "max_wall_hours": self.cfg.get("budget", "max_wall_hours", None),
            },
            "models": {
                "mode": self.cfg.get("models", "mode", "single"),
                "default": self.cfg.get("models", "default", ""),
                "planner": self.cfg.get("models", "planner", ""),
                "worker": self.cfg.get("models", "worker", ""),
            },
        }

    def experiment_detail(self, eid: str, include_artifacts: bool = False) -> dict:
        exp = self.db.experiment(eid)
        if not exp:
            raise KeyError(eid)
        parents = [self.db.experiment(x) for x in self.db.parents(eid)]
        children = [self.db.experiment(x) for x in self.db.children(eid)]
        metrics = self.db.query(
            "SELECT id,stage,seed,metric,extra_json,created_at FROM metrics WHERE experiment_id=? ORDER BY id", (eid,)
        )
        effects = self.db.query(
            "SELECT parameter,from_value,to_value,effect,confidence,stage,created_at FROM parameter_effects "
            "WHERE experiment_id=? ORDER BY id", (eid,)
        )
        agent_runs = self.db.query(
            "SELECT id,ts_start,ts_end,agent_name,task_type,provider,model_id,thinking,total_tokens,tool_calls,cost,success,error "
            "FROM agent_runs WHERE experiment_id=? OR session_id=? OR agent_name=? ORDER BY ts_start",
            (eid, f"rsi-{eid}", f"worker-{eid}"),
        )
        art_dir = self.cfg.artifact_dir / eid
        artifacts = []
        patch_preview = ""
        worker_summary = ""
        if art_dir.exists():
            for path in sorted(art_dir.iterdir()):
                if path.is_file():
                    artifacts.append({"name": path.name, "size": path.stat().st_size})
            if include_artifacts:
                patch = art_dir / "patch.diff"
                if patch.exists():
                    patch_preview = patch.read_text(encoding="utf-8", errors="replace")[:30000]
                summary = art_dir / "worker_summary.txt"
                if summary.exists():
                    worker_summary = summary.read_text(encoding="utf-8", errors="replace")[:30000]
        log_dir = self.cfg.log_dir / "experiments"
        logs = []
        if log_dir.exists():
            for path in sorted(log_dir.glob(f"{eid}*"), key=lambda x: x.stat().st_mtime, reverse=True):
                if path.is_file():
                    logs.append({"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime})
        return {
            "experiment": exp,
            "parents": [x for x in parents if x],
            "children": [x for x in children if x],
            "metrics": metrics,
            "parameter_effects": effects,
            "agent_runs": agent_runs,
            "artifacts": artifacts,
            "logs": logs,
            "patch_preview": patch_preview,
            "worker_summary": worker_summary,
        }

    def agent_transcript(self, run_id: int) -> dict:
        """Full conversation of one Pi call: every message, tool call/result and harness abort, in order."""
        run = self.db.one("SELECT * FROM agent_runs WHERE id=?", (run_id,))
        if not run:
            raise KeyError(run_id)
        path = Path(run.get("transcript_path") or "")
        records = []
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("run_id") == run_id:
                        records.append(rec)
        return {"run": run, "records": records, "transcript_path": str(path), "available": path.is_file()}

    def experiment_log(self, eid: str, lines: int = 240) -> dict:
        lines = min(max(int(lines), 20), 1000)
        log_dir = self.cfg.log_dir / "experiments"
        if not log_dir.exists():
            return {"experiment_id": eid, "files": [], "text": ""}
        paths = [p for p in log_dir.glob(f"{eid}*") if p.is_file()]
        paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        if not paths:
            return {"experiment_id": eid, "files": [], "text": ""}
        # Prefer a currently active train log, otherwise newest file.
        preferred = next((p for p in paths if ".train.log" in p.name), paths[0])
        return {
            "experiment_id": eid,
            "files": [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in paths],
            "selected": preferred.name,
            "text": tail_lines(preferred, lines),
        }

    def _handle_post(self, path: str, data: dict) -> tuple[int, dict]:
        if not bool(self.cfg.get("dashboard", "allow_control", True)):
            return HTTPStatus.FORBIDDEN, {"ok": False, "error": "dashboard control disabled"}

        if path == "/api/control":
            action = str(data.get("action", "")).strip().lower()
            if action not in {"pause", "resume", "stop"}:
                return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid action"}
            if action == "resume" and self.db.get_meta("state", "") == "PAUSED_FOR_HUMAN":
                self.db.set_meta("plateau_ack_count", self.db.completed_count())
            self.db.set_meta("control", {"action": action, "ts": time.time(), "source": "dashboard"})
            self._audit("human_control", f"{action} requested from dashboard", {"action": action})
            return HTTPStatus.OK, {"ok": True, "action": action}

        if path == "/api/node":
            eid = str(data.get("experiment_id", "")).strip()
            action = str(data.get("action", "")).strip().lower()
            exp = self.db.experiment(eid)
            if not exp:
                return HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown experiment"}
            if action in {"pin", "unpin"}:
                self.db.update_experiment(eid, pinned=1 if action == "pin" else 0)
                self._audit("human_node", f"{action} {eid}", {"experiment_id": eid})
                return HTTPStatus.OK, {"ok": True}
            if action == "branch":
                insight = str(data.get("insight", "")).strip()
                self.db.update_experiment(eid, pinned=1)
                if insight:
                    self.db.add_guidance(insight, kind="branch", experiment_id=eid)
                self._audit("human_branch", f"branch requested from {eid}", {"experiment_id": eid, "insight": insight})
                return HTTPStatus.OK, {"ok": True}
            return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid node action"}

        if path == "/api/guidance":
            text = str(data.get("text", "")).strip()
            kind = str(data.get("kind", "idea")).strip().lower()
            target = data.get("target")
            if not text:
                return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty guidance"}
            if kind == "steer":
                if not target:
                    return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "steer requires target agent"}
                self.db.add_guidance(text, kind="steer", experiment_id=str(target))
            elif kind in {"idea", "note", "branch"}:
                self.db.add_guidance(text, kind=kind, experiment_id=str(target) if target else None)
            else:
                return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "unsupported guidance kind"}
            self._audit("human_guidance", f"{kind} queued", {"kind": kind, "target": target})
            return HTTPStatus.OK, {"ok": True}

        if path == "/api/notes":
            body = str(data.get("body", ""))
            self.db.save_note(body)
            return HTTPStatus.OK, {"ok": True, "updated_at": time.time()}

        if path == "/api/ideas":
            action = str(data.get("action", "")).strip().lower()
            if action == "add":
                text = str(data.get("text", "")).strip()
                if not text:
                    return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty idea"}
                eid = str(data.get("experiment_id") or "").strip() or None  # JSON null must not become the string "None"
                if eid and not self.db.experiment(eid):
                    return HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown experiment"}
                iid = self.db.add_idea(text, experiment_id=eid)
                self._audit("idea_added", f"idea #{iid} added", {"idea_id": iid, "experiment_id": eid})
                return HTTPStatus.OK, {"ok": True, "id": iid}
            if action == "send":
                iid = int(data.get("id", 0) or 0)
                ok = self.db.send_idea(iid)
                if ok:
                    self._audit("idea_sent", f"idea #{iid} queued for next research turn", {"idea_id": iid})
                return HTTPStatus.OK, {"ok": ok}
            if action == "send_all":
                ids = data.get("ids")
                if not isinstance(ids, list):
                    ids = [x["id"] for x in self.db.list_ideas("draft", 100)]
                sent = 0
                for x in ids:
                    try:
                        sent += 1 if self.db.send_idea(int(x)) else 0
                    except (TypeError, ValueError):
                        continue
                self._audit("ideas_sent", f"{sent} ideas queued for next research turn", {"count": sent})
                return HTTPStatus.OK, {"ok": True, "sent": sent}
            if action == "delete":
                iid = int(data.get("id", 0) or 0)
                ok = self.db.delete_idea(iid)
                return HTTPStatus.OK, {"ok": ok}
            return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid idea action"}

        if path == "/api/dependency":
            did = int(data.get("id", 0) or 0)
            action = str(data.get("action", "")).strip().lower()
            if action not in {"approve", "reject"}:
                return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid action"}
            row = self.db.one("SELECT * FROM dependency_requests WHERE id=? AND status='requested'", (did,))
            if not row:
                return HTTPStatus.NOT_FOUND, {"ok": False, "error": "request not found"}
            status = "approved" if action == "approve" else "rejected"
            note = str(data.get("note", "")).strip() or f"{status} from dashboard"
            self.db.execute(
                "UPDATE dependency_requests SET status=?,resolved_at=?,note=? WHERE id=?",
                (status, time.time(), note, did),
            )
            self._audit("dependency_decision", f"{row['package']} {status}", {"id": did, "package": row["package"]})
            return HTTPStatus.OK, {"ok": True, "status": status}

        return HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}

    def start(self) -> None:
        if self.server:
            return
        host = str(self.cfg.get("dashboard", "host", "127.0.0.1"))
        port = int(self.cfg.get("dashboard", "port", 8765))
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, obj: dict) -> None:
                data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                if path in {"/", "/index.html"}:
                    data = parent.html_path.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Frame-Options", "DENY")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path == "/api/state":
                    self._send_json(HTTPStatus.OK, parent.snapshot())
                    return
                if path == "/api/experiment":
                    eid = (qs.get("id") or [""])[0]
                    try:
                        full = ((qs.get("full") or ["0"])[0].lower() in {"1", "true", "yes"})
                        self._send_json(HTTPStatus.OK, parent.experiment_detail(eid, include_artifacts=full))
                    except KeyError:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown experiment"})
                    return
                if path == "/api/transcript":
                    try:
                        self._send_json(HTTPStatus.OK, parent.agent_transcript(int((qs.get("run_id") or ["0"])[0])))
                    except (KeyError, ValueError):
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown agent run"})
                    return
                if path == "/api/log":
                    eid = (qs.get("experiment_id") or [""])[0]
                    try:
                        lines = int((qs.get("lines") or ["240"])[0])
                    except ValueError:
                        lines = 240
                    self._send_json(HTTPStatus.OK, parent.experiment_log(eid, lines))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self):
                parsed = urlparse(self.path)
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                except ValueError:
                    length = 0
                if length < 0 or length > 1_000_000:
                    self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "request too large"})
                    return
                if "application/json" not in (self.headers.get("Content-Type") or ""):
                    self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"ok": False, "error": "JSON required"})
                    return
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("object required")
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON: {exc}"})
                    return
                status, out = parent._handle_post(parsed.path, data)
                self._send_json(status, out)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="rsi-dashboard", daemon=True)
        self.thread.start()

    @property
    def address(self) -> tuple[str, int] | None:
        if not self.server:
            return None
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
