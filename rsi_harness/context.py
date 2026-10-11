from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from .config import Config
from .db import ResearchDB

KINDS = ("mechanism", "combination", "revival", "tuning")


def clip(text: Any, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def section(title: str, lines: list[str], budget: int) -> str:
    """A titled block cut on line boundaries, so a budget never truncates mid-record."""
    if not lines:
        return ""
    out, used = [], len(title) + 2
    for i, line in enumerate(lines):
        if used + len(line) + 1 > budget:
            out.append(f"… (+{len(lines) - i} more omitted)")
            break
        out.append(line)
        used += len(line) + 1
    return f"## {title}\n" + "\n".join(out)


def outcome(row: dict[str, Any]) -> str:
    """Separate 'the idea was measured' from 'the idea was never tested', and say why."""
    if row.get("metric") is not None:
        return str(row["status"])
    reason = str(row.get("failure_reason") or "")
    for needle, cause in (("timeout", "worker_timeout"), ("tool call limit", "tool_limit"), ("numeric-only", "numeric_only_patch"),
                          ("outside mutable", "path_violation"), ("immutable", "path_violation"),
                          ("preflight", "runtime_error"), ("smoke", "runtime_error"), ("failed: stage", "runtime_error")):
        if needle in reason:
            return f"NOT_RUN:{cause}"
    return f"NOT_RUN:{row['status']}"


class ContextBuilder:
    """Turns the SQLite DAG into compact, budgeted plain-text briefings for each agent role.

    Long-term memory is the database; every agent call starts a fresh session and receives only a briefing.
    """

    def __init__(self, cfg: Config, db: ResearchDB, noise=None):
        self.cfg = cfg
        self.db = db
        self.noise = noise  # callable(stage, n) -> threshold; injected by the runner

    # ---------- raw rows ----------
    def _rows(self) -> list[dict[str, Any]]:
        # 'interrupted' rows are harness restarts, not evidence.
        return self.db.query("SELECT * FROM experiments WHERE id!='baseline' AND status!='interrupted' ORDER BY created_at")

    def _n(self, row: dict[str, Any]) -> int:
        r = self.db.one("SELECT COUNT(*) AS n FROM metrics WHERE experiment_id=? AND stage=?", (row["id"], row.get("stage")))
        return int(r["n"]) if r else 0

    def _knobs(self, row: dict[str, Any]) -> str:
        defaults = self.cfg.section("experiment")
        params = json.loads(row.get("params_json") or "{}")
        return ",".join(f"{k}={v}" for k, v in params.items() if defaults.get(k) != v)

    def line(self, row: dict[str, Any], lesson: int = 130) -> str:
        """One experiment = one line: the unit of the research ledger."""
        delta = row.get("delta_baseline")
        measured = f"Δ{delta:+.3f}(n={self._n(row)})" if delta is not None else ""
        knobs = self._knobs(row)
        head = f"{row['id']}<{row.get('parent_primary') or '-'} [{row.get('kind') or '?'}] {outcome(row)} {measured}"
        body = clip(row.get("title"), 80) + (f" {{{knobs}}}" if knobs else "")
        tail = "" if row.get("metric") is None else " | " + clip(row.get("lesson"), lesson)
        return f"{head.strip()} | {body}{tail}"

    # ---------- sections ----------
    def objective(self) -> list[str]:
        mean, sigma, n = self.db.metric_stats("baseline", "quick")
        out = [f"metric={self.cfg.get('objective', 'metric')} direction={self.cfg.direction} target={self.cfg.target}"]
        if mean is not None:
            out.append(f"quick baseline mean={mean:.4f} sigma={sigma:.4f} over {n} identical runs")
        if self.noise:
            out.append(f"noise threshold |Δ| for a single run={self.noise('quick', 1):.4f}; for a 3-run mean={self.noise('quick', 3):.4f}")
        out.append("Δ inside the threshold is noise: neither evidence for nor against a mechanism.")
        return out

    def _side(self, eid: str, stage: str | None) -> str:
        """Mean of the configured secondary metrics (e.g. per-question-type accuracies), slash-separated."""
        keys = self.cfg.get("objective", "side_metrics", []) or []
        vals: dict[str, list[float]] = defaultdict(list)
        for r in self.db.query("SELECT extra_json FROM metrics WHERE experiment_id=? AND stage=?", (eid, stage)):
            extra = json.loads(r["extra_json"] or "{}")
            for k in keys:
                if isinstance(extra.get(k), (int, float)):
                    vals[k].append(float(extra[k]))
        return "/".join(f"{sum(vals[k]) / len(vals[k]):.3f}" for k in keys if vals[k])

    def node(self, r: dict[str, Any], extra_parents: list[str]) -> str:
        """One experiment as raw facts: what was planned, and what was measured (or why nothing was)."""
        meta = json.loads(r.get("metadata_json") or "{}").get("candidate", {}) or {}
        raw = meta.get("changes_hint")
        hints = [h for h in (raw if isinstance(raw, list) else [raw]) if isinstance(h, dict)]
        where = ("@" + ",".join(":".join(str(h[k]).split("/")[-1] for k in ("file", "function") if h.get(k)) for h in hints)
                 if hints else "")
        knobs = self._knobs(r)
        plan = (f"{r['id']} [{r.get('kind') or '?'}]" + (f" +{','.join(extra_parents)}" if extra_parents else "")
                + (f" {{{knobs}}}" if knobs else "") + f" {clip(r.get('title'), 70)}"
                + f" | plan: {clip(r.get('mechanism') or r.get('hypothesis'), 220)}" + (f" {where}" if where else ""))
        lesson = str(r.get("lesson") or "")
        if lesson.startswith("[KNOB CHANGE NEVER APPLIED"):
            plan += " | NOTE: knob change never applied, code identical to parent (result = parent re-run)"
        elif lesson.startswith("[INVALID: benchmark structure leak]") or lesson.startswith("[NOT COMPARABLE: pair-role leak]"):
            plan += " | NOTE: " + clip(lesson.split("|")[0], 200)
        dropped = json.loads(r.get("metadata_json") or "{}").get("migration_dropped")
        if dropped:
            plan += f" | NOTE: inherited changes to {','.join(dropped)} were dropped (benchmark loop now fixed)"
        if r.get("metric") is None:
            reason = " ".join(str(r.get("failure_reason") or r.get("status")).split())
            return plan + f" | result: NOT RUN ({outcome(r)[8:]}): {reason[-150:]}"
        dp, side = r.get("delta_parent"), self._side(r["id"], r.get("stage"))
        return (plan + f" | result: {r.get('stage')} n{self._n(r)} mean={float(r['metric']):.4f}"
                f" Δb{float(r.get('delta_baseline') or 0):+.4f}" + (f" Δp{float(dp):+.4f}" if dp is not None else "")
                + (f" side={side}" if side else ""))

    def rejected(self, limit: int = 40) -> list[str]:
        """Recent candidates that were proposed but never became experiments, with the reason."""
        rows = self.db.query("SELECT ts,message,payload_json FROM events WHERE type='candidate_rejected' ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in reversed(rows):
            c = json.loads(r.get("payload_json") or "{}")
            if "critic" in str(c.get("reason") or r["message"]):
                continue  # from the retired critic/quota stage: says nothing about the idea
            if c.get("title"):
                params = c.get("params") or {}
                out.append(f"[{c.get('kind') or '?'}] {clip(c.get('title'), 70)} parents={','.join(map(str, c.get('parent_ids') or [])) or '-'}"
                           + (f" params={json.dumps(params, ensure_ascii=False)}" if params else "")
                           + f" | plan: {clip(c.get('mechanism'), 140)} | REJECTED: {clip(c.get('reason'), 160)}")
            else:  # older events carry only "title: reason"
                out.append(f"{clip(r['message'], 260)}")
        return out

    def tree(self, budget: int) -> list[str]:
        """The whole experiment DAG as an indented tree (child under its primary parent, in creation order).

        Over budget, the least informative leaves are folded into per-parent summaries (old, not-run or
        within-noise first). Paths to promising, pinned and recent experiments are never folded.
        """
        rows = self._rows()
        by = {r["id"]: r for r in rows}
        parent, note = {}, {}
        for r in rows:
            p = r.get("parent_primary") or "baseline"
            if p != "baseline" and p not in by:
                note[r["id"]], p = f" (parent {p} was interrupted)", "baseline"
            parent[r["id"]] = p
        extra: dict[str, list[str]] = defaultdict(list)
        for e in self.db.query("SELECT parent_id,child_id FROM edges"):
            if e["child_id"] in by and e["parent_id"] != by[e["child_id"]].get("parent_primary"):
                extra[e["child_id"]].append(e["parent_id"])
        kids: dict[str, list[str]] = defaultdict(list)
        depth = {"baseline": 0}
        for r in rows:  # creation order: a parent precedes its children
            kids[parent[r["id"]]].append(r["id"])
            depth[r["id"]] = depth.get(parent[r["id"]], 0) + 1
        pad = lambda i: "  " * min(depth[i], 12)
        text = {i: pad(i) + self.node(by[i], extra[i]) + note.get(i, "") for i in by}

        keep: set[str] = set()
        anchors = [r["id"] for r in rows if r["status"] in ("promising", "champion") or r.get("pinned")] + [r["id"] for r in rows[-8:]]
        for x in anchors:
            while x in by and x not in keep:
                keep.add(x)
                x = parent[x]
        noise = self.noise("quick", 1) if self.noise else 0.0
        rank = {r["id"]: i for i, r in enumerate(rows)}
        score = {i: rank[i] + (len(rows) if by[i].get("metric") is not None and abs(float(by[i].get("delta_baseline") or 0)) >= noise else 0)
                 for i in by}

        def descendants(i: str) -> list[str]:
            out, stack = [], list(kids.get(i, []))
            while stack:
                x = stack.pop()
                out.append(x)
                stack.extend(kids.get(x, []))
            return out

        def summary(p: str, hidden_kids: list[str]) -> str:
            ids = hidden_kids + [d for h in hidden_kids for d in descendants(h)]
            measured = [by[i] for i in ids if by[i].get("metric") is not None]
            best = max((float(r.get("delta_baseline") or 0) for r in measured), default=None)
            kinds = Counter(by[i].get("kind") or "?" for i in ids)
            return ("  " * min(depth[p] + 1, 12) + f"⋯ {len(ids)} folded ({min(ids)}…{max(ids)}): {len(measured)} measured"
                    + (f", best Δb{best:+.3f}" if best is not None else "") + f", {len(ids) - len(measured)} not run, "
                    + ", ".join(f"{k}={n}" for k, n in kinds.most_common()))

        hidden: set[str] = set()

        def render() -> list[str]:
            out = ["baseline"]
            stack: list[tuple[str, bool]] = [("baseline", False)]
            while stack:  # iterative DFS keeping creation order; a parent's fold summary follows its visible children
                p, done = stack.pop()
                if done:
                    folded = [c for c in kids.get(p, []) if c in hidden]
                    if folded:
                        out.append(summary(p, folded))
                    continue
                if p != "baseline":
                    out.append(text[p])
                stack.append((p, True))
                stack.extend((c, False) for c in reversed(kids.get(p, [])) if c not in hidden)
            return out

        lines = render()
        while sum(len(x) + 1 for x in lines) > budget:
            leaves = [i for i in by if i not in hidden and i not in keep and all(c in hidden for c in kids.get(i, []))]
            if not leaves:
                break
            hidden.update(sorted(leaves, key=score.get)[: max(1, len(leaves) // 10)])
            lines = render()
        return lines

    def guidance(self) -> list[str]:
        return [f"[{g['kind']}{'@' + g['experiment_id'] if g.get('experiment_id') else ''}] {clip(g['text'], 400)}"
                for g in self.db.recent_guidance(10) if g["kind"] != "steer"]

    def idea(self) -> list[str]:
        digest = self.db.get_meta("idea_digest", {}) or {}
        if not digest.get("mechanisms"):
            return [clip(digest.get("raw", ""), 3000)] if digest.get("raw") else []
        refs = Counter()
        for r in self._rows():
            ref = json.loads(r.get("metadata_json") or "{}").get("candidate", {}).get("idea_ref")
            if ref:
                refs[str(ref)] += 1
        return [f"{m.get('id')} (tested {refs.get(str(m.get('id')), 0)}x): {clip(m.get('name'), 60)} — "
                f"{clip(m.get('claim'), 160)} | in this code: {clip(m.get('code_mapping'), 160)}"
                for m in digest["mechanisms"]]

    # ---------- briefing ----------
    def briefing(self, *, idea: bool = False) -> str:
        """Raw research record only: experiment plans, their results, rejected candidates.

        No harness-written interpretation (paths, summaries, beliefs, policy): planner-tier agents compare and
        conclude themselves. Human guidance is included because it is the human's own input.
        """
        side = "/".join(self.cfg.get("objective", "side_metrics", []) or [])
        budget = int(self.cfg.get("context", "tree_chars", 60000))
        parts = [
            section("OBJECTIVE AND NOISE", self.objective(), 600),
            section("HUMAN GUIDANCE (highest priority)", self.guidance(), 2500),
        ]
        if idea:
            parts.append(section("RESEARCH IDEA MECHANISMS (from the research note; tested = experiments citing it)", self.idea(), 3500))
        parts += [
            section("EXPERIMENT TREE: every experiment under its primary parent, in creation order. "
                    "id [kind] +extra parents {knobs vs defaults} title | plan | result: stage n=runs mean Δb=vs baseline Δp=vs parent"
                    + (f" side={side}" if side else ""), self.tree(budget), budget + 300),
            section("REJECTED CANDIDATES (proposed, never run; most recent last)", self.rejected(), 6000),
            section("ARTIFACTS (read on demand with your tools)", [
                f"{self.cfg.artifact_dir}/<experiment id>/patch.diff = its exact code change vs its primary parent, "
                f"e.g. {self.cfg.artifact_dir}/exp_000058/patch.diff",
                f"{self.cfg.artifact_dir}/<experiment id>/worker_summary.txt = what the worker says it implemented"], 600),
        ]
        return "\n\n".join(p for p in parts if p)

    def worker_context(self, parent_ids: list[str], patch_of) -> str:
        parents = [p for p in (self.db.experiment(x) for x in parent_ids) if p]
        lines = [self.line(p, 300) for p in parents]
        patches = []
        for p in parents[1:]:  # the primary parent's code is already in the worktree (see PARENT LINEAGE DIFF)
            diff = patch_of(p["id"])
            if diff:
                patches.append(f"### {p['id']} patch to re-apply on top of the primary parent\n{diff[:3000]}")
        return "\n\n".join(x for x in [
            section("PARENTS (first = primary; its code and knobs are already in your worktree)", lines, 2500),
            "\n\n".join(patches),
        ] if x)
