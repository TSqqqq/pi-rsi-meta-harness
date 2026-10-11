/**
 * RSI context hygiene: before every model call, older intermediate steps are cut down so only the
 * prompt, the agent's actions and its newest results stay in full.
 *
 * - Reasoning from earlier turns is dropped (thinking blocks, and text before an inline "</think>"):
 *   the model already acted on it.
 * - Tool results older than the last KEEP_RECENT turns are replaced by a one-line stub
 *   ("[tool output elided: N chars]"). Their tool calls (the arguments: which file, which edit)
 *   stay, so the step sequence is still visible; the agent re-reads a file when it needs it again.
 * - Results of edit/write stay (they are short and confirm what changed).
 * - With RSI_BLOCK_GIT_HISTORY=1 (workers), git log/show/reflog/blame are blocked: the lineage diff is
 *   already in the prompt.
 *
 * The transcript on disk is untouched; only the request context is trimmed. Deterministic: the same
 * history always produces the same trimmed context, which keeps provider prompt caches warm.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const KEEP_RECENT = Number(process.env.RSI_CONTEXT_KEEP_RECENT ?? 2);
const KEEP_TOOLS = new Set(["edit", "write"]);

type Block = { type: string; text?: string; [k: string]: unknown };
type Msg = { role: string; content?: string | Block[]; toolName?: string; [k: string]: unknown };

const size = (m: Msg) =>
  typeof m.content === "string" ? m.content.length
    : (m.content ?? []).reduce((n, b) => n + (typeof b.text === "string" ? b.text.length : 0), 0);

// Workers get the full lineage diff in their prompt; digging through git history only fills the context.
const BLOCK_GIT_HISTORY = process.env.RSI_BLOCK_GIT_HISTORY === "1";
const GIT_HISTORY = /\bgit\b[^|;&]*\s(log|show|reflog|blame)\b/;

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event) => {
    // logzip replaces the built-in bash with bash-compressed; both run shell commands.
    if (!BLOCK_GIT_HISTORY || (event.toolName !== "bash" && event.toolName !== "bash-compressed")) return undefined;
    const command = String((event.input as { command?: unknown }).command ?? "");
    if (!GIT_HISTORY.test(command)) return undefined;
    return { block: true, reason: "git history is not needed: the PARENT LINEAGE DIFF in your prompt is the complete change of your starting code vs baseline" };
  });

  pi.on("context", async (event) => {
    const messages = event.messages as unknown as Msg[];
    // Index of the assistant turn that starts the protected tail.
    const turns = messages.map((m, i) => (m.role === "assistant" ? i : -1)).filter((i) => i >= 0);
    const tailStart = turns.length > KEEP_RECENT ? turns[turns.length - KEEP_RECENT] : 0;
    const out = messages.map((m, i) => {
      if (i >= tailStart) return m;
      if (m.role === "assistant" && Array.isArray(m.content)) {
        // Thinking blocks, and reasoning that a local server leaves inline as "...</think>" in the text.
        let changed = false;
        const kept = m.content.filter((b) => b.type !== "thinking").map((b) => {
          if (b.type !== "text" || typeof b.text !== "string" || !b.text.includes("</think>")) return b;
          changed = true;
          return { ...b, text: b.text.slice(b.text.lastIndexOf("</think>") + 8).trim() };
        }).filter((b) => b.type !== "text" || b.text);
        return !changed && kept.length === m.content.length ? m : { ...m, content: kept };
      }
      if (m.role === "toolResult" && !KEEP_TOOLS.has(String(m.toolName))) {
        const n = size(m);
        if (n <= 200) return m;
        return { ...m, content: [{ type: "text", text: `[${m.toolName} output elided: ${n} chars; re-run the call if you need it again]` }] };
      }
      return m;
    });
    return { messages: out as unknown as typeof event.messages };
  });
}
