"""Extract Bash + MCP tool-call frequencies from all transcripts under ~/.claude/projects.

Used to populate `.claude/settings.json` permissions.allow with the most-prompted
read-only patterns. Read-only filtering + auto-allow filtering is delegated to the
calling skill — this script just tabulates raw counts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"


def leading_command(cmd: str) -> str:
    """Extract command + first subcommand pair from a shell string.

    Handles: env-var prefixes (`FOO=bar cmd`), `sudo`, `timeout`, simple pipes/`&&`,
    quoted absolute paths (`"C:\\...\\conda.exe" run -n env cmd subcmd`).
    """
    s = cmd.strip()
    # Strip env-var prefixes
    s = re.sub(r"^(?:\w+=\S+\s+)+", "", s)
    # Take first chained segment
    s = re.split(r"\s*(?:&&|\|\||;|\|)\s*", s, maxsplit=1)[0]
    # Strip leading sudo/timeout/etc
    for prefix in ("sudo", "timeout", "nohup", "exec"):
        if s.startswith(prefix + " "):
            s = s[len(prefix) + 1:]

    tokens = []
    i = 0
    while i < len(s) and len(tokens) < 4:
        if s[i] in ('"', "'"):
            quote = s[i]
            j = s.find(quote, i + 1)
            if j < 0:
                break
            tokens.append(s[i + 1: j])
            i = j + 1
            while i < len(s) and s[i] == " ":
                i += 1
        else:
            j = s.find(" ", i)
            if j < 0:
                tokens.append(s[i:])
                break
            tokens.append(s[i:j])
            i = j + 1

    if not tokens:
        return ""
    cmd0 = Path(tokens[0]).name.lower() if tokens[0] else ""

    # Special-case conda: `conda run -n env actual_cmd ...` → the actual command is what matters
    if cmd0.startswith("conda") and len(tokens) >= 4 and tokens[1] == "run" and tokens[2] == "-n":
        inner = Path(tokens[4]).name.lower() if len(tokens) > 4 else ""
        sub = tokens[5] if len(tokens) > 5 else ""
        return f"conda run {inner} {sub}".strip()

    sub = tokens[1] if len(tokens) > 1 and not tokens[1].startswith("-") else ""
    return f"{cmd0} {sub}".strip()


bash_counts: Counter[str] = Counter()
mcp_counts: Counter[str] = Counter()

jsonl_files = sorted(PROJECTS.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
print(f"Scanning {len(jsonl_files)} transcript files...")

for jf in jsonl_files:
    try:
        for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "assistant":
                continue
            msg = obj.get("message") or {}
            for item in msg.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "tool_use":
                    continue
                name = item.get("name", "")
                inp = item.get("input") or {}
                if name == "Bash":
                    cmd = leading_command(inp.get("command", ""))
                    if cmd:
                        bash_counts[cmd] += 1
                elif name.startswith("mcp__"):
                    mcp_counts[name] += 1
    except OSError:
        continue

print("\nTop 40 Bash patterns:")
for cmd, n in bash_counts.most_common(40):
    print(f"  {n:5d}  {cmd}")

print(f"\nTop 20 MCP tools (of {len(mcp_counts)} total):")
for name, n in mcp_counts.most_common(20):
    print(f"  {n:5d}  {name}")
