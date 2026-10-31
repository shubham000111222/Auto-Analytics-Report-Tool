from __future__ import annotations
import json
import re
from dataclasses import dataclass
import anthropic

SYSTEM_PROMPT = """You are a data analyst writing a plain-language report section.
You are given ONLY pre-computed statistics as JSON -- never raw data rows.
Rules:
1. Only state facts directly derivable from the JSON. Never invent numbers.
2. Every number you state must match a value in the JSON (rounding to 1 decimal is fine).
3. If the JSON doesn't support a claim, say "not enough data to assess" instead of guessing.
4. Write 5-8 one-sentence bullets: insights, anomalies, data quality issues, recommended next steps.
5. Never mention JSON, code, or that you were given statistics.
"""

NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


@dataclass
class NarrativeResult:
    text: str
    valid: bool
    unverified_numbers: list[str]


def _flatten_numbers(obj) -> list[float]:
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
        elif isinstance(o, (int, float)):
            out.append(float(o))
    walk(obj)
    return out


def validate_narrative(text: str, stats_blob: dict, tolerance: float = 0.15) -> tuple[bool, list[str]]:
    known = _flatten_numbers(stats_blob)
    unverified = []
    for raw in NUMBER_PATTERN.findall(text):
        try:
            val = float(raw)
        except ValueError:
            continue
        if val in (0, 1, 2, 3, 4, 5, 10, 100):
            continue
        if not any(abs(val - k) <= max(tolerance, abs(k) * tolerance) for k in known):
            unverified.append(raw)
    return (len(unverified) == 0, unverified)


def generate_narrative(stats_blob: dict, api_key: str, model: str = "claude-sonnet-4-5",
                        max_retries: int = 2) -> NarrativeResult:
    client = anthropic.Anthropic(api_key=api_key)
    payload = json.dumps(stats_blob, default=str)
    last_unverified: list[str] = []

    for attempt in range(max_retries + 1):
        extra = (f"\n\nYour previous answer used unverifiable numbers: {last_unverified}. "
                 f"Rewrite using only numbers from the JSON." if attempt else "")
        resp = client.messages.create(
            model=model, max_tokens=800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Statistics:\n{payload}{extra}"}],
        )
        text = resp.content[0].text
        ok, unverified = validate_narrative(text, stats_blob)
        if ok:
            return NarrativeResult(text, True, [])
        last_unverified = unverified

    return NarrativeResult(_template_fallback(stats_blob), False, last_unverified)


def _template_fallback(stats_blob: dict) -> str:
    lines = ["- AI narrative could not be fully verified; showing computed facts instead:"]
    dq = stats_blob.get("data_quality", {})
    if dq.get("worst_col"):
        lines.append(f"- {dq['worst_col']} has the highest missing rate at {dq['worst_pct']}%.")
    if "n_rows" in stats_blob:
        lines.append(f"- Dataset contains {stats_blob['n_rows']} rows and {stats_blob.get('n_cols')} columns.")
    return "\n".join(lines)
