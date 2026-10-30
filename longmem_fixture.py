import json
import re
from pathlib import Path

from harness import Case

_EXPECTED_RE = re.compile(r"expected answer is ['\"](.*?)['\"]", re.IGNORECASE | re.DOTALL)


def _render_conversation(steps) -> str:
    """Concatenate every add_messages turn, in order, as 'User:' / 'Assistant:' lines."""
    lines = []
    for step in steps:
        if step.get("step_type") != "add_messages":
            continue
        for msg in step.get("messages", []):
            who = str(msg.get("peer_id", "?")).capitalize()
            lines.append(f"{who}: {msg.get('content', '')}")
    return "\n".join(lines)


def _extract_expected(assertion: dict) -> str:
    """Pull the gold answer out of Honcho's llm_judge prompt; fall back to the whole prompt."""
    prompt = assertion.get("prompt", "")
    m = _EXPECTED_RE.search(prompt)
    return m.group(1).strip() if m else prompt.strip()


def parse_fixture(path) -> list[dict]:
    """Pure parse (no model calls): fixture file -> [{id, question, expected, context}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = data.get("steps", [])
    context = _render_conversation(steps)
    name = Path(path).stem
    out, q_idx = [], 0
    for step in steps:
        if step.get("step_type") != "query":
            continue
        assertions = step.get("assertions", [])
        expected = _extract_expected(assertions[0]) if assertions else ""
        out.append({
            "id": name if q_idx == 0 else f"{name}-{q_idx}",
            "question": step.get("input", "").strip(),
            "expected": expected,
            "context": context,
        })
        q_idx += 1
    return out


def generate_actual(client, model, context: str, question: str) -> str:
    """Produce the answer-under-test with the local model, grounded in the conversation."""
    prompt = (
        "You are a helpful assistant with memory of the conversation below. "
        "Answer the user's final question using only what the conversation supports. "
        "If the conversation does not contain the answer, say you don't have that information.\n\n"
        f"Conversation:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def load_cases(paths, client, model) -> list[Case]:
    """Parse fixtures, generate a local model answer for each query, return Cases."""
    cases = []
    for path in paths:
        for rec in parse_fixture(path):
            actual = generate_actual(client, model, rec["context"], rec["question"])
            cases.append(Case(
                id=rec["id"],
                question=rec["question"],
                expected=rec["expected"],
                actual=actual,
                context=rec["context"],
                human_label=None,
            ))
    return cases
