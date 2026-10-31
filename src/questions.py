"""Survey questions used by the experiment, read from config/questions.json.

To add a question, add an entry to config/questions.json (no code changes needed):

    "P99": {
      "topic": "short name", "role": "divisive" or "neutral",
      "codes": [1, 2, 3, 4],                      # substantive CIS answer codes, in CIS order
      "es": {"question": "...", "options": {"1": "...", "2": "...", ...}},
      "en": {"question": "...", "options": {"1": "...", "2": "...", ...}}
    }

Then re-run build_personas.py and build_prompts.py. The variable must exist in the CIS 3194 file,
and "codes" must be the answer codes that count as a real answer (N.S./N.C. are left out).
"""

from __future__ import annotations

import json

from load_cis3194 import PROJECT_ROOT

QUESTIONS_PATH = PROJECT_ROOT / "config" / "questions.json"


def load_questions(path=QUESTIONS_PATH) -> dict[str, dict]:
    """item -> {topic, role, codes, es: {question, options: {code: text}}, en: {...}} with int option codes."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = {}
    for item, q in raw.items():
        codes = [int(c) for c in q["codes"]]
        spec = {"topic": q.get("topic", item), "role": q.get("role", ""), "codes": codes}
        for lang in ("es", "en"):
            options = {int(c): t for c, t in q[lang]["options"].items()}
            if list(options) != codes:
                raise ValueError(f"{item}/{lang}: options {list(options)} do not match codes {codes}")
            spec[lang] = {"question": q[lang]["question"], "options": options}
        items[item] = spec
    return items
