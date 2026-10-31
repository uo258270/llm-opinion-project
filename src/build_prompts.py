"""Build Spanish and English prompts for every persona x item.

Each prompt = persona description + survey question + numbered substantive options +
an instruction to answer with the option number only. N.S./N.C. are not offered,
matching the target distributions (which exclude them).

Order control: every prompt exists in two versions, with the options in the CIS order
("original") and in reverse ("reversed"). Options are always numbered 1..k from the
top, so in the reversed version both position and number change; each record's
`number_to_code` maps the displayed number back to the CIS code (= the columns of
personas.csv). Model distributions from the two orders are averaged downstream.

No-persona baseline: persona "p00" is the same set of questions with no persona sentence at all
(the prompt is just the question and options). It shows what the model answers by default and lets
us test whether adding demographic information improves the match with each subgroup's answers.

More option orders (--orders cyclic): besides the original/reversed pair, every item is shown
with each cyclic rotation of the original list and of the reversed list (2k orders for an
item with k options; names "rot1", "rrot1", ...; "original" = rot0, "reversed" = rrot0).
This averages out position bias much better than two orders, but rotations of an ordinal scale
(e.g. Agree ... Disagree) are unnatural lists, so cyclic results should be read as a robustness check.
Outputs then get a "cyclic" infix: prompts_cyclic.jsonl, prompts_cyclic_letters.jsonl.

Label control: the same prompts are also written with lettered options ("A.", "B.",
...) and an instruction to answer with the letter, to test whether order effects come
from the answer token (digit) rather than from the option content. Letter records use
`label_to_code` instead of `number_to_code`; everything else is identical.

Wording:
  - Spanish: CIS 3194 questionnaire (cues3194.pdf), with two adaptations:
    questions are put in "tú" instead of "Ud." to match the persona text and the
    English "you"; P42's options, printed in the third person for the interviewer
    ("No cree en Dios"), are put in the first person.
  - English: official ISSP master-questionnaire wording (2017 Social Networks,
    2018 Religion), checked against the source questionnaires. P35 is the ISSP trend
    item ("A man's job ... a woman's job ..."), so it keeps that wording even though
    the CIS version says "marido"/"mujer" (husband/wife).
  - Both languages say the persona lives in Spain, so language is the only thing
    that differs between the two versions of a prompt.

Outputs (data/processed/, one JSON object per line):
    prompts.jsonl           numbered options (1, 2, 3, ...)
    prompts_letters.jsonl   lettered options (A, B, C, ...)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from load_cis3194 import PROJECT_ROOT
from questions import load_questions

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LANGS = ["es", "en"]
ORDER_MODES = ["two", "cyclic"]

SEX_TEXT = {
    "es": {"Hombre": "un hombre", "Mujer": "una mujer"},
    "en": {"Hombre": "a man", "Mujer": "a woman"},
}
AGE_TEXT = {
    "es": {"18-34": "de entre 18 y 34 años", "35-54": "de entre 35 y 54 años", "55+": "de 55 años o más"},
    "en": {"18-34": "aged 18 to 34", "35-54": "aged 35 to 54", "55+": "aged 55 or older"},
}
RELIGIOSITY_TEXT = {
    "es": {
        "Religiosa": "una persona religiosa",
        "Ni religiosa ni no religiosa": "una persona ni religiosa ni no religiosa",
        "No religiosa": "una persona no religiosa",
    },
    "en": {
        "Religiosa": "a religious person",
        "Ni religiosa ni no religiosa": "neither a religious nor a non-religious person",
        "No religiosa": "a non-religious person",
    },
}
PERSONA_TEMPLATE = {
    "es": "Eres {sex} {age} que vive en España. Te consideras {religiosity}.",
    "en": "You are {sex} {age} living in Spain. You consider yourself {religiosity}.",
}
# Option labels, top to bottom, and the matching answer instruction per label style.
LABELS = {
    "numbers": [str(i) for i in range(1, 13)],  # up to 12 options per question
    "letters": list("ABCDEFGHIJKL"),
}
INSTRUCTION = {
    "numbers": {
        "es": "Responde únicamente con el número de la opción que mejor refleje tu opinión.",
        "en": "Answer only with the number of the option that best reflects your view.",
    },
    "letters": {
        "es": "Responde únicamente con la letra de la opción que mejor refleje tu opinión.",
        "en": "Answer only with the letter of the option that best reflects your view.",
    },
}
MAPPING_KEY = {"numbers": "number_to_code", "letters": "label_to_code"}


def output_file(style: str, orders: str) -> str:
    infix = "_cyclic" if orders == "cyclic" else ""
    return f"prompts{infix}{'_letters' if style == 'letters' else ''}.jsonl"

ANSWER_CUE = {"es": "Respuesta:", "en": "Answer:"}

ITEMS = load_questions()  # config/questions.json


BASELINE_ID = "p00"


def persona_text(persona: pd.Series, lang: str) -> str:
    if persona["persona_id"] == BASELINE_ID:
        return ""
    return PERSONA_TEMPLATE[lang].format(
        sex=SEX_TEXT[lang][persona["sex"]],
        age=AGE_TEXT[lang][persona["age_group"]],
        religiosity=RELIGIOSITY_TEXT[lang][persona["religiosity"]],
    )


def orders_for(item: str, mode: str) -> dict[str, list[int]]:
    """Order name -> CIS codes from top to bottom. "original" and "reversed" always exist."""
    base = list(ITEMS[item]["es"]["options"])
    rev = base[::-1]
    if mode == "two":
        return {"original": base, "reversed": rev}
    k = len(base)
    out = {}
    for j in range(k):
        out["original" if j == 0 else f"rot{j}"] = base[j:] + base[:j]
    for j in range(k):
        out["reversed" if j == 0 else f"rrot{j}"] = rev[j:] + rev[:j]
    return out


def label_to_code(item: str, lang: str, order: str, style: str, mode: str = "two") -> dict[str, int]:
    """Displayed option label (always 1..k / A.. from the top) -> CIS code."""
    return dict(zip(LABELS[style], orders_for(item, mode)[order]))


def question_text(item: str, lang: str, order: str, style: str, mode: str = "two") -> str:
    spec = ITEMS[item][lang]
    options = "\n".join(
        f"{label}. {spec['options'][code]}" for label, code in label_to_code(item, lang, order, style, mode).items()
    )
    return f"{spec['question']}\n\n{options}\n\n{INSTRUCTION[style][lang]}\n{ANSWER_CUE[lang]}"


def build_prompts(personas: pd.DataFrame, style: str, mode: str = "two") -> list[dict]:
    records = []
    for _, persona in personas.iterrows():
        for item in ITEMS:
            for lang in LANGS:
                for order in orders_for(item, mode):
                    p_text, q_text = persona_text(persona, lang), question_text(item, lang, order, style, mode)
                    records.append({
                        "prompt_id": f"{persona['persona_id']}_{item}_{lang}_{order}",
                        "persona_id": persona["persona_id"],
                        "item": item,
                        "lang": lang,
                        "order": order,
                        "persona_text": p_text,
                        "question_text": q_text,
                        "prompt": f"{p_text}\n\n{q_text}" if p_text else q_text,
                        MAPPING_KEY[style]: label_to_code(item, lang, order, style, mode),
                    })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orders", choices=ORDER_MODES, default="two", help="option orders per prompt (default: two)")
    args = parser.parse_args()

    personas = pd.read_csv(PROCESSED_DIR / "personas.csv")
    personas = pd.concat([pd.DataFrame([{"persona_id": BASELINE_ID}]), personas], ignore_index=True)
    for style in LABELS:
        records = build_prompts(personas, style, args.orders)
        out_path = PROCESSED_DIR / output_file(style, args.orders)
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} prompts to {out_path}")

    print(f"\n--- {records[1]['prompt_id']} ({style}) ---\n{records[1]['prompt']}")


if __name__ == "__main__":
    main()
