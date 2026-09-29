"""Compare the numbers (1, 2, 3...) and letters (A, B, C...) runs of the same model.

Question: is the order effect (large order_tv on P11-en and P42) an artefact of the
answer token, or of option position / content?

Reads from data/results/ (the notebook's downloads):
    model_distributions_<MODEL>.csv            model_prompt_level_<MODEL>.csv            (numbers)
    model_distributions_letters_<MODEL>.csv    model_prompt_level_letters_<MODEL>.csv    (letters)
and the prompt files in data/processed/ to recover which position each code was shown in.

Prints:
  1. mean order_tv by item x language, numbers vs letters
  2. where the model's top choice sits (first / middle / last position shown), and the
     mean probability on the last position: tests the "never picks the last option" pattern
  3. for the chosen items (default P42): religiosity x top choice per order, as
     "position->code", for both variants

Usage:
    python src/compare_label_variants.py
    python src/compare_label_variants.py --model Llama-3.1-8B-Instruct --crosstab P42 P11
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from load_cis3194 import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VARIANTS = {
    "numbers": {"prefix": "", "prompts": "prompts.jsonl", "mapping": "number_to_code"},
    "letters": {"prefix": "letters_", "prompts": "prompts_letters.jsonl", "mapping": "label_to_code"},
}


def load_variant(variant: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distributions and prompt-level results, the latter with the shown position of each code."""
    spec = VARIANTS[variant]
    paths = [RESULTS_DIR / f"model_{kind}_{spec['prefix']}{model}.csv" for kind in ("distributions", "prompt_level")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing in {RESULTS_DIR}: {', '.join(missing)}")
    dist, prompt_level = (pd.read_csv(p) for p in paths)

    # position = 1-based place in the option list as shown in that prompt
    positions = []
    with (PROCESSED_DIR / spec["prompts"]).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            codes = list(record[spec["mapping"]].values())
            positions += [
                {"prompt_id": record["prompt_id"], "code": code, "position": i, "n_options": len(codes)}
                for i, code in enumerate(codes, start=1)
            ]
    prompt_level = prompt_level.merge(pd.DataFrame(positions), on=["prompt_id", "code"], validate="one_to_one")
    return dist.assign(variant=variant), prompt_level.assign(variant=variant)


def top_choices(prompt_level: pd.DataFrame) -> pd.DataFrame:
    top = prompt_level.loc[prompt_level.groupby(["variant", "prompt_id"])["proportion"].idxmax()].copy()
    top["where"] = "middle"
    top.loc[top["position"] == 1, "where"] = "first"
    top.loc[top["position"] == top["n_options"], "where"] = "last"
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Mistral-7B-Instruct-v0.3")
    parser.add_argument("--crosstab", nargs="*", default=["P42"], help="items to show religiosity x choice for")
    args = parser.parse_args()

    loaded = [load_variant(v, args.model) for v in VARIANTS]
    dist = pd.concat([d for d, _ in loaded], ignore_index=True)
    prompt_level = pd.concat([p for _, p in loaded], ignore_index=True)  # unique index needed for idxmax
    personas = pd.read_csv(PROCESSED_DIR / "personas.csv")[["persona_id", "religiosity"]]
    pd.set_option("display.width", 200)

    print(f"=== 1. Mean order_tv (0 = no order effect, 1 = completely different) — {args.model}")
    tv = dist.pivot_table(index=["item", "lang"], columns="variant", values="order_tv")[list(VARIANTS)]
    tv["letters - numbers"] = tv["letters"] - tv["numbers"]
    print(tv.round(2).to_string())

    print("\n    Mean valid_mass (probability on valid labels before renormalizing):")
    vm = prompt_level.drop_duplicates(["variant", "prompt_id"])
    print(vm.pivot_table(index=["item", "lang"], columns="variant", values="valid_mass")[list(VARIANTS)].round(3).to_string())

    top = top_choices(prompt_level)
    print("\n=== 2. Position of the top choice, % of prompts (both orders pooled)")
    where = top.groupby(["item", "lang", "variant"])["where"].value_counts(normalize=True).unstack(["variant", "where"]).fillna(0) * 100
    where = where.reindex(columns=pd.MultiIndex.from_product([list(VARIANTS), ["first", "middle", "last"]]), fill_value=0)
    print(where.round(0).astype(int).to_string())

    print("\n    Mean probability on the LAST shown option vs the FIRST:")
    ends = prompt_level[prompt_level["position"].isin([1]) | (prompt_level["position"] == prompt_level["n_options"])]
    ends = ends.assign(end=lambda d: d["position"].eq(1).map({True: "first", False: "last"}))
    print(
        ends.pivot_table(index=["item", "lang"], columns=["variant", "end"], values="proportion")
        .reindex(columns=pd.MultiIndex.from_product([list(VARIANTS), ["first", "last"]]))
        .round(3).to_string()
    )

    top = top.merge(personas, on="persona_id")
    top["choice"] = top["position"].astype(str) + "->c" + top["code"].astype(int).astype(str)
    for item in args.crosstab:
        for lang in sorted(top["lang"].unique()):
            g = top[(top["item"] == item) & (top["lang"] == lang)]
            print(f"\n=== 3. {item}-{lang}: religiosity x top choice (shown position -> CIS code)")
            for variant in VARIANTS:
                v = g[g["variant"] == variant]
                print(f"--- {variant}")
                print(pd.crosstab(v["religiosity"], [v["order"], v["choice"]]).to_string())


if __name__ == "__main__":
    main()
