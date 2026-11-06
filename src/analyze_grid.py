"""Compare the model distributions in the grid results with the real CIS 3194 distributions.

Input:  data/results/grid_results.csv  (one row per model x method x persona x item x language,
        written by notebooks/run_grid_colab.ipynb; option orders already averaged)
        data/processed/personas.csv and population_distributions.csv (real distributions)
Output (data/results/):
    grid_with_distances.csv   the grid table plus the distance to the survey for every row
    summary_*.csv             the summary tables printed below

Distances are Jensen-Shannon distances (square root of the divergence, base 2): 0 = identical, 1 = nothing in common.
In grid_with_distances.csv the no-persona rows (p00) are compared with the whole-sample distribution and the others with their own group.
The 'persona_gain' tables compare, group by group, the persona answer and the no-persona answer with that group's own real distribution.

Usage:
    python src/analyze_grid.py [path/to/grid_results.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from load_cis3194 import PROJECT_ROOT
from questions import load_questions

RESULTS = PROJECT_ROOT / "data" / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
RELIABLE_ORDER_TV = 0.9   # above this, the answer changes almost completely when the options are reversed


def real_distribution(item, persona_id, personas, population, codes):
    if persona_id == "p00":
        return population[population["item"] == item].sort_values("code")["proportion"].to_numpy()
    row = personas[personas["persona_id"] == persona_id].iloc[0]
    return np.array([row[f"{item}_{c}"] for c in codes])


def add_distances(grid: pd.DataFrame, questions: dict) -> pd.DataFrame:
    personas = pd.read_csv(PROCESSED / "personas.csv")
    population = pd.read_csv(PROCESSED / "population_distributions.csv")
    out = []
    for r in grid.itertuples(index=False):
        codes = questions[r.item]["codes"]
        q = np.array([getattr(r, f"p_{c}") for c in codes], dtype=float)
        p = real_distribution(r.item, r.persona_id, personas, population, codes)
        scale = np.arange(1, len(codes) + 1)
        out.append({
            "jsd_to_survey": jensenshannon(p, q / q.sum(), base=2) if np.isfinite(q).all() and q.sum() > 0 else np.nan,
            "real_mean": (p * scale).sum(),
            "model_mean": (q / q.sum() * scale).sum() if np.isfinite(q).all() and q.sum() > 0 else np.nan,
            "model_max_p": q.max() / q.sum() if np.isfinite(q).all() and q.sum() > 0 else np.nan,
        })
    return pd.concat([grid.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def persona_gain(d: pd.DataFrame, questions: dict) -> pd.DataFrame:
    """Does knowing the persona help? For each group, compare how far the persona answer and the no-persona (p00)
    answer are from that same group's real distribution. gain > 0 means the persona answer is closer."""
    personas = pd.read_csv(PROCESSED / "personas.csv")
    rows = []
    for (m, meth, it, lang), g in d.groupby(["model", "method", "item", "lang"]):
        codes = questions[it]["codes"]
        cols = [f"p_{c}" for c in codes]
        base = g[g["persona_id"] == "p00"][cols].to_numpy(float)[0]
        for r in g[g["persona_id"] != "p00"].itertuples(index=False):
            q = np.array([getattr(r, c) for c in cols], dtype=float)
            if np.isfinite(base).all() and np.isfinite(q).all():
                real = personas[personas["persona_id"] == r.persona_id].iloc[0][[f"{it}_{c}" for c in codes]].to_numpy(float)
                rows.append({"model": m, "method": meth, "item": it, "lang": lang, "persona_id": r.persona_id,
                             "jsd_persona": jensenshannon(real, q / q.sum(), base=2),
                             "jsd_no_persona": jensenshannon(real, base / base.sum(), base=2)})
    out = pd.DataFrame(rows)
    out["gain"] = out["jsd_no_persona"] - out["jsd_persona"]
    return out


def agreement_between_methods(grid: pd.DataFrame, questions: dict, a: str, b: str) -> pd.DataFrame:
    """Jensen-Shannon distance between the distributions of two methods for the same model, persona, item, language."""
    keys = ["model", "persona_id", "item", "lang"]
    rows = []
    for key, g in grid.groupby(keys):
        ga, gb = g[g["method"] == a], g[g["method"] == b]
        if ga.empty or gb.empty:
            continue
        codes = questions[key[2]]["codes"]
        qa = ga.iloc[0][[f"p_{c}" for c in codes]].to_numpy(float)
        qb = gb.iloc[0][[f"p_{c}" for c in codes]].to_numpy(float)
        if np.isfinite(qa).all() and np.isfinite(qb).all():
            rows.append({**dict(zip(keys, key)), "jsd_between_methods": jensenshannon(qa / qa.sum(), qb / qb.sum(), base=2)})
    return pd.DataFrame(rows)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "grid_results.csv"
    questions = load_questions()
    grid = pd.read_csv(path)
    grid["model"] = grid["model"].str.split("/").str[-1]
    d = add_distances(grid, questions)
    d["unreliable_order"] = d["order_tv"] > RELIABLE_ORDER_TV
    d.to_csv(RESULTS / "grid_with_distances.csv", index=False)

    pd.set_option("display.width", 200)
    tables = {}

    tables["by_model_method"] = d.groupby(["model", "method"])[["jsd_to_survey", "order_tv", "valid_mass_min"]].mean().round(3)
    tables["by_item_method"] = d.groupby(["item", "method"])["jsd_to_survey"].mean().round(2).unstack()
    tables["by_item_lang_method"] = d.groupby(["item", "lang", "method"])["jsd_to_survey"].mean().round(2).unstack()
    tables["by_model_item_method"] = d.groupby(["model", "item", "method"])["jsd_to_survey"].mean().round(2).unstack()
    tables["order_effect"] = d.groupby(["model", "method", "item"])["order_tv"].mean().round(2).unstack()
    tables["share_unreliable_order"] = d.groupby(["model", "method"])["unreliable_order"].mean().round(2)

    gain = persona_gain(d, questions)
    gain.to_csv(RESULTS / "persona_gain_rows.csv", index=False)
    tables["persona_gain_by_model_method"] = gain.groupby(["model", "method"])[["jsd_persona", "jsd_no_persona", "gain"]].mean().round(3)
    tables["persona_gain_by_item"] = gain.groupby(["item", "method"])["gain"].mean().round(3).unstack()

    # does the model's average answer follow the real differences between the 18 groups?
    corr = []
    for (m, meth, it, lang), g in d[d["persona_id"] != "p00"].groupby(["model", "method", "item", "lang"]):
        ok = g.dropna(subset=["model_mean"])
        corr.append({"model": m, "method": meth, "item": it, "lang": lang,
                     "corr_mean": np.corrcoef(ok["real_mean"], ok["model_mean"])[0, 1] if ok["model_mean"].std() > 1e-9 else np.nan,
                     "sd_model_mean": ok["model_mean"].std(), "sd_real_mean": ok["real_mean"].std()})
    tables["tracking_groups"] = pd.DataFrame(corr).round(2)

    ag = pd.concat([agreement_between_methods(d, questions, "prob", "prompt").assign(pair="prob vs prompt"),
                    agreement_between_methods(d, questions, "prob", "prompt_greedy").assign(pair="prob vs prompt_greedy"),
                    agreement_between_methods(d, questions, "prompt", "prompt_greedy").assign(pair="prompt vs prompt_greedy")])
    tables["agreement_between_methods"] = ag.groupby(["model", "pair"])["jsd_between_methods"].mean().round(3).unstack()
    tables["agreement_by_item"] = ag.groupby(["pair", "item", "model"])["jsd_between_methods"].mean().round(2).unstack()

    for name, t in tables.items():
        t.to_csv(RESULTS / f"summary_{name}.csv")
        print(f"\n=== {name} ===\n{t.to_string()}")


if __name__ == "__main__":
    main()
