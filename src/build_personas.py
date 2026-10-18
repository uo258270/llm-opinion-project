"""Build the persona table for CIS 3194: sex x age group x self-rated religiosity.

For every persona cell with at least MIN_N respondents, compute the weighted response
distribution over the substantive categories of each target item (N.S./N.C. excluded
and the distribution renormalised).

Variables (confirmed against cues3194.pdf / FT3194.pdf):
    P66   sex                           1 Hombre, 2 Mujer
    EDAD  age in years (18-96)
    P58   self-rated religiosity (ISSP scale), 1 Extremadamente religiosa ... 4 Ni religiosa, ni no
          religiosa ... 7 Extremadamente no religiosa, 98 N.S., 99 N.C. -- collapsed to 3 levels.
          (Not P90, which is the CIS denominational classification: Católico/a / Otra / Ninguna.)
    PESO2 survey weight (the technical sheet says PESO2 must be used for the whole sample;
          PESO1 is the selection probability only, before the response-rate correction)
    P42   belief in God                 1-6 substantive, 8/9 non-substantive
    P35   "La responsabilidad del marido consiste en ganar dinero; la de la mujer en
          cuidar de la casa y de la familia"   1-5 substantive (1 = Muy de acuerdo)
    P11   generalised interpersonal trust      1-4 substantive
    P33   sexual relations between two adults of the same sex: 1 Siempre está mal ...
          4 Nunca está mal (LGTBI item)         1-4 substantive

Outputs (data/processed/):
    personas.csv                 one row per persona: cell counts + wide distributions
    persona_distributions.csv    long format: persona x item x category
"""

from __future__ import annotations

import pandas as pd

from load_cis3194 import PROJECT_ROOT, load_cis3194

OUT_DIR = PROJECT_ROOT / "data" / "processed"

MIN_N = 30
WEIGHT = "PESO2"

SEX = {1: "Hombre", 2: "Mujer"}
RELIGIOSITY = {1: "Religiosa", 2: "Religiosa", 3: "Religiosa",
               4: "Ni religiosa ni no religiosa",
               5: "No religiosa", 6: "No religiosa", 7: "No religiosa"}  # 98/99 dropped

# Upper bounds inclusive.
AGE_BINS = [17, 34, 54, 200]
AGE_LABELS = ["18-34", "35-54", "55+"]

# item -> substantive codes (8 = N.S. and 9 = N.C. are excluded)
ITEMS = {
    "P42": [1, 2, 3, 4, 5, 6],
    "P35": [1, 2, 3, 4, 5],
    "P11": [1, 2, 3, 4],
    "P33": [1, 2, 3, 4],
}

PERSONA_KEYS = ["sex", "age_group", "religiosity"]


def add_persona_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recode sex, age, and religiosity; drop respondents missing any persona attribute."""
    out = df.assign(
        sex=df["P66"].map(SEX),
        age_group=pd.cut(df["EDAD"].where(df["EDAD"] != 99), AGE_BINS, labels=AGE_LABELS).astype("string"),
        religiosity=df["P58"].map(RELIGIOSITY),
    )
    return out.dropna(subset=PERSONA_KEYS)


def item_distribution(cell: pd.DataFrame, item: str) -> tuple[int, pd.Series]:
    """Unweighted valid n and weighted proportions over the item's substantive codes."""
    codes = ITEMS[item]
    valid = cell[cell[item].isin(codes)]
    props = valid.groupby(item)[WEIGHT].sum().reindex(codes, fill_value=0.0)
    return len(valid), props / props.sum()


def build_personas(df: pd.DataFrame, meta) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_persona_columns(df)

    wide_rows, long_rows = [], []
    for (sex, age, rel), cell in df.groupby(PERSONA_KEYS, sort=True):
        if len(cell) < MIN_N:
            continue
        row = {"sex": sex, "age_group": age, "religiosity": rel, "n": len(cell), "weighted_n": cell[WEIGHT].sum()}
        for item in ITEMS:
            n_valid, props = item_distribution(cell, item)
            row[f"{item}_n_valid"] = n_valid
            for code, p in props.items():
                row[f"{item}_{code}"] = p
                long_rows.append({
                    "sex": sex, "age_group": age, "religiosity": rel, "item": item, "code": code,
                    "label": meta.variable_value_labels[item][float(code)],
                    "n_valid": n_valid, "proportion": p,
                })
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide.insert(0, "persona_id", [f"p{i:02d}" for i in range(1, len(wide) + 1)])
    long = pd.DataFrame(long_rows).merge(wide[["persona_id", *PERSONA_KEYS]], on=PERSONA_KEYS)
    long = long[["persona_id", *PERSONA_KEYS, "item", "code", "label", "n_valid", "proportion"]]
    return wide, long


def main() -> None:
    df, meta = load_cis3194()
    wide, long = build_personas(df, meta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_DIR / "personas.csv", index=False)
    long.to_csv(OUT_DIR / "persona_distributions.csv", index=False)

    n_total = len(add_persona_columns(df))
    print(f"{len(wide)} personas with n >= {MIN_N}, covering {wide['n'].sum()}/{n_total} respondents")
    cols = ["persona_id", *PERSONA_KEYS, "n", *[f"{i}_n_valid" for i in ITEMS]]
    print(wide[cols].to_string(index=False))
    low = wide[(wide[[f"{i}_n_valid" for i in ITEMS]] < MIN_N).any(axis=1)]
    if len(low):
        print(f"\nNote: {len(low)} personas have < {MIN_N} valid answers on some item after excluding N.S./N.C.")
    print(f"\nWrote {OUT_DIR / 'personas.csv'} and {OUT_DIR / 'persona_distributions.csv'}")


if __name__ == "__main__":
    main()
