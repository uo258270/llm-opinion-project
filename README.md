# LLM opinion distributions vs. Spanish public opinion (CIS 3194)

Do open-weight LLMs reproduce the real distribution of Spanish public opinion on morally
divisive topics, compared with a neutral control, and does prompting in Spanish vs.
English change that?

**Data:** CIS Study 3194, *Redes Sociales (I) / Religión (III) (ISSP)*, fieldwork
Oct 2017 – Jan 2018, n = 1,733, weighted with `PESO2`.

| Item | Topic | Role |
|---|---|---|
| P42 | Belief in God (6 categories) | divisive / religious |
| P35 | "A man's job is to earn money; a woman's job is to look after the home and family" (5) | divisive / ideological |
| P11 | Generalised interpersonal trust (4) | neutral control |

**Personas:** sex × age (18–34 / 35–54 / 55+) × self-rated religiosity (P58, collapsed to
religious / neither / non-religious). All 18 cells have n ≥ 30.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/load_cis3194.py        # download CIS microdata into data/raw/ (not in the repo)
.venv/bin/python src/build_personas.py      # -> data/processed/personas.csv, persona_distributions.csv
.venv/bin/python src/build_prompts.py       # -> data/processed/prompts.jsonl, prompts_letters.jsonl
```

Then run `notebooks/run_mistral_colab.ipynb` on Google Colab (T4 GPU). It reads the
next-token probabilities of the answer labels in one forward pass, averages the original
and reversed option orders, and writes `model_distributions_*.csv` and
`model_prompt_level_*.csv` (put them in `data/results/`). `VARIANT` switches between
numbered (1, 2, 3…) and lettered (A, B, C…) options.

```bash
.venv/bin/python src/compare_label_variants.py   # numbers vs letters: order effects
```

## Layout

```
src/                 data download, persona table, prompts, analysis
notebooks/           Colab notebook for model runs
data/processed/      persona table, real distributions, prompts
data/results/        model outputs
```

The CIS microdata is not redistributed here. `src/load_cis3194.py` downloads it from
the [CIS study page](https://www.cis.es/es/estudios/redes-sociales-i-/-religion-iii-issp-?origen=estudio&idEstudio=14366).
