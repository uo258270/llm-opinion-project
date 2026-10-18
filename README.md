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
| P33 | Sexual relations between two adults of the same sex (4) | divisive / LGTBI |

**Personas:** sex × age (18–34 / 35–54 / 55+) × self-rated religiosity (P58, collapsed to
religious / neither / non-religious). All 18 cells have n ≥ 30 and cover 98.9% of the sample (1,714 of 1,733).
A no-persona baseline (`p00`) is also prompted, and compared with the whole-sample distribution in
`data/processed/population_distributions.csv`.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/load_cis3194.py        # download CIS microdata into data/raw/ (not in the repo)
.venv/bin/python src/build_personas.py      # -> data/processed/personas.csv, persona_distributions.csv, population_distributions.csv
.venv/bin/python src/build_prompts.py       # -> data/processed/prompts.jsonl (two option orders)
.venv/bin/python src/build_prompts.py --orders cyclic   # -> prompts_cyclic.jsonl (all rotations of the options)
```

Then run `notebooks/run_models_colab.ipynb` on Google Colab (T4 GPU)
([Colab link](https://colab.research.google.com/drive/1PNIcmF7nDHti6RlflZt1VuB8oNT8JHwA?usp=sharing)).
Set `MODEL_KEY` to `mistral` (Mistral-7B-Instruct-v0.3, France) or `qwen` (Qwen2.5-7B-Instruct, China).
It reads the next-token probabilities of the answer labels in one forward pass, averages the
option orders, and writes `model_distributions_*.csv` and `model_prompt_level_*.csv`
(put them in `data/results/`). `notebooks/run_mistral_colab.ipynb` is the first Mistral-only version.

```bash
.venv/bin/python src/compare_label_variants.py   # numbers vs letters: order effects (first Mistral run)
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
