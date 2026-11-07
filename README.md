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

Questions and answer options use the literal wording of the questionnaires: the CIS 3194
questionnaire and show cards in Spanish, the ISSP 2018 Religion IV questionnaire in English
(`config/questions.json`).

**Personas:** sex × age (18–34 / 35–54 / 55+) × self-rated religiosity (P58, collapsed to
religious / neither / non-religious). Cells with at least 30 respondents are kept: 18 personas
(the smallest has n = 35) covering 98.9% of the sample (1,714 of 1,733).
A no-persona baseline (`p00`) is also prompted and compared with the whole-sample distribution in
`data/processed/population_distributions.csv`.

## Methods

Each prompt is run with three methods:

| Method | How the answer distribution is obtained |
|---|---|
| `prob` | One forward pass: probability that the reply starts with each option number, renormalised |
| `prompt_greedy` | Temperature 0, one answer, the number is parsed |
| `prompt` | 30 samples at temperature 1 (no top-k, no top-p, no repetition penalty), answers are parsed and counted |

Models: Mistral-7B-Instruct-v0.3 (France) and Qwen2.5-7B-Instruct (China), loaded in 8-bit.
Every prompt is run with the options in the original and in reversed order, and the two
distributions are averaged (`order_tv` measures how much they differ).
The metric is the Jensen–Shannon distance (base 2) between the model's distribution and the
real, weighted survey distribution of the same group.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/load_cis3194.py        # download CIS microdata into data/raw/ (not in the repo)
.venv/bin/python src/build_personas.py      # -> data/processed/personas.csv, persona_distributions.csv, population_distributions.csv
.venv/bin/python src/build_prompts.py       # -> data/processed/prompts.jsonl (304 prompts, two option orders)
.venv/bin/python src/build_prompts.py --orders cyclic   # -> prompts_cyclic.jsonl (1,444 prompts, all rotations of the options)
```

Then run `notebooks/run_grid_colab.ipynb` on Google Colab (T4 GPU). It runs the full grid
(model × method × persona × question × language × option order), saves checkpoints to Google Drive,
and writes `prompt_level_<model>__<method>.csv` and `grid_results.csv`. Put `grid_results.csv` in
`data/results/` and run the analysis:

```bash
.venv/bin/python src/analyze_grid.py        # reads data/results/grid_results.csv
```

It writes `grid_with_distances.csv`, `persona_gain_rows.csv` and several `summary_*.csv` tables
(distance by model, method, question and language; persona gain; agreement between methods;
order effect; correlation between groups) to `data/results/`.

## Layout

```
config/              questions.json: question wording and options (Spanish and English)
src/                 data download, persona table, prompts, analysis
notebooks/           Colab notebook that runs the grid
data/processed/      persona table, real distributions, prompts
data/results/        model outputs and analysis tables
```

The CIS microdata is not redistributed here. `src/load_cis3194.py` downloads it from
the [CIS study page](https://www.cis.es/es/estudios/redes-sociales-i-/-religion-iii-issp-?origen=estudio&idEstudio=14366).
