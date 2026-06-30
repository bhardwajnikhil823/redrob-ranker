# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A fast, **explainable, CPU-only** ranker that selects the top-100 candidates for the
*"Senior AI Engineer — Founding Team"* job description from a 100,000-candidate pool.

It is built around one idea the JD states explicitly:

> *"The right answer involves reasoning about the gap between what the JD **says**
> and what the JD **means**. A candidate who has all the AI keywords listed as
> skills but whose title is 'Marketing Manager' is not a fit."*

So this system scores **what candidates demonstrably did** (career-history text,
role titles) and treats the `skills` array — the field a keyword-stuffer inflates —
as weak, corroborating evidence at most.

---

## Reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv      # -> "Submission is valid."
```

Runs in **~30 s** on a 16 GB CPU-only machine for the full 100K pool — no GPU, no
network. Also accepts `candidates.jsonl.gz` and the pretty-printed
`sample_candidates.json` (use `--top-k` < pool size for small samples).

Optional sanity checks (honeypot rate, distributions):

```bash
python diagnostics.py --candidates ./candidates.jsonl
```

---

## Sandbox / demo (`app.py`)

A small Streamlit front-end that runs the **exact same** ranking code on a small
sample and produces a downloadable, spec-compliant CSV — this is the mandatory
sandbox from submission_spec Section 10.5.

```bash
python -m streamlit run app.py        # then open the printed Local URL
```

**Deploy for free** (gives you the `sandbox_link` for your metadata):

- **Streamlit Community Cloud** — push this repo to GitHub, go to
  share.streamlit.io, pick the repo, set the main file to `app.py`. It installs
  from `requirements.txt` automatically.
- **Hugging Face Spaces** — create a *Streamlit* Space and push this repo to it.

The app defaults to the bundled `sample_candidates.json` and also accepts an
uploaded `.json` / `.jsonl` / `.jsonl.gz`, so organizers can verify it end to end
on a small sample within the compute budget.

---

## How it works

For each candidate a **base score in [0, 1]** is computed as a weighted sum of
nine interpretable components, then multiplied by a **behavioural availability
modifier** derived from `redrob_signals`. Detected honeypots are forced to the
bottom.

| Component | Weight | What it rewards |
|---|---|---|
| `substance` | 0.24 | Hands-on **retrieval / ranking / recommendation / search** work found in the candidate's own *career descriptions* (double-weighted core terms). This is the main anti-stuffer signal. |
| `semantic` | 0.20 | TF-IDF (1–2 gram) cosine similarity between a *distilled* JD query and the candidate's summary + career text. |
| `title` | 0.16 | Tiered role gate: AI/ML roles 1.0 → general SWE 0.7 → adjacent eng 0.42 → unrelated 0.25 → **non-technical 0.10**. |
| `experience` | 0.10 | 5–9 year band (ideal 6–8), soft falloff outside it. |
| `product` | 0.09 | Product company vs **pure IT-services/consulting** career (the JD's explicit do-not-want). |
| `domain` | 0.07 | **NLP/IR** (good) vs computer-vision/speech-only (explicit negative). |
| `location` | 0.08 | Pune/Noida → NCR → metro → other-India → **outside-India** (no visa sponsorship). |
| `evaluation` | 0.04 | Explicit ranking-evaluation experience (NDCG/MRR/MAP/A-B testing). |
| `education` | 0.02 | Institution tier (minor). |

**Availability modifier (≈ 0.5–1.1×):** combines `recruiter_response_rate`,
recency of `last_active_date`, `open_to_work_flag`, `interview_completion_rate`
and `profile_completeness_score`. A perfect-on-paper candidate who hasn't logged
in for months and never replies is down-weighted — they are not actually hireable.

### Why TF-IDF and not an LLM / transformer embedding?

The compute budget (5 min, 16 GB, **CPU only, no network**) rules out per-candidate
LLM calls, and the JD itself frames this as a production-realism test. A sparse
TF-IDF model over the *career text* captures topical overlap with the role in
seconds and — crucially — naturally ignores a stuffed skills list because that
field is excluded from the document text. Semantic similarity is intentionally
only 20% of the score; the structured, auditable components do the heavy lifting
and make every ranking defensible at the Stage-5 interview.

### Honeypot handling

`honeypot_report()` flags **logically impossible** profiles — a single role longer
than the entire declared career, total tenure ≫ years of experience, reversed or
future dates, education that ends before it starts, and the designed
"many expert skills with 0 months used" pattern. Flagged profiles are multiplied
by `0.02` and pushed out of contention. On the released pool this flags **22**
candidates and yields **0 honeypots in the top 100** (limit: 10%).

> Note: a skill *duration* exceeding the professional career is **not** treated as
> impossible — people learn skills in college and on side projects — an early
> over-aggressive version of this check incorrectly flagged ~9% of the pool; see
> the git history for the fix.

### Keyword-stuffer mismatch guard

Beyond the title gate, `stuffer_penalty()` directly targets the JD's named trap —
*"all the AI keywords listed as skills but the title is 'Marketing Manager'"*. It
cross-checks the number of **claimed** advanced/expert AI skills against the
**demonstrated** AI work in the career text (core build-terms + ML-terms). When
claims are high but demonstration is essentially absent and the title is
non-technical/adjacent, the composite score is multiplied by `0.30–0.75`. A
genuine "Tier-5" candidate (non-AI title but a real ML career) is **never**
penalised, because the guard only fires when demonstrated work is missing. On the
released pool this flags **28** stuffers — all already outside the top 100, which
is exactly the intended no-op-on-the-good-ones behaviour.

---

## Tests

```bash
python -m unittest test_rank -v
```

`test_rank.py` covers honeypot detection (including a **regression test** for the
skill-duration-vs-career false positive that once flagged ~9% of the pool),
tiered title scoring, the experience/location/product components, the
keyword-stuffer guard (including Tier-5 protection), and an end-to-end ranking +
CSV-format check. A skipped-by-default integration test runs the full pipeline on
`sample_candidates.json` when present.

---

## Reasoning column

`build_reasoning()` produces a grounded, 1–2 sentence justification per candidate
using **only facts present in the profile** (years, current title/company, the
strongest build-evidence phrase from their career text, top *actually-used*
skills, location, and live signal values). It states **honest concerns**
(out-of-band experience, outside-India, services-only career, weak availability,
CV/speech lean) so the tone matches the rank. Skills are cited as demonstrated
"depth" only when the role/substance backs it up, otherwise neutrally as "lists".

---

## Files

| File | Purpose |
|---|---|
| `rank.py` | The ranker. Single command produces `submission.csv`. |
| `diagnostics.py` | Honeypot-rate and distribution sanity checks. |
| `app.py` | Streamlit sandbox/demo (the mandatory hosted demo). |
| `test_rank.py` | Unit + regression tests (`python -m unittest test_rank`). |
| `requirements.txt` | Pinned dependencies. |
| `submission_metadata.yaml` | Portal metadata mirror (fill in your team details). |
| `validate_submission.py` | Official format validator (provided in the bundle). |

## Compute environment

CPU-only, no GPU, no network during ranking. Peak memory comfortably under 16 GB
(the 465 MB pool plus a sparse TF-IDF matrix). Total runtime ≈ 30 s for 100K
candidates.
