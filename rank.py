#!/usr/bin/env python3
"""
Redrob Hackathon - Intelligent Candidate Discovery & Ranking
=============================================================
rank.py - produces a top-100 ranking CSV for the "Senior AI Engineer - Founding
Team" job description from a pool of up to 100,000 candidates.

Design philosophy (see README.md for the long version)
-------------------------------------------------------
The dataset is built to punish naive keyword matching. The JD itself says:

    "The right answer involves reasoning about the gap between what the JD says
     and what the JD means. A candidate who has all the AI keywords listed as
     skills but whose title is 'Marketing Manager' is not a fit."

So this ranker deliberately:
  * Scores *demonstrated work* in `career_history` descriptions and the summary,
    NOT the raw `skills` list (the skills array is the keyword-stuffer trap).
  * Detects honeypots / internally-inconsistent profiles and forces them down.
  * Penalises pure-services/consulting careers and non-India locations (the JD
    does not sponsor visas and wants Pune/Noida-reachable people).
  * Applies a multiplicative *availability* modifier from `redrob_signals`
    (a perfect-on-paper candidate who never replies is not actually hireable).

All compute is CPU-only, no network, and runs well inside the 5 min / 16 GB
budget for 100K candidates.

Usage
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Supports plain `.jsonl`, gzipped `.jsonl.gz`, and a pretty-printed `.json`
array (e.g. sample_candidates.json) for quick local iteration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sys
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# 1. Job-description model
# ---------------------------------------------------------------------------
# A *distilled* query is used for semantic matching instead of the raw JD text.
# The raw JD is full of meta-commentary ("Let's be honest about this role...")
# that only adds noise to TF-IDF. This string captures what the role actually
# needs, in the vocabulary a strong candidate would use to describe their work.
JD_QUERY = """
senior ai engineer applied machine learning production embeddings retrieval
ranking system search relevance recommendation recommender semantic search
hybrid search vector database vector search faiss pinecone weaviate qdrant
milvus opensearch elasticsearch nearest neighbour learning to rank
information retrieval nlp natural language processing transformers llm
re-ranking re ranker fine-tuning lora evaluation ndcg mrr map ab testing
offline online metrics recall precision feature pipeline model serving
inference latency python product company shipped end to end at scale
matching candidate jd marketplace recruiting hr tech
""".strip()

# Core "what we actually built" terms. Hitting these in a candidate's
# *career descriptions* is the strongest positive signal in the whole ranker.
CORE_BUILD_TERMS = [
    "recommend", "recommender", "ranking", "rank ", "learning to rank",
    "retrieval", "information retrieval", "search relevance", "relevance",
    "semantic search", "embedding", "vector search", "vector database",
    "nearest neighbor", "nearest neighbour", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "elastic search",
    "personalization", "personalisation", "candidate generation",
    "matching engine", "similarity search", "ann ", "re-rank", "rerank",
]
# General ML/IR evidence - supportive but weaker than the core terms above.
ML_TERMS = [
    "machine learning", "deep learning", "neural", "pytorch", "tensorflow",
    "scikit", "transformer", "bert", "llm", "rag", "fine-tun", "nlp",
    "natural language", "model serving", "inference", "feature pipeline",
    "feature store", "mlops", "embeddings", "classification", "regression",
    "xgboost", "gradient boosting", "a/b test", "ab test", "ndcg", "mrr",
]
# Evaluation-maturity terms (the JD calls this out as a hard requirement).
EVAL_TERMS = ["ndcg", "mrr", "map ", "a/b test", "ab test", "offline", "online metric",
              "evaluation framework", "recall@", "precision@", "ranking metric"]

# NLP/IR is wanted; CV/speech/robotics-as-primary is an explicit negative.
NLP_TERMS = ["nlp", "natural language", "information retrieval", "search",
             "ranking", "recommendation", "embedding", "text", "semantic",
             "transformer", "bert", "llm", "retrieval"]
CV_SPEECH_TERMS = ["computer vision", "image classification", "object detection",
                   "segmentation", "cnn", "opencv", "speech recognition", "asr",
                   "text to speech", "tts", "robotics", "slam", "lidar", "ocr",
                   "facial", "pose estimation"]

# AI/ML buzzwords as they appear in the *skills* array. A keyword stuffer loads
# these up; we cross-check the count against demonstrated work in career text.
AI_SKILL_TERMS = [
    "rag", "llm", "large language model", "gpt", "generative ai", "prompt",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "vector", "embedding",
    "transformer", "bert", "fine-tun", "lora", "peft", "langchain", "llamaindex",
    "semantic search", "retrieval", "rerank", "sentence transformer",
    "hugging face", "huggingface", "recommendation", "learning to rank",
    "nlp", "deep learning", "pytorch", "tensorflow", "machine learning",
]

# Title tiers. A keyword stuffer wears a title that does NOT match its skills
# list, so the title is a strong gate - but we tier it, because the JD's "Tier
# 5" candidate (a backend/data engineer who actually built ML systems) should
# still be reachable via the `substance` component even on a non-ML title.
#
# Tier A - directly AI/ML/IR roles (full credit).
AI_ML_TITLE_TERMS = [
    "machine learning", "ml engineer", "ml scientist", "mle", "ai engineer",
    "ai/ml", "applied scientist", "data scientist", "research engineer",
    "research scientist", "nlp engineer", "search engineer", "relevance engineer",
    "deep learning", "recommendation", "ai researcher", "ml researcher",
]
# Tier B - general software roles a strong candidate could pivot from.
SWE_TITLE_TERMS = [
    "software engineer", "backend engineer", "back-end engineer", "data engineer",
    "founding engineer", "platform engineer", "staff engineer", "principal engineer",
    "sde", "software developer", "full stack", "fullstack", "full-stack",
    "engineering manager", "mlops", "data architect", "solutions architect",
]
# Tier C - engineering but not ML-adjacent (low base; substance can rescue).
ADJACENT_ENG_TITLE_TERMS = [
    "frontend", "front-end", "front end", "mobile", "android", "ios ",
    "devops", "qa ", "test engineer", "sdet", "automation", "support engineer",
    "salesforce", "sap ", "wordpress", "drupal", "ui ", "ux ",
]
# Tier D - unrelated engineering disciplines.
UNRELATED_ENG_TITLE_TERMS = [
    "mechanical", "civil", "electrical", "chemical", "hardware", "embedded",
    "network engineer", "biomedical", "industrial engineer",
]
# Decoy / non-technical titles - the keyword-stuffer trap usually wears one.
NON_TECH_TITLE_TERMS = [
    "hr ", "human resource", "recruit", "talent acquisition", "marketing",
    "sales", "graphic design", "designer", "content writer", "copywrit",
    "seo", "account manager", "customer success", "social media", "brand ",
    "community manager", "business development", "operations manager",
    "office manager", "administrator", "receptionist", "accountant",
    "project manager", "program manager", "consultant",
]

# Indian IT-services / consulting firms. A career spent *entirely* here is an
# explicit do-not-want; current-services-with-prior-product is fine.
SERVICES_FIRMS = [
    "tata consultancy", "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "lti ",
    "l&t infotech", "larsen", "mphasis", "hexaware", "syntel", "igate",
    "birlasoft", "zensar", "nttdata", "ntt data", "dxc", "atos", "ibm global",
    "persistent systems", "cybage", "coforge", "virtusa",
]

# Location buckets (lower-cased substring match against location + country).
TARGET_CITIES = ["pune", "noida"]                       # JD's home bases
NCR_CITIES = ["delhi", "gurgaon", "gurugram", "ghaziabad", "faridabad", "ncr"]
METRO_CITIES = ["hyderabad", "mumbai", "bangalore", "bengaluru", "chennai"]

# ---------------------------------------------------------------------------
# 2. Component weights (base score in [0, 1] before the availability modifier)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic":   0.20,   # TF-IDF cosine vs the distilled JD query
    "title":      0.16,   # is the role genuinely technical?
    "substance":  0.24,   # did they *build* retrieval/ranking/reco systems?
    "experience": 0.10,   # 5-9 yrs band (ideal 6-8)
    "product":    0.09,   # product company vs pure services/consulting
    "domain":     0.07,   # NLP/IR (good) vs CV/speech-only (bad)
    "location":   0.08,   # Pune/Noida reachable, India, visa-free
    "evaluation": 0.04,   # explicit evaluation-framework experience
    "education":  0.02,   # institution tier (minor)
}

# ---------------------------------------------------------------------------
# 3. Loading
# ---------------------------------------------------------------------------
def _open_any(path: str) -> io.TextIOBase:
    """Open .jsonl, .jsonl.gz (by extension or magic bytes) as UTF-8 text."""
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if path.endswith(".gz") or magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """Load candidates from .jsonl/.jsonl.gz (one object per line) or a .json array."""
    # Pretty-printed JSON array (e.g. sample_candidates.json).
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    out: List[Dict[str, Any]] = []
    with _open_any(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# ---------------------------------------------------------------------------
# 4. Small parsing helpers
# ---------------------------------------------------------------------------
def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _lower(s: Any) -> str:
    return s.lower() if isinstance(s, str) else ""


def _count_terms(text: str, terms: Iterable[str]) -> int:
    """Number of *distinct* terms from `terms` present in `text`."""
    return sum(1 for t in terms if t in text)


def career_text(c: Dict[str, Any]) -> str:
    """Concatenated *evidence* text: summary + every role title + description.

    Deliberately excludes the `skills` array - that is exactly the field a
    keyword stuffer inflates. We judge candidates by what they describe doing.
    """
    parts: List[str] = []
    prof = c.get("profile", {}) or {}
    parts.append(_lower(prof.get("headline")))
    parts.append(_lower(prof.get("summary")))
    parts.append(_lower(prof.get("current_title")))
    parts.append(_lower(prof.get("current_industry")))
    for role in c.get("career_history", []) or []:
        parts.append(_lower(role.get("title")))
        parts.append(_lower(role.get("description")))
        parts.append(_lower(role.get("industry")))
    return " ".join(p for p in parts if p)

# ---------------------------------------------------------------------------
# 5. Honeypot / internal-consistency detection
# ---------------------------------------------------------------------------
def honeypot_report(c: Dict[str, Any], today: date) -> Tuple[bool, List[str]]:
    """Return (is_honeypot, reasons). `is_honeypot` is True only on a *hard*,
    logically-impossible inconsistency; soft oddities are reported but do not
    by themselves disqualify."""
    reasons: List[str] = []
    hard = False

    prof = c.get("profile", {}) or {}
    yoe = prof.get("years_of_experience")
    yoe = float(yoe) if isinstance(yoe, (int, float)) else None
    yoe_months = (yoe * 12.0) if yoe is not None else None

    roles = c.get("career_history", []) or []
    total_role_months = 0
    for role in roles:
        dm = role.get("duration_months")
        dm = int(dm) if isinstance(dm, (int, float)) else 0
        total_role_months += dm

        # A single role longer than the whole declared career is impossible.
        if yoe_months is not None and dm > yoe_months + 6:
            hard = True
            reasons.append(
                f"role '{role.get('title','?')}' lasts {dm} mo but total experience "
                f"is only ~{int(yoe_months)} mo")

        sd, ed = _parse_date(role.get("start_date")), _parse_date(role.get("end_date"))
        if sd and ed and sd > ed:
            hard = True
            reasons.append(f"role '{role.get('title','?')}' ends before it starts")
        if sd and sd > today:
            hard = True
            reasons.append(f"role '{role.get('title','?')}' starts in the future")

        # Claimed tenure longer than the actual date span is impossible - this is
        # the "8 years of experience at a company founded 3 years ago" honeypot
        # expressed in the data (duration_months far exceeds the months actually
        # spanned by start_date -> end_date/today). A 6-month buffer absorbs
        # rounding so only genuine impossibilities are flagged.
        role_end = ed if ed else (today if role.get("is_current") else None)
        if sd and role_end and role_end >= sd:
            span_months = (role_end - sd).days / 30.44
            if dm > span_months + 6:
                hard = True
                reasons.append(
                    f"role '{role.get('title','?')}' claims {dm} mo but its dates "
                    f"span only ~{int(span_months)} mo")

        if role.get("is_current") and role.get("end_date"):
            reasons.append("role flagged current but has an end date")

    # Total tenure far exceeding declared experience (allow some overlap).
    if yoe_months is not None and yoe_months > 0 and total_role_months > yoe_months * 2.0 + 12:
        hard = True
        reasons.append(
            f"career history sums to {total_role_months} mo vs ~{int(yoe_months)} mo declared")

    # Skills: the designed honeypot pattern is "expert/advanced proficiency in
    # many skills with 0 months used". (Note: a skill *duration* exceeding the
    # professional career is NOT impossible - people learn skills in college or
    # on side projects - so that is deliberately NOT treated as inconsistent.)
    high_prof_zero_dur = 0
    for sk in c.get("skills", []) or []:
        prof_level = _lower(sk.get("proficiency"))
        dur = sk.get("duration_months")
        dur = int(dur) if isinstance(dur, (int, float)) else None
        if prof_level in ("advanced", "expert") and dur == 0:
            high_prof_zero_dur += 1
    if high_prof_zero_dur >= 4:
        reasons.append(f"{high_prof_zero_dur} advanced/expert skills with 0 months used")
        if high_prof_zero_dur >= 7:
            hard = True

    # Education sanity.
    for ed in c.get("education", []) or []:
        sy, ey = ed.get("start_year"), ed.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            hard = True
            reasons.append("education ends before it starts")

    return hard, reasons

# ---------------------------------------------------------------------------
# 6. Structured scoring components (each returns a value in [0, 1])
# ---------------------------------------------------------------------------
def score_title(c: Dict[str, Any]) -> Tuple[float, str]:
    """Tiered title gate. Non-technical titles score very low (the stuffer
    trap); only genuine AI/ML roles get full credit. General SWE sits in the
    middle and relies on the `substance` component to climb."""
    title = _lower(c.get("profile", {}).get("current_title"))
    if not title:
        return 0.40, "unknown"
    # Non-technical decoy wins only if there is no real ML title present.
    if any(t in title for t in NON_TECH_TITLE_TERMS) and not any(
            t in title for t in AI_ML_TITLE_TERMS):
        return 0.10, "non_tech"
    if any(t in title for t in AI_ML_TITLE_TERMS):
        return 1.00, "ai_ml"
    if any(t in title for t in SWE_TITLE_TERMS):
        return 0.70, "swe"
    if any(t in title for t in ADJACENT_ENG_TITLE_TERMS):
        return 0.42, "adjacent_eng"
    if any(t in title for t in UNRELATED_ENG_TITLE_TERMS):
        return 0.25, "unrelated_eng"
    # Generic "engineer"/"developer"/"scientist"/"architect" with no qualifier.
    if any(t in title for t in ("engineer", "developer", "scientist", "architect")):
        return 0.55, "generic_tech"
    return 0.35, "other"


def score_substance(text: str) -> Tuple[float, int, int]:
    core = _count_terms(text, CORE_BUILD_TERMS)
    ml = _count_terms(text, ML_TERMS)
    raw = 2.0 * core + 1.0 * ml
    return 1.0 - math.exp(-raw / 5.0), core, ml


def score_experience(yoe: Optional[float]) -> float:
    if yoe is None:
        return 0.4
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe < 6 or 8 < yoe <= 9:
        return 0.9
    if 4 <= yoe < 5 or 9 < yoe <= 10:
        return 0.75
    if 3 <= yoe < 4 or 10 < yoe <= 12:
        return 0.55
    if yoe < 3:
        return 0.30
    return 0.45  # > 12


def score_product(c: Dict[str, Any]) -> Tuple[float, float]:
    """Returns (score, services_fraction). Pure-services career is penalised
    hard; current-services-with-prior-product is only lightly penalised."""
    roles = c.get("career_history", []) or []
    if not roles:
        return 0.6, 0.0
    flags = [any(f in _lower(r.get("company")) for f in SERVICES_FIRMS) for r in roles]
    frac = sum(flags) / len(flags)
    if frac >= 0.999:
        return 0.25, frac                  # entire career in services
    current_services = bool(flags and flags[0])
    if current_services and frac < 0.999:
        return 0.70, frac                  # at a services firm now, product before
    return 1.0 - 0.55 * frac, frac


def score_domain(text: str) -> Tuple[float, int, int]:
    nlp = _count_terms(text, NLP_TERMS)
    cv = _count_terms(text, CV_SPEECH_TERMS)
    if nlp == 0 and cv == 0:
        return 0.5, nlp, cv
    diff = nlp - cv
    return 1.0 / (1.0 + math.exp(-0.6 * diff)), nlp, cv


def score_location(c: Dict[str, Any]) -> Tuple[float, str]:
    prof = c.get("profile", {}) or {}
    loc = _lower(prof.get("location"))
    country = _lower(prof.get("country"))
    sig = c.get("redrob_signals", {}) or {}
    relocate = bool(sig.get("willing_to_relocate"))

    in_india = ("india" in country) or any(
        city in loc for city in TARGET_CITIES + NCR_CITIES + METRO_CITIES)
    if not in_india and country and country != "india":
        # No visa sponsorship - outside India is a hard practical blocker.
        return (0.30 if relocate else 0.15), "outside_india"
    if any(city in loc for city in TARGET_CITIES):
        return 1.0, "target"
    if any(city in loc for city in NCR_CITIES):
        return 0.90, "ncr"
    if any(city in loc for city in METRO_CITIES):
        return 0.82, "metro"
    return (0.80 if relocate else 0.55), "india_other"


def score_education(c: Dict[str, Any]) -> float:
    best = 0.5
    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45,
                "unknown": 0.5}
    for ed in c.get("education", []) or []:
        best = max(best, tier_map.get(_lower(ed.get("tier")), 0.5))
    return best


def score_evaluation(text: str) -> float:
    return min(1.0, _count_terms(text, EVAL_TERMS) / 2.0)


def count_claimed_ai_skills(c: Dict[str, Any]) -> int:
    """How many AI/ML skills the candidate *claims* at advanced/expert level."""
    claimed = 0
    for sk in c.get("skills", []) or []:
        name = _lower(sk.get("name"))
        prof = _lower(sk.get("proficiency"))
        if name and prof in ("advanced", "expert") and any(
                t in name for t in AI_SKILL_TERMS):
            claimed += 1
    return claimed


def stuffer_penalty(title_kind: str, claimed_ai: int, demonstrated: int
                    ) -> Tuple[float, bool]:
    """Penalise the JD's named trap: a profile that *lists* lots of AI skills
    but whose title and career text show no supporting work. Returns a
    multiplicative factor in (0, 1] and an `is_stuffer` flag.

    `demonstrated` = distinct core build-terms + ML-terms found in the career
    text (i.e. evidence of real work, not the skills list). The penalty only
    fires when claims are high AND demonstration is essentially absent, so a
    genuine "Tier-5" candidate (non-AI title but real ML career) is never hit.
    """
    if claimed_ai >= 3 and demonstrated <= 1:
        if title_kind == "non_tech":
            return 0.30, True
        if title_kind in ("adjacent_eng", "unrelated_eng"):
            return 0.55, True
        if title_kind == "generic_tech":
            return 0.75, True
    if title_kind == "non_tech" and claimed_ai >= 1 and demonstrated == 0:
        return 0.50, True
    return 1.0, False


# ---------------------------------------------------------------------------
# 7. Behavioural availability modifier (multiplies the base score)
# ---------------------------------------------------------------------------
def availability_modifier(c: Dict[str, Any], today: date) -> Tuple[float, Dict[str, Any]]:
    sig = c.get("redrob_signals", {}) or {}

    response = sig.get("recruiter_response_rate")
    response = float(response) if isinstance(response, (int, float)) else 0.3

    last_active = _parse_date(sig.get("last_active_date"))
    days_inactive = (today - last_active).days if last_active else 365
    if days_inactive <= 14:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 120:
        recency = 0.55
    elif days_inactive <= 240:
        recency = 0.35
    else:
        recency = 0.20

    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.7
    interview = sig.get("interview_completion_rate")
    interview = float(interview) if isinstance(interview, (int, float)) else 0.5
    completeness = sig.get("profile_completeness_score")
    completeness = (float(completeness) / 100.0
                    if isinstance(completeness, (int, float)) else 0.5)

    avail = (0.35 * response + 0.30 * recency + 0.15 * open_flag +
             0.10 * interview + 0.10 * completeness)
    modifier = 0.5 + 0.6 * avail                      # -> roughly [0.5, 1.1]
    info = {"response": response, "days_inactive": days_inactive,
            "open": bool(sig.get("open_to_work_flag")), "recency": recency}
    return modifier, info

# ---------------------------------------------------------------------------
# 8. Reasoning generation (grounded in real facts; varied; honest on concerns)
# ---------------------------------------------------------------------------
def top_real_skills(c: Dict[str, Any], n: int = 2) -> List[str]:
    """Highest-endorsed, actually-used skills - never invents anything."""
    skills = [s for s in (c.get("skills", []) or [])
              if isinstance(s.get("duration_months"), (int, float))
              and s.get("duration_months", 0) > 0]
    skills.sort(key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
                reverse=True)
    return [s.get("name") for s in skills[:n] if s.get("name")]


def find_evidence_phrase(text: str) -> Optional[str]:
    """Return a human-readable description of the strongest build-evidence found
    in the candidate's own career text (used in reasoning)."""
    # (substring to look for, readable phrase to print) - most specific first.
    ordered = [
        ("learning to rank", "learning-to-rank"),
        ("recommender", "recommender-system"),
        ("recommendation", "recommendation-system"),
        ("recommend", "recommendation-system"),
        ("semantic search", "semantic-search"),
        ("vector search", "vector-search"),
        ("vector database", "vector-search"),
        ("search relevance", "search-relevance"),
        ("relevance", "search-relevance"),
        ("retrieval", "retrieval"),
        ("ranking", "ranking"),
        ("rank ", "ranking"),
        ("personaliz", "personalization"),
        ("personalis", "personalization"),
        ("embedding", "embeddings"),
        ("faiss", "FAISS retrieval"),
        ("elasticsearch", "Elasticsearch search"),
        ("opensearch", "OpenSearch search"),
        ("matching engine", "matching-engine"),
        ("similarity search", "similarity-search"),
    ]
    for needle, phrase in ordered:
        if needle in text:
            return phrase
    return None


def build_reasoning(c: Dict[str, Any], feats: Dict[str, Any]) -> str:
    prof = c.get("profile", {}) or {}
    yoe = prof.get("years_of_experience")
    title = prof.get("current_title") or "professional"
    company = prof.get("current_company") or "their current company"
    skills = top_real_skills(c, 2)
    skills_txt = " and ".join(skills) if skills else None
    info = feats["avail"]
    evidence = find_evidence_phrase(feats["text"])

    # --- positive clause (varies with what's actually strong) ---
    yrs = f"{yoe:.1f} yrs" if isinstance(yoe, (int, float)) else "unstated tenure"
    lead = f"{title} with {yrs} at {company}"
    pos: List[str] = []
    if evidence and feats["substance"] >= 0.5:
        pos.append(f"career history shows hands-on {evidence} work (the JD's core need)")
    elif feats["title_kind"] == "ai_ml" and feats["substance"] >= 0.35:
        pos.append("genuine applied-ML background relevant to retrieval/ranking")
    if skills_txt:
        # Cite skills as demonstrated "depth" only when the role/substance backs
        # it up; otherwise state neutrally that they're listed (avoids implying
        # competence a likely keyword-stuffer hasn't shown).
        credible = (feats["title_kind"] in ("ai_ml", "swe", "generic_tech")
                    and feats["substance"] >= 0.30)
        pos.append((f"depth in {skills_txt}" if credible
                    else f"lists {skills_txt}"))
    if feats["loc_kind"] in ("target", "ncr", "metro"):
        loc = prof.get("location")
        pos.append(f"{loc}-based (matches the Pune/Noida footprint)")
    if info["response"] >= 0.6 and info["days_inactive"] <= 30:
        pos.append(f"strongly engaged (response rate {info['response']:.2f}, "
                   f"active {info['days_inactive']}d ago)")

    # --- concern clause (be honest; this is checked at Stage 4) ---
    concerns: List[str] = []
    if feats["honeypot"]:
        concerns.append("profile has internal inconsistencies")
    if feats.get("is_stuffer") and feats["title_kind"] != "non_tech":
        concerns.append(f"lists {feats.get('claimed_ai', 0)} AI skills but the "
                        f"career history shows no supporting ML/retrieval work")
    if feats["title_kind"] == "non_tech":
        concerns.append(f"title ('{title}') is non-technical despite an AI skills list")
    if feats["services_frac"] >= 0.999:
        concerns.append("entire career at IT-services firms")
    if feats["loc_kind"] == "outside_india":
        concerns.append(f"based outside India ({prof.get('country')}), no visa sponsorship")
    if isinstance(yoe, (int, float)) and not (5 <= yoe <= 9):
        concerns.append(f"experience ({yrs}) sits outside the 5-9 band")
    if info["response"] < 0.3 or info["days_inactive"] > 120:
        concerns.append(f"weak availability (response {info['response']:.2f}, "
                        f"last active {info['days_inactive']}d ago)")
    if feats["domain_cv"] > feats["domain_nlp"] and feats["domain_cv"] >= 2:
        concerns.append("background leans computer-vision/speech rather than NLP/IR")

    sentence = lead + (", " + "; ".join(pos) if pos else "") + "."
    if concerns:
        sentence += " Concern: " + "; ".join(concerns[:2]) + "."
    # Keep it to ~2 sentences / reasonable length.
    return sentence[:320]

# ---------------------------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------------------------
def _semantic_scores(texts: List[str]) -> np.ndarray:
    """TF-IDF cosine of each candidate's career text against the distilled JD
    query, normalised to [0, 1].

    Robust to tiny / homogeneous corpora (e.g. small sandbox uploads): if the
    aggressive df-pruning empties the vocabulary it retries with no pruning, and
    finally falls back to all-zeros so the other (structured) components still
    rank the candidates.
    """
    n = len(texts)
    for cfg in (dict(min_df=2, max_df=0.6), dict(min_df=1, max_df=1.0)):
        try:
            vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                  max_features=200_000, stop_words="english", **cfg)
            doc = vec.fit_transform(texts)                    # (n, V) sparse
            jd = vec.transform([JD_QUERY])                    # (1, V) sparse
            sims = (doc @ jd.T).toarray().ravel()             # cosine (L2-normed rows)
            smax = float(sims.max())
            return sims / smax if smax > 0 else np.zeros(n)
        except ValueError:
            continue
    return np.zeros(n)


def rank(candidates: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
    n = len(candidates)
    if n == 0:
        return []

    # Reference "today" = latest activity in the pool (robust to dataset vintage).
    today = date(2026, 6, 28)
    for c in candidates:
        d = _parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
        if d and d > today:
            today = d

    # --- semantic similarity (TF-IDF cosine vs the distilled JD query) ---
    texts = [career_text(c) for c in candidates]
    sims_norm = _semantic_scores(texts)

    scored: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        prof = c.get("profile", {}) or {}
        yoe = prof.get("years_of_experience")
        yoe = float(yoe) if isinstance(yoe, (int, float)) else None
        text = texts[i]

        hard_hp, hp_reasons = honeypot_report(c, today)
        title_s, title_kind = score_title(c)
        subst_s, core_hits, ml_hits = score_substance(text)
        prod_s, services_frac = score_product(c)
        dom_s, nlp_hits, cv_hits = score_domain(text)
        loc_s, loc_kind = score_location(c)

        base = (
            WEIGHTS["semantic"]   * sims_norm[i] +
            WEIGHTS["title"]      * title_s +
            WEIGHTS["substance"]  * subst_s +
            WEIGHTS["experience"] * score_experience(yoe) +
            WEIGHTS["product"]    * prod_s +
            WEIGHTS["domain"]     * dom_s +
            WEIGHTS["location"]   * loc_s +
            WEIGHTS["evaluation"] * score_evaluation(text) +
            WEIGHTS["education"]  * score_education(c)
        )

        modifier, avail = availability_modifier(c, today)
        composite = base * modifier

        # Keyword-stuffer guard: many claimed AI skills but no demonstrated work.
        claimed_ai = count_claimed_ai_skills(c)
        demonstrated = core_hits + ml_hits
        stuffer_factor, is_stuffer = stuffer_penalty(title_kind, claimed_ai, demonstrated)
        composite *= stuffer_factor

        if hard_hp:
            composite *= 0.02                     # force honeypots out of the top

        feats = {
            "text": text, "title_kind": title_kind, "substance": subst_s,
            "services_frac": services_frac, "loc_kind": loc_kind,
            "domain_nlp": nlp_hits, "domain_cv": cv_hits, "avail": avail,
            "honeypot": hard_hp, "hp_reasons": hp_reasons,
            "is_stuffer": is_stuffer, "claimed_ai": claimed_ai,
        }
        scored.append({
            "candidate_id": c.get("candidate_id"),
            "score": composite,
            "reasoning": build_reasoning(c, feats),
        })

    # Round first, then sort by (-score, candidate_id) so the output is
    # non-increasing AND equal scores are ordered by candidate_id ascending
    # (exactly what validate_submission.py requires).
    for r in scored:
        r["score"] = round(float(r["score"]), 6)
    scored.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return scored[:top_k]


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, r in enumerate(rows, start=1):
            writer.writerow([r["candidate_id"], rank_pos,
                             f"{r['score']:.6f}", r["reasoning"]])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob JD.")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl[.gz] or a .json array.")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path (default: submission.csv).")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args(argv)

    t0 = datetime.now()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates "
          f"({(datetime.now() - t0).total_seconds():.1f}s)", file=sys.stderr)

    rows = rank(candidates, top_k=args.top_k)
    write_csv(rows, args.out)

    dt = (datetime.now() - t0).total_seconds()
    print(f"Wrote {len(rows)} ranked candidates to {args.out} "
          f"(total {dt:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
