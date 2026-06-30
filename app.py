#!/usr/bin/env python3
"""
app.py - Sandbox / demo for the Redrob candidate ranker.

A small Streamlit front-end that satisfies the hackathon's mandatory sandbox
requirement (submission_spec Section 10.5): it accepts a small candidate sample,
runs the *exact same* ranking code as the offline submission (`rank.py`), and
produces a downloadable, spec-compliant CSV - end to end, on CPU, no network.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud or Hugging Face Spaces by pointing the
platform at this repo; it installs from requirements.txt automatically.
"""

from __future__ import annotations

import os
import tempfile
import time

import pandas as pd
import streamlit as st

import rank as R

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("Redrob - Intelligent Candidate Ranker")
st.caption(
    "Explainable, CPU-only ranking for the *Senior AI Engineer - Founding Team* "
    "JD. Scores demonstrated career work (not the keyword-stuffable skills list), "
    "detects honeypots, and applies a behavioural availability modifier. "
    "Same code path as the offline `rank.py` submission."
)

with st.sidebar:
    st.header("Input")
    source = st.radio(
        "Candidate source",
        ["Bundled sample (50 candidates)", "Upload your own"],
    )
    uploaded = None
    if source == "Upload your own":
        uploaded = st.file_uploader(
            "candidates file (.json / .jsonl / .jsonl.gz)",
            type=["json", "jsonl", "gz"],
        )
    top_k = st.slider("How many to rank (top-K)", 5, 100, 20)
    run = st.button("Rank candidates", type="primary")
    st.markdown("---")
    st.caption("CPU only - no GPU, no network calls during ranking.")


def _load_uploaded(file) -> list:
    """Persist the uploaded bytes to a temp file so rank.load_candidates can
    auto-detect .json / .jsonl / .jsonl.gz by extension and magic bytes."""
    name = file.name
    suffix = name[name.find("."):] if "." in name else ".jsonl"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        return R.load_candidates(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if run:
    if source == "Upload your own" and uploaded is None:
        st.warning("Please upload a file, or switch to the bundled sample.")
        st.stop()

    with st.spinner("Loading and ranking..."):
        t0 = time.time()
        try:
            if source == "Upload your own":
                candidates = _load_uploaded(uploaded)
            else:
                candidates = R.load_candidates("sample_candidates.json")
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the UI
            st.error(f"Could not read candidates: {exc}")
            st.stop()

        if not candidates:
            st.error("No candidates found in the input.")
            st.stop()

        k = min(top_k, len(candidates))
        rows = R.rank(candidates, top_k=k)

        # Produce the exact spec-compliant CSV (candidate_id,rank,score,reasoning).
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            csv_path = tmp.name
        R.write_csv(rows, csv_path)
        with open(csv_path, "rb") as fh:
            csv_bytes = fh.read()
        os.unlink(csv_path)
        elapsed = time.time() - t0

    st.success(
        f"Ranked {len(candidates):,} candidates -> top {k} in {elapsed:.2f}s "
        f"(CPU, no network)."
    )

    display = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "candidate_id": r["candidate_id"],
                "score": r["score"],
                "reasoning": r["reasoning"],
            }
            for i, r in enumerate(rows)
        ]
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )
else:
    st.info("Pick a source in the sidebar and click **Rank candidates**.")
