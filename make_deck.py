#!/usr/bin/env python3
"""
make_deck.py - generate the approach deck (PDF) required by the submission portal.

Produces a clean, slide-style 16:9 PDF that explains *what* was built, *why*, and
*how* it works - tied to the actual implementation in rank.py. Reproducible:

    pip install reportlab
    python make_deck.py --out approach_deck.pdf
"""

from __future__ import annotations

import argparse

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# Standard landscape A4 (max portal/validator compatibility).
PAGE_W, PAGE_H = landscape(A4)


def _embed_fonts() -> bool:
    """Override the non-embedded base-14 fonts with embedded TrueType faces so
    the PDF is fully self-contained.

    The reference deck that the portal accepted was LaTeX output with embedded
    fonts; reportlab's default base-14 Helvetica is NOT embedded, which strict
    upload validators can reject. Re-registering the same logical names as TTF
    means every existing setFont('Helvetica'...) call now uses an embedded font
    with no other code changes. Falls back silently to base-14 if no system
    font is found (e.g. on a non-macOS machine).
    """
    import os
    sup = "/System/Library/Fonts/Supplemental/"
    faces = {
        "Helvetica":         [sup + "Arial.ttf"],
        "Helvetica-Bold":    [sup + "Arial Bold.ttf"],
        "Helvetica-Oblique": [sup + "Arial Italic.ttf", sup + "Arial.ttf"],
        "Courier":           [sup + "Courier New.ttf", sup + "Arial.ttf"],
    }
    embedded = False
    for name, paths in faces.items():
        for p in paths:
            if os.path.exists(p):
                try:
                    pdfmetrics.registerFont(TTFont(name, p))
                    embedded = True
                    break
                except Exception:
                    continue
    if embedded:
        registerFontFamily("Helvetica", normal="Helvetica",
                           bold="Helvetica-Bold", italic="Helvetica-Oblique",
                           boldItalic="Helvetica-Bold")
    return embedded


_FONTS_EMBEDDED = _embed_fonts()

NAVY = colors.HexColor("#1F4E78")
BLUE = colors.HexColor("#2E86C1")
LIGHT = colors.HexColor("#EAF2F8")
GREY = colors.HexColor("#2C3E50")
MUTED = colors.HexColor("#5D6D7E")

TEAM = "Redrob Ranker  -  Nikhil Bhardwaj"
GITHUB = "github.com/bhardwajnikhil823/redrob-ranker"
SANDBOX = "redrob-ranker-nikhil.streamlit.app"

# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=15, leading=22,
                      textColor=GREY)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=18, bulletIndent=2,
                        spaceAfter=10)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=13, leading=19, leftIndent=40,
                     bulletIndent=24, textColor=MUTED, spaceAfter=6)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=12, leading=15,
                      textColor=GREY)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def _chrome(c: canvas.Canvas, title: str, idx: int, total: int) -> None:
    """Draw the common slide frame: background, title bar, footer."""
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Title bar.
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 0.95 * inch, PAGE_W, 0.95 * inch, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, PAGE_H - 1.0 * inch, PAGE_W, 0.05 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 23)
    c.drawString(0.6 * inch, PAGE_H - 0.66 * inch, title)
    # Footer.
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9.5)
    c.drawString(0.6 * inch, 0.32 * inch, TEAM)
    c.drawRightString(PAGE_W - 0.6 * inch, 0.32 * inch,
                      f"{idx} / {total}   ·   {GITHUB}")


def _body_frame(c: canvas.Canvas, flowables) -> None:
    frame = Frame(0.6 * inch, 0.6 * inch, PAGE_W - 1.2 * inch,
                  PAGE_H - 1.7 * inch, leftPadding=0, rightPadding=0,
                  topPadding=10, bottomPadding=0, showBoundary=0)
    frame.addFromList(flowables, c)


def _bullets(items):
    out = []
    for it in items:
        if isinstance(it, tuple):  # (text, "sub")
            out.append(Paragraph(it[0], SUB, bulletText="–"))
        else:
            out.append(Paragraph(it, BULLET, bulletText="•"))
    return out


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def slide_title(c: canvas.Canvas, total: int) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, PAGE_H * 0.5 - 3, PAGE_W, 6, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 31)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.62,
                        "Intelligent Candidate Discovery & Ranking")
    c.setFillColor(LIGHT)
    c.setFont("Helvetica", 19)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.54,
                        "Redrob Hackathon  -  Senior AI Engineer (Founding Team)")

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Oblique", 17)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.40,
                        "An explainable, CPU-only ranker that scores demonstrated"
                        " work - not keywords.")
    c.setFillColor(LIGHT)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.30, TEAM)
    c.setFont("Helvetica", 12.5)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.22, f"Code: {GITHUB}")
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.17, f"Live sandbox: {SANDBOX}")
    c.showPage()


def slide_problem(c, total):
    _chrome(c, "The problem: rank 100K candidates for a hard-to-fill role", 2, total)
    _body_frame(c, _bullets([
        "The JD is written to separate what it <b>says</b> from what it <b>means</b>: "
        "a perfect AI-keyword skills list behind a <i>Marketing Manager</i> title is "
        "<b>not</b> a fit; a plain-language profile that actually built a recommender "
        "<b>is</b>.",
        "The pool is adversarial by design: keyword stuffers, understated strong "
        "candidates, behavioral twins, and ~80 <b>honeypots</b> with subtly impossible "
        "profiles.",
        "Hard constraints: top-100 ranking, <b>CPU-only, &le;5 min, &le;16 GB, no "
        "network</b> - an LLM-per-candidate approach simply cannot scale here.",
        "Our thesis: <b>score demonstrated work</b> from career history, weigh <b>real "
        "availability</b>, and refuse to be fooled by skill-list keywords.",
    ]))
    c.showPage()


def slide_architecture(c, total):
    _chrome(c, "Architecture: a transparent scoring pipeline", 3, total)
    steps = ["Load 100K candidates (streamed JSONL)",
             "TF-IDF semantic match  +  9 structured components",
             "Weighted base score  in [0, 1]",
             "x  behavioral availability modifier  (~0.5-1.1)",
             "Honeypot & keyword-stuffer guards",
             "Top 100  +  grounded reasoning  ->  CSV / XLSX"]
    data = [[Paragraph(f"<b>{i+1}</b>", CELL_B), Paragraph(s, CELL)]
            for i, s in enumerate(steps)]
    t = Table(data, colWidths=[0.5 * inch, 9.7 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#D5DBDB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBDB")),
    ]))
    _body_frame(c, [t, Paragraph(
        "<br/>Pure scikit-learn + rule logic - <b>no LLM, no GPU, no network</b>. "
        "Measured ~30 s and ~3.5 GB peak for the full 100K pool.", BODY)])
    c.showPage()


def slide_components(c, total):
    _chrome(c, "Scoring: nine interpretable components", 4, total)
    rows = [
        ("substance", "0.24", "Hands-on retrieval / ranking / recommendation work in the career text"),
        ("semantic", "0.20", "TF-IDF (1-2gram) cosine similarity to a distilled JD query"),
        ("title", "0.16", "Tiered role gate: AI/ML -> SWE -> adjacent -> non-technical"),
        ("experience", "0.10", "5-9 year band (ideal 6-8), soft falloff outside"),
        ("product", "0.09", "Product company vs pure IT-services / consulting career"),
        ("location", "0.08", "Pune/Noida -> NCR -> metro -> other-India -> outside India"),
        ("domain", "0.07", "NLP / IR (good) vs computer-vision / speech-only (negative)"),
        ("evaluation", "0.04", "Ranking-eval experience: NDCG / MRR / MAP / A-B testing"),
        ("education", "0.02", "Institution tier (minor)"),
    ]
    data = [[Paragraph("<b>Component</b>", CELL_B), Paragraph("<b>Weight</b>", CELL_B),
             Paragraph("<b>What it rewards</b>", CELL_B)]]
    for name, w, desc in rows:
        data.append([Paragraph(name, CELL_B), Paragraph(w, CELL), Paragraph(desc, CELL)])
    t = Table(data, colWidths=[1.8 * inch, 0.9 * inch, 7.5 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBDB")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, NAVY),
    ]))
    _body_frame(c, [t, Paragraph(
        "<br/>Final score = base (sum of the nine) <b>x</b> behavioral availability "
        "modifier; honeypots are then forced out.", BODY)])
    c.showPage()


def slide_traps(c, total):
    _chrome(c, "Beating the traps (the core differentiator)", 5, total)
    _body_frame(c, _bullets([
        "<b>Score career descriptions, not the skills array</b> - the exact field a "
        "stuffer inflates is excluded from the text we match against.",
        "<b>Tiered title gate</b> - non-technical titles are capped low; only genuine "
        "AI/ML roles get full credit (general SWE can still climb via real evidence).",
        "<b>Keyword-stuffer guard</b> - many claimed AI skills but no supporting career "
        "evidence -> x0.30-0.75 (<b>28 caught</b>), while genuine career-switchers are "
        "explicitly protected.",
        "<b>Honeypot detection</b> - logical impossibilities are forced out (x0.02):",
        ("role duration &gt; its own date span (\"8 yrs at a 3-yr-old company\")", "sub"),
        ("declared tenure &gt; total experience; reversed or future dates", "sub"),
        ("many \"expert\" skills with 0 months of usage", "sub"),
    ]))
    c.showPage()


def slide_availability(c, total):
    _chrome(c, "Hireability: behavioral availability modifier", 6, total)
    _body_frame(c, _bullets([
        "A perfect-on-paper candidate who never replies and hasn't logged in for "
        "months is, for hiring purposes, <b>not actually available</b>.",
        "A multiplicative modifier (~0.5-1.1x) combines five Redrob signals:",
        ("recruiter_response_rate  ·  last-active recency  ·  open-to-work", "sub"),
        ("interview_completion_rate  ·  profile_completeness", "sub"),
        "Because it <b>multiplies</b> the base score, strong-but-unavailable profiles "
        "are <b>down-weighted, not deleted</b> - they can still surface if truly elite.",
        "This directly answers the JD's instruction to \"down-weight appropriately\" "
        "candidates who aren't reachable.",
    ]))
    c.showPage()


def slide_reasoning(c, total):
    _chrome(c, "Explainable, grounded reasoning per candidate", 7, total)
    _body_frame(c, _bullets([
        "Every pick carries a 1-2 sentence justification built <b>only from facts in "
        "the profile</b>: years, title, company, strongest career evidence, top "
        "actually-used skills, location, and live signal values.",
        "<b>Honest about concerns</b> - out-of-band experience, outside India, "
        "services-only career, weak availability, CV/speech lean - so the tone matches "
        "the rank (checked at Stage 4).",
        "<b>No hallucination</b> - a skill is cited as \"depth\" only when career "
        "evidence supports it, otherwise neutrally as \"lists\".",
        "Result: a reviewer can audit <b>why</b> any candidate sits where they do.",
    ]))
    c.showPage()


def slide_results(c, total):
    _chrome(c, "Results & validation", 8, total)
    metrics = [
        ("Top-100 titles", "100% technical AI/ML roles"),
        ("Experience fit", "82% within the 5-9 year band"),
        ("Honeypots in top 100", "0   (disqualification limit: 10%)"),
        ("Keyword-stuffers", "28 down-weighted out of contention"),
        ("Compute", "~30 s  ·  ~3.5 GB peak   (limits: 5 min / 16 GB)"),
        ("Format & tests", "passes validate_submission.py  ·  23 tests green"),
    ]
    data = [[Paragraph(f"<b>{k}</b>", CELL_B), Paragraph(v, CELL)] for k, v in metrics]
    t = Table(data, colWidths=[3.1 * inch, 7.2 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBDB")),
        ("LINEAFTER", (0, 0), (0, -1), 0.5, colors.HexColor("#D5DBDB")),
    ]))
    _body_frame(c, [t])
    c.showPage()


def slide_repro(c, total):
    _chrome(c, "Reproducibility & engineering", 9, total)
    _body_frame(c, _bullets([
        "<b>One command</b>: <font face='Courier'>python rank.py --candidates "
        "candidates.jsonl --out submission.csv</font>",
        "Pinned <font face='Courier'>requirements.txt</font>; CPU-only; no network "
        "during ranking - reproduces inside the Stage-3 sandbox unchanged.",
        "<b>23 unit/regression tests</b> (incl. a honeypot false-positive regression); "
        "authentic, staged git history.",
        "<b>Live sandbox</b> (Streamlit) + <b>Dockerfile</b> for one-command "
        "reproduction on a small sample.",
        "Built to be <b>defended</b>: every rank traces to explicit, inspectable logic - "
        "not an opaque model dump.",
    ]))
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(PAGE_W / 2, 0.85 * inch,
                        f"Code: {GITHUB}      ·      Sandbox: {SANDBOX}")
    c.showPage()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="approach_deck.pdf")
    args = ap.parse_args()

    c = canvas.Canvas(args.out, pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Redrob Ranker - Approach Deck")
    c.setAuthor("Nikhil Bhardwaj")

    total = 9
    slide_title(c, total)
    slide_problem(c, total)
    slide_architecture(c, total)
    slide_components(c, total)
    slide_traps(c, total)
    slide_availability(c, total)
    slide_reasoning(c, total)
    slide_results(c, total)
    slide_repro(c, total)
    c.save()
    print(f"Wrote {total}-slide deck to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
