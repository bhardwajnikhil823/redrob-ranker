#!/usr/bin/env python3
"""
Unit tests for the Redrob ranker (`rank.py`).

Run:
    python -m unittest test_rank -v
    # or
    python -m unittest discover -v

Pure-function tests use small synthetic candidates; one integration test runs
the full pipeline on the bundled sample_candidates.json if it is present.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date

import rank as R

TODAY = date(2026, 6, 28)


def make_candidate(**overrides):
    """Build a schema-shaped 'good' candidate; override any field per test."""
    c = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Machine Learning Engineer",
            "summary": ("Experienced engineer who built recommendation and "
                        "ranking systems with embeddings and vector search "
                        "over large data; strong Python and retrieval work."),
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Machine Learning Engineer",
            "current_company": "Flipkart",
            "current_company_size": "1001-5000",
            "current_industry": "Internet",
        },
        "career_history": [{
            "company": "Flipkart", "title": "Machine Learning Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 40,
            "is_current": True, "industry": "Internet", "company_size": "1001-5000",
            "description": ("Built a recommendation system and search ranking "
                            "pipeline using embeddings and learning to rank; "
                            "ran NDCG evaluation and A/B tests on data."),
        }],
        "education": [{
            "institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
            "start_year": 2014, "end_year": 2018, "grade": "8.5", "tier": "tier_1",
        }],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40,
             "duration_months": 80},
            {"name": "Recommendation Systems", "proficiency": "advanced",
             "endorsements": 30, "duration_months": 50},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2022-01-01",
            "last_active_date": "2026-06-20", "open_to_work_flag": True,
            "profile_views_received_30d": 50, "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8, "avg_response_time_hours": 4,
            "skill_assessment_scores": {}, "connection_count": 300,
            "endorsements_received": 100, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 70, "search_appearance_30d": 40,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
    # Shallow-merge overrides into the relevant sub-objects.
    for key, val in overrides.items():
        if key in ("profile", "redrob_signals") and isinstance(val, dict):
            c[key].update(val)
        else:
            c[key] = val
    return c


class TestHoneypot(unittest.TestCase):
    def test_normal_candidate_not_flagged(self):
        hard, _ = R.honeypot_report(make_candidate(), TODAY)
        self.assertFalse(hard)

    def test_single_role_longer_than_career(self):
        c = make_candidate(
            profile={"years_of_experience": 5.0},
            career_history=[{"company": "X", "title": "Eng",
                             "start_date": "2018-01-01", "end_date": None,
                             "duration_months": 200, "is_current": True,
                             "industry": "Internet", "company_size": "51-200",
                             "description": "work"}])
        hard, reasons = R.honeypot_report(c, TODAY)
        self.assertTrue(hard, reasons)

    def test_reversed_role_dates(self):
        c = make_candidate(career_history=[{
            "company": "X", "title": "Eng", "start_date": "2022-01-01",
            "end_date": "2020-01-01", "duration_months": 10, "is_current": False,
            "industry": "Internet", "company_size": "51-200", "description": "w"}])
        hard, _ = R.honeypot_report(c, TODAY)
        self.assertTrue(hard)

    def test_education_ends_before_start(self):
        c = make_candidate(education=[{
            "institution": "X", "degree": "B", "field_of_study": "CS",
            "start_year": 2020, "end_year": 2018}])
        hard, _ = R.honeypot_report(c, TODAY)
        self.assertTrue(hard)

    def test_many_zero_duration_expert_skills(self):
        skills = [{"name": f"Skill{i}", "proficiency": "expert",
                   "endorsements": 0, "duration_months": 0} for i in range(8)]
        hard, _ = R.honeypot_report(make_candidate(skills=skills), TODAY)
        self.assertTrue(hard)

    def test_skill_used_longer_than_career_is_NOT_honeypot(self):
        # Regression: a skill duration exceeding the professional career is
        # legitimate (college / side projects) and must NOT flag a honeypot.
        # An earlier version of this check flagged ~9% of the whole pool.
        c = make_candidate(
            profile={"years_of_experience": 4.0},
            career_history=[{"company": "X", "title": "Eng",
                             "start_date": "2022-01-01", "end_date": None,
                             "duration_months": 40, "is_current": True,
                             "industry": "Internet", "company_size": "51-200",
                             "description": "work"}],
            skills=[{"name": "Python", "proficiency": "expert",
                     "endorsements": 10, "duration_months": 120}])  # 10 yrs
        hard, _ = R.honeypot_report(c, TODAY)
        self.assertFalse(hard)


class TestTitle(unittest.TestCase):
    def _kind(self, title):
        return R.score_title(make_candidate(profile={"current_title": title}))

    def test_ai_ml_title_full_credit(self):
        score, kind = self._kind("Senior Machine Learning Engineer")
        self.assertEqual(kind, "ai_ml")
        self.assertEqual(score, 1.0)

    def test_non_tech_title_low(self):
        score, kind = self._kind("HR Manager")
        self.assertEqual(kind, "non_tech")
        self.assertLess(score, 0.2)

    def test_software_engineer_is_swe_tier(self):
        score, kind = self._kind("Software Engineer")
        self.assertEqual(kind, "swe")

    def test_frontend_is_adjacent(self):
        _, kind = self._kind("Frontend Engineer")
        self.assertEqual(kind, "adjacent_eng")

    def test_mechanical_is_unrelated(self):
        _, kind = self._kind("Mechanical Engineer")
        self.assertEqual(kind, "unrelated_eng")


class TestComponentScores(unittest.TestCase):
    def test_experience_band(self):
        self.assertEqual(R.score_experience(7.0), 1.0)
        self.assertEqual(R.score_experience(5.5), 0.9)
        self.assertGreater(R.score_experience(7.0), R.score_experience(3.0))
        self.assertGreater(R.score_experience(7.0), R.score_experience(15.0))

    def test_location_pune_best_outside_india_worst(self):
        pune, _ = R.score_location(make_candidate())
        out, kind = R.score_location(make_candidate(
            profile={"location": "Toronto", "country": "Canada"},
            redrob_signals={"willing_to_relocate": False}))
        self.assertEqual(pune, 1.0)
        self.assertEqual(kind, "outside_india")
        self.assertLess(out, 0.3)

    def test_product_vs_services(self):
        product, _ = R.score_product(make_candidate())
        all_services, frac = R.score_product(make_candidate(career_history=[
            {"company": "Infosys", "title": "Eng", "start_date": "2018-01-01",
             "end_date": None, "duration_months": 40, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "w"}]))
        self.assertEqual(product, 1.0)
        self.assertLess(all_services, 0.4)
        self.assertEqual(frac, 1.0)


class TestStufferPenalty(unittest.TestCase):
    def test_non_tech_with_claimed_skills_no_evidence_penalised(self):
        factor, is_stuffer = R.stuffer_penalty("non_tech", claimed_ai=5, demonstrated=0)
        self.assertTrue(is_stuffer)
        self.assertLessEqual(factor, 0.35)

    def test_genuine_ml_not_penalised(self):
        factor, is_stuffer = R.stuffer_penalty("ai_ml", claimed_ai=5, demonstrated=6)
        self.assertFalse(is_stuffer)
        self.assertEqual(factor, 1.0)

    def test_tier5_non_tech_with_real_evidence_not_penalised(self):
        # Non-AI title but demonstrated work present -> protected.
        factor, is_stuffer = R.stuffer_penalty("non_tech", claimed_ai=4, demonstrated=5)
        self.assertFalse(is_stuffer)
        self.assertEqual(factor, 1.0)

    def test_count_claimed_ai_skills(self):
        c = make_candidate(skills=[
            {"name": "RAG", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 1, "duration_months": 1},
            {"name": "Cooking", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
            {"name": "LLMs", "proficiency": "beginner", "endorsements": 1, "duration_months": 1},
        ])
        # RAG + Pinecone count (advanced/expert AI); Cooking is not AI;
        # LLMs is AI but only 'beginner' so excluded.
        self.assertEqual(R.count_claimed_ai_skills(c), 2)


class TestEndToEnd(unittest.TestCase):
    def _shared_text(self, extra):
        # Common filler keeps the TF-IDF vocabulary non-empty on tiny corpora.
        return ("engineer working with data and systems and teams. " + extra)

    def test_rank_orders_and_sinks_traps(self):
        strong = make_candidate(candidate_id="CAND_0000001")
        honeypot = make_candidate(
            candidate_id="CAND_0000002",
            profile={"years_of_experience": 5.0,
                     "summary": self._shared_text("built ranking and data systems")},
            career_history=[{"company": "X", "title": "ML Engineer",
                             "start_date": "2018-01-01", "end_date": None,
                             "duration_months": 200, "is_current": True,
                             "industry": "Internet", "company_size": "51-200",
                             "description": self._shared_text("ranking systems")}])
        stuffer = make_candidate(
            candidate_id="CAND_0000003",
            profile={"current_title": "HR Manager", "headline": "HR Manager",
                     "summary": self._shared_text("recruiting and people teams")},
            career_history=[{"company": "Y", "title": "HR Manager",
                             "start_date": "2019-01-01", "end_date": None,
                             "duration_months": 50, "is_current": True,
                             "industry": "HR", "company_size": "201-500",
                             "description": self._shared_text("hiring and payroll")}],
            skills=[{"name": n, "proficiency": "expert", "endorsements": 5,
                     "duration_months": 20} for n in
                    ["RAG", "Pinecone", "LLMs", "Vector Search", "Transformers"]])
        outside = make_candidate(
            candidate_id="CAND_0000004",
            profile={"location": "London", "country": "United Kingdom",
                     "summary": self._shared_text("built recommendation systems")},
            redrob_signals={"willing_to_relocate": False})

        ranked = R.rank([honeypot, stuffer, outside, strong], top_k=4)

        ids = [r["candidate_id"] for r in ranked]
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)                       # unique
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))   # non-increasing
        self.assertEqual(ranked[0]["candidate_id"], "CAND_0000001")   # strong wins
        self.assertEqual(ranked[-1]["candidate_id"], "CAND_0000002")  # honeypot last
        self.assertNotIn("CAND_0000003", ids[:2])                # stuffer not near top

    def test_write_csv_format(self):
        ranked = R.rank([make_candidate(candidate_id=f"CAND_000000{i}",
                                        profile={"summary": self._shared_text(f"role {i}")})
                         for i in range(1, 6)], top_k=5)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            R.write_csv(ranked, path)
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        finally:
            os.unlink(path)
        self.assertEqual(lines[0], "candidate_id,rank,score,reasoning")
        self.assertEqual(len(lines), 6)                          # header + 5
        self.assertTrue(lines[1].split(",")[1] == "1")           # first rank is 1


@unittest.skipUnless(os.path.exists("sample_candidates.json"),
                     "sample_candidates.json not in cwd")
class TestSampleIntegration(unittest.TestCase):
    def test_runs_on_bundled_sample(self):
        candidates = R.load_candidates("sample_candidates.json")
        self.assertGreater(len(candidates), 0)
        ranked = R.rank(candidates, top_k=10)
        self.assertEqual(len(ranked), 10)
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for r in ranked:
            self.assertRegex(r["candidate_id"], r"^CAND_\d{7}$")
            self.assertTrue(r["reasoning"])                      # non-empty reasoning


if __name__ == "__main__":
    unittest.main(verbosity=2)
