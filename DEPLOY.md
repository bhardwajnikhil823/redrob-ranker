# Deploying the sandbox

The hackathon requires a **working hosted sandbox** where organizers can run the
ranker on a small sample (submission_spec Section 10.5). The app is `app.py`
(Streamlit) and it calls the same `rank.py` used for the offline submission.

Pick **one** of the options below and paste the resulting URL into
`submission_metadata.yaml` → `sandbox_link`.

---

## Option A — Streamlit Community Cloud (easiest, free)

1. Push this repo to GitHub (see **Pushing to GitHub** below).
2. Go to <https://share.streamlit.io> → **Create app** → select your repo + branch.
3. Set **Main file path** to `app.py` → **Deploy**.
4. Copy the public URL into `sandbox_link`.

Streamlit Cloud installs from `requirements.txt` automatically (CPU, free tier).

---

## Option B — Hugging Face Spaces (free)

1. Create a Space at <https://huggingface.co/new-space> → **SDK: Streamlit**.
2. Push this repo to the Space's git remote (or upload the files). HF detects
   `app.py` automatically.
3. Copy the Space URL into `sandbox_link`.

---

## Option C — Docker (self-contained; spec-accepted alternative)

The spec allows a `docker run` recipe in place of a hosted link. This image is
verified to build and serve the app unmodified.

```bash
docker build -t redrob-ranker .
docker run --rm -p 8501:8501 redrob-ranker
# open http://localhost:8501
```

For a shareable link, push to a public registry:

```bash
docker tag redrob-ranker YOUR_DOCKERHUB_USER/redrob-ranker:latest
docker push YOUR_DOCKERHUB_USER/redrob-ranker:latest
```

Organizers can then run:

```bash
docker run --rm -p 8501:8501 YOUR_DOCKERHUB_USER/redrob-ranker:latest
```

---

## Pushing to GitHub

The local commits currently use a placeholder identity. Set your real identity
and (optionally) rewrite the existing commits to it, then push:

```bash
git config user.name "Your Real Name"
git config user.email "your-github-email@example.com"

# Optional: re-author the existing commits with your identity
git rebase --root --exec "git commit --amend --no-edit --reset-author"

# Create an empty repo on github.com first (no README), then:
git remote add origin https://github.com/YOUR_USERNAME/redrob-ranker.git
git push -u origin main
```

After pushing, put the repo URL into `submission_metadata.yaml` → `github_repo`.

---

## Full-submission reproduction (Stage 3)

The sandbox runs on a small sample. To reproduce the **full** `submission.csv`,
organizers mount the candidate pool and run the ranker (not the app):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv   # ~30s, CPU
```
