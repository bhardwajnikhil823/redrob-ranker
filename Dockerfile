# Self-contained sandbox image for the Redrob ranker (Streamlit demo).
# Satisfies submission_spec 10.5's "docker run recipe" alternative.
#
#   docker build -t redrob-ranker .
#   docker run --rm -p 8501:8501 redrob-ranker
#   # open http://localhost:8501
#
FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching. scikit-learn/scipy/numpy ship
# manylinux wheels, so no system build toolchain is required.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, the ranker it calls, Streamlit config, and the bundled sample so the
# demo works out of the box with no upload.
COPY rank.py app.py ./
COPY .streamlit/ ./.streamlit/
COPY sample_candidates.json ./

EXPOSE 8501

# Streamlit must bind 0.0.0.0 to be reachable from outside the container.
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
