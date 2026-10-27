FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY knowledge-base ./knowledge-base
COPY data ./data
RUN useradd -u 10001 -m app && mkdir /state && chown app:app /state
USER app
ENV CHROMA_DIR=/state/chroma_db DEMO_BUDGET_PATH=/state/budget.sqlite3 HOST=0.0.0.0 PORT=8000
CMD ["python", "-m", "src.main", "--web"]
