
FROM python:3.12-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ./src/
COPY data/ ./data/


# this image can never reach a local Ollama, so groq is the only sane default here
ENV LLM_PROVIDER=groq

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]