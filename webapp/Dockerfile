FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY ADNI_data/webapp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY ADNI_data/webapp /app/ADNI_data/webapp
COPY ADNI_data/results/web_calculator_36m_model2_lr_v1/artifacts /app/ADNI_data/results/web_calculator_36m_model2_lr_v1/artifacts

EXPOSE 8501

CMD ["streamlit", "run", "ADNI_data/webapp/minimal_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
