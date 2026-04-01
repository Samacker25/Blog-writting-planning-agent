FROM python:3.11-slim

WORKDIR /monolithic_app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend_aws.py", "--server.port=8501", "--server.address=0.0.0.0"]