FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx numpy scipy
COPY cefield_agent.py .
CMD ["python", "cefield_agent.py"]
