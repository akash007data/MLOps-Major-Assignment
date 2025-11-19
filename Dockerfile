FROM python:3.12-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY app.py .
COPY notebooks/models/savedmodel.pth notebooks/models/savedmodel.pth
EXPOSE 5000
ENV FLASK_ENV=production
CMD ["python", "app.py"]
