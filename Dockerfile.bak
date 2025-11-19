# Use official Python slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements & install
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and the saved model
COPY app.py .
COPY notebooks/models/savedmodel.pth notebooks/models/savedmodel.pth

# Expose port and run
EXPOSE 5000
ENV FLASK_ENV=production
CMD ["python", "app.py"]
