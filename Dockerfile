FROM python:3.10-slim

WORKDIR /app

# 1. Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# 2. Install dependencies 
# (By copying requirements first, we save build time if you only change code later)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio requests

# 3. Copy the rest of your application
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
COPY main.py .

# 4. Environment Variables
ENV DOCKER_ENV=true
# Use 127.0.0.1 because both apps are now "roommates" in the same container
ENV BACKEND_URL=http://127.0.0.1:8000/predict
# Ensure Python outputs logs immediately to the terminal
ENV PYTHONUNBUFFERED=1

# 5. Hugging Face Spaces specifically looks for port 7860
EXPOSE 7860

# 6. Start the Backend and Frontend
# 'uvicorn' runs the API in the background (&)
# 'python main.py' runs the UI in the foreground
CMD uvicorn app:app --host 0.0.0.0 --port 8000 & python main.py