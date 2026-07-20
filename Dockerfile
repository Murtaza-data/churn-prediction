# Recipe to build the churn API image.

# 1. Bring our own Python 3.12 (the fix for the whole "wrong Python" saga).
FROM python:3.12-slim

# 2. Work inside /app in the container.
WORKDIR /app

# 3. Install libraries FIRST (cached unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the app code + the model into the image.
COPY main.py preprocessing.py ./
COPY model/ ./model/

# 5. The app listens on this port (informational).
EXPOSE 8000

# 6. Command run when the container starts: launch the API.
#    Reads $PORT so it works on Render too.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
