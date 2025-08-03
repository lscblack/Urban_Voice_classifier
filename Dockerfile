# Base Python image
# FROM python:3.9-slim
# Base Python image with Alpine for a smaller footprint
FROM python:3.9-slim as build


# Set working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install dependencies with pip
RUN pip install --no-cache-dir -r requirements.txt
# copy environment variables
COPY .env /app/.env
# Copy all app files
COPY . ./
COPY ./models /app/models

# Make port 8000 available
EXPOSE 8000

# This is where the FastAPI app is located
WORKDIR /app/src

# The CMD command starts the FastAPI server.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]