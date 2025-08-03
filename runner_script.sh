#!/bin/bash

# Step 1: Build the Docker image
sudo docker build -t urbansound-api:latest .

# Step 2: Stop and remove any existing container with the same name
sudo docker rm -f urbansound-api-container-v2 || true

# Step 3: Run the container in detached mode
sudo docker run -d -p 8000:8000 --name urbansound-api-container-v2 urbansound-api:latest

# Step 4: Tag the image with your Docker Hub username/repo
sudo docker tag urbansound-api:latest lscblack/urbansound-api:latest

# Step 5: Push the image to Docker Hub
sudo docker push lscblack/urbansound-api:latest
