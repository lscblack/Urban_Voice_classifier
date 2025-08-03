#!/bin/bash

# Step 1: 
sudo docker build -t urbansound-api:latest .

# Step 2: 
sudo docker rm -f urbansound-api-container-v2 || true

# Step 3:
sudo docker run -d -p 8000:8000 --name urbansound-api-container-v2 urbansound-api:latest

# Step 4: 
sudo docker tag urbansound-api:latest lscblack/urbansound-api:latest

# Step 5: 
sudo docker push lscblack/urbansound-api:latest
