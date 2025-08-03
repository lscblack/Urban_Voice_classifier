from locust import HttpUser, task, between
import random
import os
from random import choice

# Sample WAV files for testing use fold with data from UrbanSound dataset not just files fold and loop through files
SAMPLE_FILES = []

fold_path = os.path.abspath("data/audio/fold3")
# fold_path = f"/media/lscblack/files/projects/Machine Learning Projects/UrbanSound/urbansound_classifier/data/audio/fold3"
for file in os.listdir(fold_path):
    if file.endswith(".wav"):
        SAMPLE_FILES.append(os.path.join(fold_path, file))

# List of valid class labels
CLASS_LABELS = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

class UrbanSoundUser(HttpUser):
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between tasks
    
    def on_start(self):
        """Check if sample files exist"""
        for file in SAMPLE_FILES:
            if not os.path.exists(file):
                print(f"Warning: Sample file {file} not found. Some tests will be skipped.")
    
    @task(5)
    def predict_sound(self):
        """Test the prediction endpoint"""
        if not SAMPLE_FILES:
            return
            
        file_path = choice(SAMPLE_FILES)
        
        if not os.path.exists(file_path):
            return
            
        with open(file_path, "rb") as f:
            response = self.client.post(
                "/predict",
                files={"file": (os.path.basename(file_path), f, "audio/wav")}
            )
            
            # Validate response
            if response.status_code != 200:
                print(f"Prediction failed: {response.text}")
    
    @task(1)
    def log_prediction(self):
        """Test logging a prediction with a true label"""
        if not SAMPLE_FILES or not CLASS_LABELS:
            return
            
        file_path = choice(SAMPLE_FILES)
        true_label = choice(CLASS_LABELS)
        
        if not os.path.exists(file_path):
            return
            
        with open(file_path, "rb") as f:
            response = self.client.post(
                "/log_prediction",
                files={"file": (os.path.basename(file_path), f, "audio/wav")},
                data={"true_label": true_label}
            )
            
            if response.status_code != 200:
                print(f"Log prediction failed: {response.text}")
    
    @task(1)
    def get_metrics(self):
        """Test getting metrics"""
        self.client.get("/metrics")
    
    @task(1)
    def get_model_info(self):
        """Test getting model info"""
        self.client.get("/model_info")
    
    @task(1)
    def get_training_history(self):
        """Test getting training history"""
        self.client.get("/training_history?limit=5")
    
    @task(1)
    def get_prediction_history(self):
        """Test getting prediction history"""
        self.client.get("/prediction_history?limit=5")
    
    @task(1)
    def reset_metrics(self):
        """Test resetting metrics"""
        self.client.post("/reset_metrics")
    
    @task(1)
    def retrain_model(self):
        """Test retraining endpoint (note: this is a heavy operation)"""
        if len(SAMPLE_FILES) < 2 or not CLASS_LABELS:
            return
            
        # We need at least 2 files to test retraining
        files_to_upload = SAMPLE_FILES[:2]
        labels = [choice(CLASS_LABELS) for _ in files_to_upload]
        
        files = []
        for file_path in files_to_upload:
            if not os.path.exists(file_path):
                return
            files.append(("files", (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")))
        
        # Add the labels as form data
        data = {"test_size": 0.2}
        for i, label in enumerate(labels):
            data[f"labels"] = label
        
        response = self.client.post(
            "/retrain",
            files=files,
            data=data
        )
        
        # Close all the open files
        for _, (_, file, _) in files:
            file.close()
            
        if response.status_code != 200:
            print(f"Retraining failed: {response.text}")