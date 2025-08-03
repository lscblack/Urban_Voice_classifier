from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import librosa
from fastapi import HTTPException  
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import uvicorn
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2 import sql
from datetime import datetime
from typing import List
import io
import json
from urllib.parse import urlparse
# Define your PostgreSQL database URL (Aiven connection string)
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise EnvironmentError("DATABASE_URL is not set.")

# Parse database URL
result = urlparse(db_url)
username = result.username
password = result.password
database = result.path[1:]  # removes leading '/'
hostname = result.hostname
port = result.port
# Database connection setup
def get_db_connection():
    return  psycopg2.connect(
        dbname=database,
        user=username,
        password=password,
        host=hostname,
        port=port,
        sslmode="require"
    )

# Initialize database tables
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Model evaluations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_evaluations (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) DEFAULT 'Random Forest Model 1',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accuracy FLOAT,
        precision FLOAT,
        recall FLOAT,
        f1_score FLOAT,
        classification_report TEXT,
        confusion_matrix TEXT,
        notes TEXT,
        is_retraining BOOLEAN DEFAULT FALSE,
        samples_added INTEGER DEFAULT 0
    )
    """)
        # Add is_retraining column if it doesn't exist
    cursor.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name='model_evaluations' AND column_name='is_retraining'
        ) THEN
            ALTER TABLE model_evaluations ADD COLUMN is_retraining BOOLEAN DEFAULT FALSE;
        END IF;
    END
    $$;
    """)
    # Prediction history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_name VARCHAR(255),
        true_label VARCHAR(100),
        predicted_label VARCHAR(100),
        confidence FLOAT,
        is_correct BOOLEAN
    )
    """)
    
    # Model versions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id SERIAL PRIMARY KEY,
        version_name VARCHAR(100),
        model_path VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT FALSE,
        performance_metrics TEXT
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

# Initialize database tables on startup
init_db()

# Cell 4: Extract Features (MFCCs)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Initialize app with CORS middleware
app = FastAPI(title="UrbanSound Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
MODEL_PATH = '../models/urbansound_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
ENCODER_PATH = '../models/label_encoder.pkl'
CLASS_NAMES = '../models/class_names.txt'
# load previous datasets csv in models
X_train = pd.read_csv('../models/X_train.csv')
X_test = pd.read_csv('../models/X_test.csv')
y_train = pd.read_csv('../models/y_train.csv')
y_test = pd.read_csv('../models/y_test.csv')

# Load model artifacts with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    CLASS_NAMES = list(label_encoder.classes_)
    print(f"Successfully loaded model. Classes: {CLASS_NAMES}")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

# For metrics tracking - using class for better state management
class MetricsTracker:
    def __init__(self):
        self.predictions: List[str] = []
        self.labels: List[str] = []
    
    def add_prediction(self, prediction: str):
        self.predictions.append(prediction)
    
    def add_label(self, label: str):
        self.labels.append(label)
    
    def add_pair(self, label: str, prediction: str):
        self.labels.append(label)
        self.predictions.append(prediction)
    
    def reset(self):
        self.predictions = []
        self.labels = []
    
    def get_counts(self):
        return len(self.labels), len(self.predictions)

metrics_tracker = MetricsTracker()

# ----------- UTILITIES -----------
def extract_features(file_path: str) -> np.ndarray:
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def save_model_evaluation(metrics: dict, is_retraining: bool = False, samples_added: int = 0, notes: str = ""):
    """Save model evaluation metrics to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO model_evaluations (
            model_name, accuracy, precision, recall, f1_score,
            classification_report, confusion_matrix, notes, is_retraining, samples_added
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            "UrbanSound Classifier",
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1_score'),
            json.dumps(metrics.get('classification_report')),
            json.dumps(metrics.get('confusion_matrix')),
            notes,
            is_retraining,
            samples_added
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving model evaluation: {str(e)}")
        return False

def log_prediction_to_db(file_name: str, true_label: str, predicted_label: str, confidence: float = None):
    """Log prediction to database history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO prediction_history (
            file_name, true_label, predicted_label, confidence, is_correct
        ) VALUES (%s, %s, %s, %s, %s)
        """, (
            file_name,
            true_label,
            predicted_label,
            confidence,
            true_label == predicted_label
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error logging prediction: {str(e)}")
        return False

def get_test_data():
    """Helper function to get test data for evaluation"""
    return X_test.values, y_test.values.ravel()

def calculate_improvement(old_report, new_report):
    """Calculate performance improvement between models"""
    improvement = {}
    for metric in ['precision', 'recall', 'f1-score']:
        for avg in ['micro', 'macro', 'weighted']:
            if avg in old_report and avg in new_report:
                key = f"{metric}_{avg}"
                improvement[key] = new_report[avg][metric] - old_report[avg][metric]
    return improvement

# ----------- PREDICTION ENDPOINT -----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction_id = None
    prediction_time = None
    db_status = "not_attempted"
    
    try:
        # Save temporary file
        filepath = f"temp_{file.filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        # Extract and scale features
        features = extract_features(filepath)
        X = scaler.transform([features])
        
        # Get prediction
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        confidence = float(np.max(y_proba))
        class_name = label_encoder.inverse_transform(y_pred)[0]
        
        # Log prediction to database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO prediction_history (
                file_name, 
                predicted_label, 
                confidence
            ) VALUES (%s, %s, %s)
            RETURNING id, timestamp
            """, (
                file.filename,
                class_name,
                confidence
            ))
            
            # Get the inserted record details
            record = cursor.fetchone()
            prediction_id = record[0]
            prediction_time = record[1].isoformat()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            db_status = "success"
        except Exception as db_error:
            db_status = f"database_error: {str(db_error)}"
        
        # Clean up temporary file
        os.remove(filepath)
        
        # Prepare probabilities dictionary
        probabilities = {
            cls: {
                "probability": float(prob),
                "is_predicted": cls == class_name
            } 
            for cls, prob in zip(label_encoder.classes_, y_proba[0])
        }
        
        return {
            "prediction": class_name,
            "confidence": confidence,
            "prediction_id": prediction_id,
            "timestamp": prediction_time,
            "probabilities": probabilities,
            "db_status": db_status
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Prediction failed: {str(e)}",
                "db_status": db_status
            }
        )  
    
# ----------- RETRAINING ENDPOINT -----------
@app.post("/retrain")
async def retrain(
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...),
    test_size: float = Form(0.2)
):
    try:
        # Validate inputs
        if len(files) != len(labels):
            raise HTTPException(status_code=400, detail="Number of files and labels must match")
            
        if len(files) < 10 or len(files) % 10 != 0:
            raise HTTPException(
                status_code=400, 
                detail="Minimum 10 files required and must be in multiples of 10"
            )

        # Validate all files are WAV
        for file in files:
            if not file.filename.lower().endswith('.wav'):
                raise HTTPException(status_code=400, detail="Only WAV files are supported")

        # 1. Evaluate old model first
        X_test_old, y_test_old = get_test_data()
        y_pred_old = model.predict(X_test_old)
        old_report = classification_report(y_test_old, y_pred_old, output_dict=True)
        old_conf_matrix = confusion_matrix(y_test_old, y_pred_old).tolist()
        
        # Save old model metrics
        save_model_evaluation({
            "accuracy": accuracy_score(y_test_old, y_pred_old),
            "precision": old_report['weighted avg']['precision'],
            "recall": old_report['weighted avg']['recall'],
            "f1_score": old_report['weighted avg']['f1-score'],
            "classification_report": old_report,
            "confusion_matrix": old_conf_matrix
        }, is_retraining=False, notes="Pre-retraining evaluation")

        # 2. Process all new files
        all_mfccs = []
        all_labels = []
        
        for file, label in zip(files, labels):
            contents = await file.read()
            with io.BytesIO(contents) as audio_file:
                # Save temporary file for feature extraction
                temp_path = f"retrain_temp_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(contents)
                
                mfcc = extract_features(temp_path)
                all_mfccs.append(mfcc)
                all_labels.append(label)
                os.remove(temp_path)

        # Prepare new data plus old train data
        X_new = np.vstack((X_train, np.array(all_mfccs)))
        y_new = np.vstack((y_train, label_encoder.transform(all_labels)))
        X_new = scaler.transform(X_new)
        
        # 3. Retrain model
        model.fit(X_new, y_new.ravel())
        
        # 4. Evaluate new model
        y_pred_new = model.predict(X_test_old)
        new_report = classification_report(y_test_old, y_pred_new, output_dict=True)
        new_conf_matrix = confusion_matrix(y_test_old, y_pred_new).tolist()
        
        # Save new model metrics
        save_model_evaluation({
            "accuracy": accuracy_score(y_test_old, y_pred_new),
            "precision": new_report['weighted avg']['precision'],
            "recall": new_report['weighted avg']['recall'],
            "f1_score": new_report['weighted avg']['f1-score'],
            "classification_report": new_report,
            "confusion_matrix": new_conf_matrix
        }, is_retraining=True, samples_added=len(files), notes=f"Retrained with {len(files)} samples")

        # 5. Save updated model
        model_path = '../models/urbansound_model.pkl'
        joblib.dump(model, model_path)

        return {
            "status": "success",
            "old_model_performance": old_report,
            "new_model_performance": new_report,
            "improvement": calculate_improvement(old_report, new_report),
            "samples_added": len(files),
            "model_path": model_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
# ----------- METRICS ENDPOINT -----------
@app.get("/metrics")
def metrics():
    label_count, pred_count = metrics_tracker.get_counts()
    
    if label_count == 0 or pred_count == 0:
        return {"message": "No labeled predictions available for metrics"}
    
    if label_count != pred_count:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Label/prediction count mismatch ({label_count} labels vs {pred_count} predictions)",
                "solution": "Use /log_prediction endpoint to ensure balanced labels and predictions"
            }
        )
    
    try:
        y_true = label_encoder.transform(metrics_tracker.labels)
        y_pred = label_encoder.transform(metrics_tracker.predictions)
        
        # Get the actual unique classes present in the data
        present_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
        present_class_names = label_encoder.inverse_transform(present_classes)
        
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            labels=present_classes,
            target_names=present_class_names
        )
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()

        response = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "total_samples": label_count,
            "classes_present": present_class_names.tolist(),
            "all_classes": CLASS_NAMES
        }

        return jsonable_encoder(response)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Metrics calculation failed: {str(e)}"}
        )

# ----------- LOG PREDICTION WITH LABEL -----------
@app.post("/log_prediction")
async def log_prediction(file: UploadFile = File(...), true_label: str = Form(...)):
    try:
        # Validate input
        if true_label not in CLASS_NAMES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid label. Must be one of: {CLASS_NAMES}"}
            )

        # Save and extract features
        filepath = f"log_{file.filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        features = extract_features(filepath)
        X = scaler.transform([features])
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        confidence = float(np.max(y_proba))
        predicted_label = label_encoder.inverse_transform(y_pred)[0]
        os.remove(filepath)

        # Log for metrics
        metrics_tracker.add_pair(true_label, predicted_label)

        # Log to database
        log_prediction_to_db(
            file_name=file.filename,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence
        )

        return {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_correct": true_label == predicted_label,
            "status": "Prediction logged successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Logging prediction failed: {str(e)}"}
        )

# ----------- RESET METRICS -----------
@app.post("/reset_metrics")
def reset_metrics():
    metrics_tracker.reset()
    return {"status": "Metrics tracking reset"}

# ----------- MODEL INFO ENDPOINT -----------
@app.get("/model_info")
async def model_info():
    try:
        model_type = type(model).__name__
        encoder_map = dict(zip(
            label_encoder.classes_.tolist(),
            label_encoder.transform(label_encoder.classes_).tolist()
        ))

        label_count, pred_count = metrics_tracker.get_counts()

        response = {
            "model_type": model_type,
            "classes": CLASS_NAMES,
            "encoder_mapping": encoder_map,
            "metrics_tracking": {
                "labels_count": label_count,
                "predictions_count": pred_count,
                "balanced": label_count == pred_count
            }
        }

        return jsonable_encoder(response)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch model info: {str(e)}"}
        )

# ----------- TRAINING HISTORY ENDPOINT -----------
@app.get("/training_history")
async def get_training_history(limit: int = 10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if the columns exist
        cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'model_evaluations' 
        AND column_name IN ('is_retraining', 'samples_added')
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Build query based on available columns
        if 'is_retraining' in existing_columns and 'samples_added' in existing_columns:
            query = """
            SELECT id, model_name, timestamp, accuracy, precision, recall, f1_score, 
                   is_retraining, samples_added, notes
            FROM model_evaluations
            ORDER BY timestamp DESC
            LIMIT %s
            """
        else:
            query = """
            SELECT id, model_name, timestamp, accuracy, precision, recall, f1_score, notes
            FROM model_evaluations
            ORDER BY timestamp DESC
            LIMIT %s
            """
        
        cursor.execute(query, (limit,))
        
        history = []
        for row in cursor.fetchall():
            entry = {
                "id": row[0],
                "model_name": row[1],
                "timestamp": row[2].isoformat(),
                "accuracy": row[3],
                "precision": row[4],
                "recall": row[5],
                "f1_score": row[6],
                "notes": row[-1]  # Last column is always notes
            }
            
            # Add optional fields if they exist
            if 'is_retraining' in existing_columns and 'samples_added' in existing_columns:
                entry.update({
                    "is_retraining": row[7],
                    "samples_added": row[8]
                })
            
            history.append(entry)
        
        cursor.close()
        conn.close()
        
        return {"training_history": history}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch training history: {str(e)}"}
        )


# ----------- PREDICTION HISTORY ENDPOINT -----------
@app.get("/prediction_history")
async def get_prediction_history(limit: int = 10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, timestamp, file_name, true_label, predicted_label, confidence, is_correct
        FROM prediction_history
        ORDER BY timestamp DESC
        LIMIT %s
        """, (limit,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row[0],
                "timestamp": row[1].isoformat(),
                "file_name": row[2],
                "true_label": row[3],
                "predicted_label": row[4],
                "confidence": row[5],
                "is_correct": row[6]
            })
        
        cursor.close()
        conn.close()
        
        return {"prediction_history": history}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch prediction history: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)