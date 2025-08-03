## 📦 UrbanSound8K - Environmental Sound Classification

### 🎯 Project Objective

To develop and deploy an end-to-end machine learning system that classifies environmental sounds using the UrbanSound8K dataset. The system supports prediction, retraining, metric logging, database integration, and load testing simulation using Locust.

---

## 🧠 Problem Solved

Detecting and classifying urban sounds such as sirens, dog barks, or engine idling is crucial in today’s smart cities. These sounds are often indicators of important events or potential hazards. For example, real-time detection of sirens can help emergency response systems prioritize tasks, while recognizing engine idling or dog barks can assist in monitoring traffic flow or ensuring public safety.

This project uses machine learning to automate the classification of these sounds, enabling more efficient surveillance, accessibility services, and urban noise management. By doing so, it supports smarter, more responsive environments—whether for enhancing public safety, improving accessibility for people with disabilities, or mapping noise pollution in urban spaces.

link to [dataset](https://urbansounddataset.weebly.com/download-urbansound8k.html)
---

## 🧪 Model Performance

| Metric        | Score    |
| ------------- | -------- |
| **Accuracy**  | `0.8975` |
| **F1-Score**  | `0.90`   |
| **Precision** | `0.91`   |
| **Recall**    | `0.89`   |

> Model: `RandomForestClassifier(n_estimators=100)`
> Dataset: Fold 1 of UrbanSound8K

---

## 📊 Classification Report (Summary)

```text
              precision    recall  f1-score   support
    air_cond     0.94      0.95      0.95       203
    car_horn     0.97      0.83      0.89        86
    children     0.78      0.89      0.83       183
    dog_bark     0.85      0.85      0.85       201
    drilling     0.90      0.84      0.87       206
    engine_id    0.93      0.99      0.96       193
    gun_shot     0.98      0.78      0.87        72
    jackhammer   0.93      0.95      0.94       208
    siren        0.93      0.97      0.95       165
    street_mus   0.88      0.83      0.86       230
```

---

## 🚀 Features

✅ MFCC-based feature extraction
✅ Random Forest model training + evaluation
✅ FastAPI backend for predictions and retraining
✅ PostgreSQL database logging for:

* Model metadata
* Training sessions
* Predictions
* Performance metrics
  ✅ Live dashboard via API endpoints
  ✅ Load testing via Locust

---

## 🎥 Video Demo

📺 **YouTube Link:** [https://youtu.be/YOUR\_DEMO\_LINK](https://youtu.be/YOUR_DEMO_LINK)

---

## 🌐 Deployed API URL (Backend)

📡 **Live URL** [http://172.31.17.16:8000/docs](http://172.31.17.16:8000/docs)
*(Hosted On AWS EC2)*

## 🌐 Deployed UI URL (Frontend)
📡 **Live URL:** [https://soundclassifier-lscblack.netlify.app](https://soundclassifier-lscblack.netlify.app)
*(hosted on netlify)*


## 🐳 Setup Instructions for backend

### 1. Clone the Repo

```bash
git clone https://github.com/lscblack/Urban_Voice_classifier
cd Urban_Voice_classifier
```

### 2. Install Docker & Build

```bash
sudo docker build -t urbansound-api .
```

### 3. Run the API

```bash
sudo docker run -p 8000:8000 urbansound-api
```

### 4. Access Docs

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Run Locust Simulation

### 1. Start Locust

```bash
locust -f simulation/locustfile.py --host=http://localhost:8000
```

### 2. Open Web UI

Navigate to [http://localhost:8089](http://localhost:8089)

### 3. Sample Results Summary:

| Users | Requests/sec | Median Latency | Failure Rate |
| ----- | ------------ | -------------- | ------------ |
| 50    | 150 req/s    | 200 ms         | 0%           |
| 100   | 280 req/s    | 230 ms         | 1.2%         |


### Frontend Setup Instructions

1. **Navigate to the Client Directory**:

   ```bash
   cd client
   ```

2. **Install Dependencies**:

   * If you don’t have **pnpm** installed, use **npm** to install it globally:

     ```bash
     npm install -g pnpm
     ```
   * Once **pnpm** is installed, run the following to install project dependencies:

     ```bash
     pnpm install
     ```

3. **Start the Development Server**:

   * Run the following command to start the development server:

     ```bash
     pnpm run dev
     ```


## 🗃️ Database Tables Used

* `model_versions`
* `prediction_history`
* `model_evaluations`


Fully integrated via psycopg2 with automatic logging.


---

## 🧰 Tech Stack

| Component        | Tool/Library       |
| ---------------- | ------------------ |
| ML Framework     | Scikit-learn       |
| Audio Processing | Librosa            |
| Backend API      | FastAPI            |
| Database         | PostgreSQL (Aiven) |
| Simulation       | Locust             |
| Deployment       | Docker/aws/netlify |

---

## ✍️ Author

**Lscblack (Loue Sauveur Christian)**
[github.com/lscblack](https://github.com/lscblack)

