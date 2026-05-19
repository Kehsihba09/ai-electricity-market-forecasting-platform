# ⚡ AI Electricity Market Forecasting Platform

An enterprise-grade real-time electricity price forecasting platform built using machine learning, MLOps, streaming infrastructure, and optimized deployment pipelines.

The system simulates a live electricity market environment using Kafka-based streaming architecture, performs optimized XGBoost forecasting with Optuna hyperparameter tuning, tracks experiments using MLflow, and serves low-latency predictions through FastAPI and Streamlit dashboards.

Designed as a production-style ML systems engineering project inspired by institutional energy trading and real-time forecasting infrastructure.

## 🚀 Live Demo

[Open Live Forecasting Dashboard]
(https://ai-electricity-market-forecasting-platform-k8nkc57dsdqkfxmtous.streamlit.app/)

## 🏗️ System Architecture

```text
                 Historical IEX Dataset
                           │
                           ▼
                 Feature Engineering Pipeline
                           │
                           ▼
                    Optuna Tuning
                           │
                           ▼
                    XGBoost Training
                           │
                           ▼
                     MLflow Tracking
                           │
                           ▼
                 ONNX Optimization Layer
                           │
                           ▼
               Kafka Streaming Simulation
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      FastAPI Inference API      Streamlit Dashboard
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 Real-Time Forecasting
```
## 🚀 Core Features

* Real-time electricity price forecasting
* Kafka-based streaming simulation
* Advanced feature engineering for time-series forecasting
* Optuna-powered hyperparameter optimization
* MLflow experiment tracking and artifact management
* FastAPI production inference backend
* Streamlit enterprise dashboard
* ONNX runtime optimization
* Dockerized deployment pipeline
* Modular enterprise-grade project architecture

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| ML Models | XGBoost, LightGBM |
| Optimization | Optuna |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Streaming | Apache Kafka |
| Deployment | Docker |
| Optimization | ONNX Runtime |
| Language | Python |

## 📊 Model Benchmark Results

| Model | Avg Latency | Notes |
|---|---|---|
| Pickle XGBoost | 1-5 ms | Native inference |
| ONNX Runtime | 1-3 ms | Optimized deployment runtime |
| Kafka Streaming | Real-Time | Event-driven forecasting |

## ⚙️ Local Setup

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-electricity-market-forecasting-platform.git
cd ai-electricity-market-forecasting-platform

Create Virtual Environment
    python -m venv venv
Activate Environment
    Windows
        venv\Scripts\activate
Install Dependencies
    pip install -r requirements.txt


🧠 Training Pipeline

Run the complete end-to-end training workflow:

python training_pipeline.py

This pipeline performs:

Data ingestion
Data preprocessing
Feature engineering
Train-validation-test split
Optuna hyperparameter tuning
XGBoost model training
MLflow experiment logging
Model serialization (.pkl)
ONNX model conversion
Benchmark generation

📡 Kafka Streaming Simulation

Start Kafka producer:

python -m src.ingestion.kafka_producer

Start Kafka consumer:

python -m src.ingestion.kafka_consumer

This simulates:

live electricity market streaming
real-time event ingestion
enterprise-style forecasting infrastructure

🚀 Start FastAPI Backend

Launch the production inference API:
python -m uvicorn app.main:app --reload

Open Swagger API docs:
http://127.0.0.1:8000/docs

Available endpoints:

/forecast
/health
/metrics

📈 Launch Streamlit Dashboard

Start the enterprise dashboard UI:

    streamlit run app/dashboard.py

Dashboard features:

Real-time forecast visualization
Actual vs predicted price curves
Market volatility tracking
System health monitoring
Live API integration

⚡ ONNX Optimization

Convert trained XGBoost model to ONNX:

python -m src.optimization.convert_to_onnx

Run latency benchmarking:

python -m src.optimization.benchmark

This compares:

native pickle inference
ONNX runtime inference
📊 Launch MLflow Tracking UI

Start MLflow dashboard:

mlflow ui

Open:

http://127.0.0.1:5000

Track:

experiments
hyperparameters
metrics
artifacts
trained models
🐳 Docker Deployment

Build Docker image:

docker build -t electricity-forecasting-api .

Run containerized API:

docker run -p 8000:8000 electricity-forecasting-api

Access deployed API:

http://localhost:8000/docs

🔄 Git Workflow

Check repository status:

git status

Stage all changes:

git add .

Commit updates:

git commit -m "your commit message"

Push to GitHub:

git push

Pull latest updates:

git pull

Clone repository:

git clone https://github.com/YOUR_USERNAME/ai-electricity-market-forecasting-platform.git