## MLOPs Project using Regression on Time Series Data

## 🌟 Live Application

- **🎯 Frontend Dashboard**: [http://housing-alb-1577325325.us-east-1.elb.amazonaws.com:8080/dashboard/](http://housing-alb-1577325325.us-east-1.elb.amazonaws.com:8080/dashboard/)
- **🔧 Backend API**: [http://housing-alb-1577325325.us-east-1.elb.amazonaws.com/](http://housing-alb-1577325325.us-east-1.elb.amazonaws.com/)
- **📚 API Documentation**: [http://housing-alb-1577325325.us-east-1.elb.amazonaws.com/docs](http://housing-alb-1577325325.us-east-1.elb.amazonaws.com/docs)

> **⚠️ Performance Tip**: When using the dashboard, **select a specific region** rather than "All Regions" to avoid load balancer timeouts due to large dataset processing.

## Project Overview

Housing Price Regressor is an end-to-end MLOPs project that includes every step of a production grade machine learning pipeline for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization using Docker, AWS cloud deployment, and comprehensive testing using pytest. The system includes both a REST API backend using FastAPI and a Streamlit dashboard frontend for interactive predictions.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering

  - `load.py`: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
  - `preprocess.py`: City normalization, deduplication, outlier removal
  - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`**: Model training and hyperparameter optimization

  - `train.py`: Baseline XGBoost training with configurable parameters
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow integration
  - `eval.py`: Model evaluation and metrics calculation

- **`src/inference_pipeline/`**: Production inference

  - `inference.py`: Applies same preprocessing/encoding transformations using saved encoders

- **`src/batch/`**: Batch prediction processing

  - `run_monthly.py`: Generates monthly predictions on holdout data

- **`src/api/`**: FastAPI web service
  - `main.py`: REST API with S3 integration, health checks, prediction endpoints, and batch processing

### Web Applications

- **`app.py`**: Streamlit dashboard for interactive housing price predictions
  - Real-time predictions via FastAPI integration
  - Interactive filtering by year, month, and region
  - Visualization of predictions vs actuals with metrics (MAE, RMSE, % Error)
  - Yearly trend analysis with highlighted selected periods

### Cloud Infrastructure & Deployment

**🏗️ Production Architecture**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub       │───▶│     ECR        │───▶│      ECS       │
│   Repository    │    │  Container     │    │   Cluster      │
│                │    │   Registry      │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      S3        │◀───│   Load Balancer │◀───│   Users        │
│   Data Store   │    │      (ALB)      │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**🛡️ AWS Services**:

- **S3**: Data and model storage (`housing-price-regressor` bucket)
- **ECR**: Docker container registry
- **ECS**: Container orchestration with Fargate (`housing-cluster`)
- **ALB**: Application Load Balancer (`housing-alb-1577325325.us-east-1.elb.amazonaws.com`)
- **CloudWatch**: Logging and monitoring

**🚀 CI/CD Pipeline**:

1. **Code Push** to main branch → Triggers GitHub Actions
2. **Build Docker** images for API and Streamlit
3. **Push to ECR** with SHA-based tagging
4. **Update ECS** services with force-new-deployment
5. **Health Check** deployment status via CloudWatch

**🔐 Environment Variables**:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (recommended to store in GitHub Secrets)
- `AWS_REGION`: `Your AWS Region`
- `S3_BUCKET`: `Your S3 Bucket Name`

- **AWS S3 Integration**: Data and model storage in `S3_BUCKET` bucket
- **Amazon ECR**: Container registry for Docker images
- **Amazon ECS**: Container orchestration with Fargate
- **Application Load Balancer**: Traffic distribution and routing
- **CI/CD Pipeline**: Automated deployment via GitHub Actions

#### ECS Services:

- **housing-api-service**: FastAPI backend (port 8000, 1024 CPU, 3072 MB memory)
- **housing-streamlit-service**: Streamlit dashboard (port 8501, 512 CPU, 1024 MB memory)

### Data Leakage Prevention

The project implements strict data leakage prevention:

- Time-based splits (not random)
- Encoders fitted only on training data
- Leakage-prone columns dropped before training
- Schema alignment enforced between train/eval/inference

## Common Commands

### Environment Setup

```bash
# Install dependencies using uv
uv sync
```

### Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_training.py
pytest tests/test_inference.py

# Run with verbose output
pytest -v
```

### Data Pipeline

```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline

```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

### Inference

```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv

# Batch monthly predictions
python src/batch/run_monthly.py
```

### API Service

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit Dashboard

```bash
# Start Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

```bash
# Build API container
docker build -t housing-regression .

# Build Streamlit container
docker build -t housing-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 housing-regression

# Run Streamlit container
docker run -p 8501:8501 housing-streamlit
```

### MLflow Tracking

```bash
# Start MLflow UI (view experiments)
mlflow ui
```

## 🚨 Troubleshooting

### Common Production Issues

**⏱️ Load Balancer Timeouts**:

- **Cause**: Selecting "All Regions" processes too much data (>100k records)
- **Solution**: Always select specific regions for better performance
- **Recommendation**: Use region filtering for faster response times

**🚀 Container Startup Issues**:

- **Check**: CloudWatch logs (`/ecs/housing-api`, `/ecs/housing-streamlit`)
- **Verify**: IAM role `ecsTaskExecutionRole` has S3 and CloudWatch permissions
- **Ensure**: Task definitions have correct environment variables

**🔗 API Connection Issues**:

- **Frontend**: Verify `API_URL` points to ALB, not internal service name
- **Backend**: Check health endpoint `GET /health` returns 200
- **Network**: Confirm security groups allow ALB → container traffic

**🐛 NaN JSON Serialization Error**:

- **Cause**: `InvalidJSONError: Out of range float values are not JSON compliant: nan`
- **Solution**: Ensure `lat` and `lng` columns are filled with 0 in feature engineering pipeline
- **Fix Location**: `src/feature_pipeline/feature_engineering.py`

### Debug Commands

```bash
# Check ECS service status
aws ecs describe-services --cluster housing-cluster --services housing-api-service housing-streamlit-service

# Check running tasks
aws ecs list-tasks --cluster housing-cluster

# Check target group health
aws elbv2 describe-target-health --target-group-arn <TG_ARN>

# Check container logs
aws logs tail /ecs/housing-api --follow
aws logs tail /ecs/housing-streamlit --follow
```

## Key Design Patterns

### Pipeline Modularity

Each pipeline component can be run independently with consistent interfaces. All modules accept configurable input/output paths for testing isolation.

### Cloud-Native Architecture

- **S3-First Storage**: Models and data automatically sync from S3 buckets
- **Containerized Services**: Both API and dashboard run in Docker containers
- **Auto-scaling Infrastructure**: ECS Fargate provides serverless container scaling
- **Environment-based Configuration**: Separate configs for local development and production

### Encoder Persistence

Frequency and target encoders are saved as pickle files during training and loaded during inference to ensure consistent transformations.

### Configuration Management

Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through function parameters or environment variables. Production deployments use AWS environment variables.

### Testing Strategy

- Unit tests for individual pipeline components
- Integration tests for end-to-end pipeline flows
- Smoke tests for inference pipeline
- All tests use temporary directories to avoid touching production data

## Dependencies

Key production dependencies (see `pyproject.toml`):

- **ML/Data**: `xgboost==3.0.4`, `scikit-learn`, `pandas==2.1.1`, `numpy==1.26.4`
- **API**: `fastapi`, `uvicorn`
- **Dashboard**: `streamlit`, `plotly`
- **Cloud**: `boto3` (AWS integration)
- **Experimentation**: `mlflow`, `optuna`
- **Quality**: `great-expectations`, `evidently`

## File Structure Notes

- **`data/`**: Raw, processed, and prediction data (time-structured, S3-synced)
- **`models/`**: Trained models and encoders (pkl files, S3-synced)
- **`mlruns/`**: MLflow experiment tracking data
- **`configs/`**: YAML configuration files
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Comprehensive test suite with sample data
- **AWS Task Definitions**: `housing-api-task-def.json`, `streamlit-task-def.json`
- **CI/CD**: `.github/workflows/ci.yml` for automated deployment
