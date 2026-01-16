# Source Code Modules

This directory contains all the core application logic organized by functionality.

## 📁 Directory Structure

```
src/
├── 📂 feature_pipeline/     # Data processing & feature engineering
├── 📂 training_pipeline/    # Model training & hyperparameter tuning
├── 📂 inference_pipeline/   # Production inference logic
├── 📂 batch/             # Batch prediction processing
└── 📂 api/              # FastAPI REST service
```

## 🔧 Module Overview

### 📊 Feature Pipeline (`feature_pipeline/`)

**Purpose**: Transform raw housing data into ML-ready features

**Key Files**:

- `load.py` - Time-aware data splitting (train/eval/holdout)
- `preprocess.py` - Data cleaning, deduplication, outlier removal
- `feature_engineering.py` - Feature creation and encoding

**Data Flow**:

```
Raw Data → Time Split → Preprocessing → Feature Engineering → ML Ready
```

### 🤖 Training Pipeline (`training_pipeline/`)

**Purpose**: Train, tune, and evaluate XGBoost models

**Key Files**:

- `train.py` - Baseline model training
- `tune.py` - Hyperparameter optimization with Optuna
- `eval.py` - Model performance evaluation

**MLflow Integration**:

- Experiments tracked in `mlruns/`
- Models and parameters logged automatically
- Best models selected based on validation metrics

### 🎯 Inference Pipeline (`inference_pipeline/`)

**Purpose**: Production-ready model inference with consistent preprocessing

**Key Files**:

- `inference.py` - Core prediction logic
- Applies same transformations as training pipeline
- Loads encoders and models from S3

### 📦 Batch Processing (`batch/`)

**Purpose**: Automated batch predictions on schedule

**Key Files**:

- `run_monthly.py` - Monthly prediction generation
- Outputs predictions to S3 for downstream processing

### 🌐 API Service (`api/`)

**Purpose**: FastAPI REST endpoints for model serving

**Key Files**:

- `main.py` - FastAPI application with all endpoints
- Health checks, prediction endpoints, batch processing
- S3 integration for model/data loading

## 🔄 Data Flow

```
Raw Data → Feature Pipeline → Training Pipeline → Models → Inference Pipeline → API → Users
```

## 🧪 Testing

```bash
# Run tests for specific modules
pytest tests/test_features.py          # Feature pipeline tests
pytest tests/test_training.py         # Training pipeline tests
pytest tests/test_inference.py        # Inference pipeline tests
pytest tests/test_api.py              # API endpoint tests
```

## 🚀 Local Development

```bash
# Run individual modules
python -m src.feature_pipeline.load
python -m src.feature_pipeline.preprocess
python -m src.feature_pipeline.feature_engineering

# Train models
python src/training_pipeline.train
python src/training_pipeline.tune

# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📝 Key Design Principles

- **Modularity**: Each pipeline can run independently
- **Consistency**: Same preprocessing in training and inference
- **Reproducibility**: All random seeds fixed, experiments tracked
- **Production Ready**: Error handling, logging, monitoring built-in
