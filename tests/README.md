# Test Suite

Comprehensive test suite covering all components of the housing price prediction pipeline.

## 📁 Directory Structure

```
tests/
├── 🧪 test_features.py         # Feature pipeline tests
├── 🤖 test_training.py        # Training pipeline tests
├── 🎯 test_inference.py      # Inference pipeline tests
├── 🌐 test_api.py             # API endpoint tests
├── 📊 test_data.py            # Test data utilities
└── 📋 conftest.py             # Pytest configuration
```

## 🧪 Test Categories

### Feature Pipeline Tests (`test_features.py`)

**Coverage**: Data loading, preprocessing, feature engineering

**Test Cases**:

```python
def test_time_based_splitting():
    """Verify train/eval/holdout splits are time-aware"""

def test_data_preprocessing():
    """Test deduplication, outlier removal, cleaning"""

def test_feature_engineering():
    """Validate date features, encoding, NaN handling"""

def test_encoder_consistency():
    """Ensure same encoders applied to all datasets"""
```

**Data Quality Tests**:

- No data leakage between splits
- Consistent categorical encoding
- Proper NaN value handling
- Feature schema validation

### Training Pipeline Tests (`test_training.py`)

**Coverage**: Model training, hyperparameter tuning, evaluation

**Test Cases**:

```python
def test_model_training():
    """Verify XGBoost model trains successfully"""

def test_hyperparameter_tuning():
    """Test Optuna optimization process"""

def test_model_evaluation():
    """Validate performance metrics calculation"""

def test_mlflow_tracking():
    """Ensure experiments are logged properly"""
```

**Model Validation Tests**:

- Model serialization/deserialization
- Performance metric calculation
- MLflow experiment tracking
- Hyperparameter optimization

### Inference Pipeline Tests (`test_inference.py`)

**Coverage**: Production inference logic, encoder loading

**Test Cases**:

```python
def test_single_prediction():
    """Test individual house price prediction"""

def test_batch_prediction():
    """Test multiple predictions at once"""

def test_encoder_loading():
    """Verify encoders load correctly from S3"""

def test_schema_alignment():
    """Ensure input schema matches training"""
```

**Production Readiness Tests**:

- S3 integration for model loading
- Consistent preprocessing pipeline
- Error handling for invalid inputs
- Performance under load

### API Tests (`test_api.py`)

**Coverage**: FastAPI endpoints, health checks, error handling

**Test Cases**:

```python
def test_health_endpoint():
    """Test GET /health returns 200"""

def test_predict_endpoint():
    """Test POST /predict with valid data"""

def test_batch_predict_endpoint():
    """Test POST /batch-predict"""

def test_api_documentation():
    """Verify /docs endpoint accessible"""
```

**API Validation Tests**:

- Request/response validation
- Error handling for malformed data
- Rate limiting and authentication
- Load balancer health checks

### Test Data (`test_data.py`)

**Purpose**: Provide consistent test datasets for all tests

**Features**:

- Synthetic housing data with known properties
- Edge cases (missing values, extreme prices)
- Multiple property types and locations
- Time-based test scenarios

## 🛠️ Running Tests

### All Tests

```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_features.py
```

### Test Categories

```bash
# Feature pipeline tests only
pytest tests/test_features.py -v

# Training pipeline tests only
pytest tests/test_training.py -v

# API integration tests
pytest tests/test_api.py -v

# Inference pipeline tests
pytest tests/test_inference.py -v
```

### Parallel Execution

```bash
# Run tests in parallel for speed
pytest -n auto

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Only integration tests
```

## 📊 Test Configuration

### Pytest Configuration (`conftest.py`)

**Fixtures Available**:

- `sample_data` - Small synthetic dataset for testing
- `trained_model` - Pre-trained XGBoost model
- `test_encoders` - Fitted target and frequency encoders
- `s3_client` - Mocked S3 client for testing
- `api_client` - FastAPI test client

**Environment Setup**:

- Test database with temporary files
- Mock AWS credentials for testing
- Isolated test environment
- Consistent random seeds

## 🎯 Test Data Strategy

### Synthetic Data Generation

**Realistic Properties**:

- Price range: $50,000 - $2,000,000
- Bedrooms: 1-6 bedrooms
- Bathrooms: 1-4 bathrooms
- Square footage: 800-5,000 sqft
- Locations: Major US cities with realistic coordinates

**Edge Cases**:

- Missing values in various columns
- Extreme outlier prices
- Invalid categorical values
- Future dates in time features

### Test Splits

**Time-Aware Testing**:

- Train data: Pre-2020 dates
- Eval data: 2020-2021 dates
- Holdout data: 2022+ dates
- Consistent with production pipeline

## 📈 Coverage Goals

### Target Coverage Metrics

**Overall Coverage**: >90% of codebase
**Branch Coverage**: >85% decision paths
**Integration Coverage**: All major workflows tested

**Critical Paths**:

- Data preprocessing pipeline: 100%
- Model training workflow: 100%
- API endpoints: 100%
- Error handling: 95%

## 🚨 Test Categories

### Unit Tests

- Individual function testing
- Mock external dependencies
- Fast execution, isolated testing
- Edge case validation

### Integration Tests

- End-to-end workflow testing
- Real S3 integration (test bucket)
- API-to-model integration
- Database/file system interactions

### Performance Tests

- Load testing for API endpoints
- Memory usage validation
- Response time benchmarks
- Concurrent request handling

## 🐛 Debugging Tests

### Common Test Failures

**Data Issues**:

```bash
# Debug data loading failures
pytest tests/test_features.py::test_data_loading -v -s

# Check feature engineering output
pytest tests/test_features.py::test_feature_engineering -v -s --pdb
```

**Model Issues**:

```bash
# Debug model training
pytest tests/test_training.py::test_model_training -v -s

# Check hyperparameter optimization
pytest tests/test_training.py::test_hyperparameter_tuning -v -s
```

**API Issues**:

```bash
# Debug API endpoints
pytest tests/test_api.py::test_predict_endpoint -v -s

# Test with real data
pytest tests/test_api.py::test_integration -v -s
```

## 🔄 Continuous Integration

### GitHub Actions Integration

**Test Triggers**:

- Every pull request
- Every push to main branch
- Scheduled nightly runs

**Test Results**:

- Coverage reports uploaded as artifacts
- Test results published to PR comments
- Failed tests block deployment

### Quality Gates

**Deployment Requirements**:

- All tests must pass
- Coverage threshold: >85%
- No critical security vulnerabilities
- Performance benchmarks met

## 📝 Best Practices

### Test Writing

**Arrange-Act-Assert Pattern**:

```python
def test_prediction_logic():
    # Arrange
    test_data = create_test_property()
    expected_price = 250000

    # Act
    actual_price = predict_price(test_data)

    # Assert
    assert actual_price == expected_price
```

**Descriptive Test Names**:

- `test_feature_engineering_creates_date_features`
- `test_api_predict_endpoint_returns_200_for_valid_data`
- `test_model_training_improves_with_more_data`

**Test Isolation**:

- No dependency between tests
- Fresh fixtures for each test
- Cleanup after each test
- Deterministic outcomes

### Mock Usage

**External Dependencies**:

- S3 calls mocked for unit tests
- Database connections mocked
- External APIs mocked
- File system isolated

**Real Integration**:

- Separate integration test suite
- Uses real AWS test resources
- Tests actual deployment pipeline
- Validates end-to-end functionality
