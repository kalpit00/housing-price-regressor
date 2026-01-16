# Notebooks Directory

Exploratory data analysis and experimentation notebooks for the housing price prediction project.

## 📁 Directory Structure

```
notebooks/
├── 📊 01_eda_exploration.ipynb     # Initial data exploration
├── 🔧 02_feature_engineering.ipynb    # Feature development and testing
├── 🤖 03_model_experiments.ipynb      # Model experimentation and comparison
├── 📈 04_model_evaluation.ipynb       # Model performance analysis
└── 🚀 05_deployment_analysis.ipynb    # Production deployment monitoring
```

## 📊 Notebook Categories

### 01_eda_exploration.ipynb

**Purpose**: Initial understanding of the housing dataset

**Analyses**:

- Data distribution and summary statistics
- Missing value patterns and data quality
- Price trends over time and by location
- Feature correlations and relationships
- Outlier detection and data cleaning decisions

**Key Visualizations**:

- Price distribution histograms
- Geographic price maps
- Time series price trends
- Feature correlation heatmaps
- Missing value matrices

### 02_feature_engineering.ipynb

**Purpose**: Develop and validate feature engineering approaches

**Experiments**:

- Date feature creation (year, month, day of week)
- Categorical encoding strategies (target vs frequency)
- Geographic feature engineering
- Feature importance analysis
- Data leakage prevention validation

**Validation**:

- Feature impact on model performance
- Encoding consistency across data splits
- Computational efficiency comparison

### 03_model_experiments.ipynb

**Purpose**: Compare different modeling approaches

**Experiments**:

- XGBoost vs Random Forest vs Linear Regression
- Hyperparameter sensitivity analysis
- Cross-validation strategy comparison
- Feature selection experiments
- Ensemble methods testing

**Results Tracking**:

- MLflow experiment logging
- Performance metric comparison
- Training time analysis
- Model complexity vs accuracy trade-offs

### 04_model_evaluation.ipynb

**Purpose**: Comprehensive model performance analysis

**Analyses**:

- Error distribution by price range
- Geographic prediction accuracy
- Temporal performance drift
- Feature importance validation
- Residual analysis and patterns

**Visualizations**:

- Predicted vs actual scatter plots
- Residual plots and QQ plots
- Feature importance charts
- Error heatmaps by region
- Time series performance tracking

### 05_deployment_analysis.ipynb

**Purpose**: Monitor and analyze production model performance

**Monitoring**:

- Live API performance metrics
- Prediction drift detection
- User behavior analysis
- System resource utilization
- Error pattern analysis

**Optimization**:

- Model retraining scheduling
- A/B testing framework
- Performance bottleneck identification
- Cost optimization analysis

## 🛠️ Notebook Environment

### Setup Instructions

```bash
# Activate virtual environment
source venv/bin/activate

# Install Jupyter and dependencies
pip install jupyterlab
pip install matplotlib seaborn plotly

# Start Jupyter Lab
jupyter lab
```

### Kernel Configuration

**Recommended Kernel**: Python 3 (venv)
**Required Extensions**:

- `jupyter-contrib-nbextensions`
- `ipywidgets` for interactive plots
- `matplotlib` for static visualizations
- `plotly` for interactive charts

### Data Access

**Notebooks automatically sync** with project data:

```python
# Access project data
import sys
sys.path.append('..')
from src.feature_pipeline import load_data

# Or use relative paths
data_path = '../data/processed/feature_engineered_train.csv'
```

## 📈 Analysis Workflow

### 1. Data Understanding

```python
# Load raw data
df = pd.read_csv('../data/raw/housing_listings.csv')

# Basic statistics
print(df.describe())
print(df.info())

# Visualize distributions
import matplotlib.pyplot as plt
df['price'].hist(bins=50)
plt.show()
```

### 2. Feature Development

```python
# Test new features
def create_new_features(df):
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['age'] = 2023 - df['year_built']
    return df

# Validate impact
from sklearn.model_selection import cross_val_score
score = cross_val_score(model, X, y, cv=5)
print(f"New feature CV score: {score.mean():.4f}")
```

### 3. Model Experimentation

```python
# Log experiments to MLflow
import mlflow

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({'rmse': rmse, 'mae': mae})

    # Save model
    mlflow.sklearn.log_model(model, "model")
```

### 4. Production Validation

```python
# Test on holdout set
holdout_data = pd.read_csv('../data/processed/feature_engineered_holdout.csv')
predictions = model.predict(holdout_data[features])

# Calculate metrics
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(holdout_data['price'], predictions)
print(f"Holdout MAE: ${mae:,.0f}")
```

## 🎯 Key Findings

### Data Insights

**Price Distribution**:

- Median price: ~$350,000
- Right-skewed distribution (few luxury outliers)
- Strong geographic variation
- Seasonal patterns present

**Feature Importance**:

- Location (lat/lng) most important
- Square footage strong predictor
- Bedroom/bathroom count significant
- Date features provide incremental value

### Model Performance

**Best Approach**: XGBoost with target encoding

- **MAE**: ~$50,000 on validation set
- **R²**: ~0.75 explanatory power
- **Robust**: Consistent across time periods
- **Efficient**: <1s prediction time

### Production Considerations

**Accuracy by Price Range**:

- Budget homes (<$150k): Higher error percentage
- Mid-range ($150k-500k): Best performance
- Luxury (>$500k): Lower error but fewer samples

**Regional Variation**:

- Urban areas: Better prediction accuracy
- Rural areas: Higher variance
- Data density impacts performance
- Local market factors important

## 🔄 Experiment Tracking

### MLflow Integration

**Experiment Organization**:

```
mlruns/
├── experiment_1/    # Baseline models
├── experiment_2/    # Feature engineering tests
├── experiment_3/    # Hyperparameter tuning
└── experiment_4/    # Final model selection
```

**Logged Metrics**:

- Training and validation scores
- Hyperparameter combinations
- Feature importance rankings
- Training time and resource usage
- Model serialization metadata

### Reproducibility

**Random Seeds**:

- Fixed seeds for all random processes
- Consistent train/test splits
- Reproducible hyperparameter search
- Deterministic model training

## 🚨 Common Issues

### Notebook Challenges

**Memory Usage**:

- Large datasets may exceed RAM
- **Solution**: Use chunking or sampling
- **Monitoring**: Track memory with `!free -h`

**Computation Time**:

- Complex experiments can be slow
- **Solution**: Use parallel processing
- **Optimization**: Profile and optimize bottlenecks

### Version Control

**Notebook Size**:

- Large notebooks difficult to diff
- **Solution**: Modularize code into scripts
- **Best Practice**: Use notebooks for exploration, scripts for production

## 📝 Best Practices

### Notebook Organization

**Cell Structure**:

1. Import libraries and setup
2. Load and prepare data
3. Analysis/experimentation
4. Results and visualization
5. Insights and next steps

**Documentation**:

- Markdown cells explaining methodology
- Code comments for complex logic
- Results interpretation and conclusions
- Next experiment suggestions

### Collaboration

**Sharing**:

- Export notebooks to HTML for sharing
- Use relative paths for portability
- Clear output cell results
- Version control key findings

### Production Transition

**From Notebook to Script**:

- Extract working code to `.py` files
- Add proper error handling
- Include logging and monitoring
- Test with production data

## 🚀 Deployment Notes

### When Ready for Production

**Sign-off Criteria**:

- Model performance meets requirements
- Code is modular and tested
- Documentation is complete
- Production deployment tested

**Deployment Steps**:

1. Finalize model in `models/` directory
2. Update API to use new model version
3. Test end-to-end pipeline
4. Deploy via GitHub Actions CI/CD
5. Monitor production performance
