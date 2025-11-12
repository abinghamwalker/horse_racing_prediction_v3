# Horse Racing Prediction Pipeline

A production-ready betting system for predicting horse race winners using gradient boosting models (XGBoost, CatBoost, LightGBM).

## 🎯 Project Goal

Build a profitable horse racing betting system that:
- Predicts win probability at race start
- Uses only information available before betting
- Prevents data leakage through rigorous temporal validation
- Optimizes for ROI, not just accuracy

## 🏗️ Project Structure

```
horse_racing_pipeline/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── constants.py          # All configuration constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extraction.py         # Data loading and preparation
│   │   └── validation.py         # Data validation utilities
│   ├── features/
│   │   └── (coming from notebook 03)
│   ├── models/
│   │   └── (coming from notebook 05)
│   └── backtesting/
│       └── (coming from notebook 06)
├── tests/
│   └── test_extraction.py        # Unit tests
├── data/
│   ├── raw/                      # Raw CSV files
│   ├── processed/                # Cleaned data
│   └── model_ready/              # Feature-engineered data
├── notebooks/                     # Original Jupyter notebooks
└── production_models/             # Trained models
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data import load_and_prepare_data

# Load and prepare data (removes leakage, creates target)
df = load_and_prepare_data()

# Access features and target
from src.data import split_features_target
X, y = split_features_target(df)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Win rate: {y.mean()*100:.2f}%")
```

## 🔒 Data Leakage Prevention

The pipeline automatically removes these post-race columns:

**Result Columns (known only after race):**
- `Place` (finishing position - converted to binary target `Won`)
- `Winning Distance`
- `SP Win Return`, `E/W Return`, `Betfair Win Return`
- `Place Return`, `Betfair Lay Return`, `Place Lay Return`

**In-Play Columns (not available at bet time):**
- `IP Min`, `IP Max` (in-play prices)

**Post-Race Analysis:**
- `Tick reduction`, `Tick inflation`
- `% BSP reduction`, `% BSP inflation`

## 📊 Data Validation

The pipeline performs comprehensive validation:

```python
from src.data import validate_dataset

df, summary = validate_dataset(
    df,
    check_target=True,
    target_col='Won'
)

# Summary includes:
# - Temporal ordering checks
# - Duplicate detection
# - Win rate statistics
# - Yearly distribution
# - Data quality metrics
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_extraction.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Module Documentation

### `src.config.constants`

Contains all configuration constants:
- File paths
- Column names
- Leakage column lists
- Validation parameters

### `src.data.extraction`

Main data loading and preparation functions:

- `load_raw_data()` - Load CSV or Parquet files
- `load_and_prepare_data()` - Complete pipeline (recommended)
- `prepare_temporal_features()` - Convert dates, sort temporally
- `create_target_variable()` - Create binary win indicator
- `remove_leakage_columns()` - Remove post-race information
- `split_features_target()` - Split X and y

### `src.data.validation`

Data validation utilities:

- `validate_temporal_ordering()` - Check date ranges, future dates
- `validate_row_count()` - Minimum row validation
- `check_duplicates()` - Find duplicate rows
- `validate_win_rate()` - Check win rate statistics
- `validate_yearly_distribution()` - Yearly race distribution

## 🎓 Development Workflow

1. **Data Extraction** (✅ Complete - from notebook 01)
   - Load raw data
   - Remove leakage
   - Create target variable

2. **Data Cleaning** (Next - from notebook 02)
   - Handle missing values
   - Fix data types
   - Outlier detection

3. **Feature Engineering** (From notebook 03)
   - Historical statistics
   - Time-based features
   - Market features

4. **Feature Selection** (From notebook 04)
   - Reduce dimensionality
   - Remove correlated features

5. **Model Training** (From notebook 05)
   - XGBoost, CatBoost, LightGBM
   - Hyperparameter tuning
   - Cross-validation

6. **Backtesting** (From notebook 06)
   - Out-of-time validation
   - ROI calculation
   - Strategy evaluation

## ⚙️ Configuration

Edit `src/config/constants.py` to customize:

```python
# Data paths
RAW_DATA_FILE = "your_data.csv"

# Validation
EXPECTED_MIN_ROWS = 1000
DATA_START_DATE = "2020-01-01"

# Random seed
RANDOM_SEED = 42
```

## 🐛 Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.data import load_and_prepare_data
df = load_and_prepare_data()
```

## 📈 Next Steps

- [ ] Complete data cleaning module (notebook 02)
- [ ] Implement feature engineering (notebook 03)
- [ ] Add feature selection (notebook 04)
- [ ] Create model training pipeline (notebook 05)
- [ ] Build backtesting framework (notebook 06)
- [ ] Add API for real-time predictions
- [ ] Create deployment scripts

## 📄 License

[Your License]

## 👤 Author

[Your Name]

---

**⚠️ Disclaimer:** This is a betting system. Past performance does not guarantee future results. Bet responsibly.
