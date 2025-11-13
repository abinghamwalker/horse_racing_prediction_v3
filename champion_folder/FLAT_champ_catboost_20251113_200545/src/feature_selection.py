#!/usr/bin/env python3
"""
Feature Selection Pipeline for Horse Racing Prediction

Usage:
    python feature_selection.py

This script:
1. Loads feature-engineered data
2. Splits data chronologically (train/val/test)
3. Removes leakage, low-variance, and correlated features
4. Saves model-ready data and selection artifacts

Output:
    - data/model_ready/04_model_ready_data.parquet
    - data/model_ready/feature_selector.pkl
    - data/model_readyfinal_feature_names.txt
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use __file__ to dynamically find the project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(".").resolve()

MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"
REDUCTION_READY_DIR = PROJECT_ROOT / "data" / "reduction_ready"

INPUT_FEATURES_PATH = REDUCTION_READY_DIR / "03_features_engineered.parquet"
OUTPUT_DATA_PATH = MODEL_READY_DIR / "04_model_ready_data.parquet"
SELECTOR_PATH = MODEL_READY_DIR / "feature_selector.pkl"
FEATURE_NAMES_FILE = MODEL_READY_DIR / "final_feature_names.txt"

MODEL_READY_PATH = MODEL_READY_DIR / "04_model_ready_data.parquet"
FINAL_FEATURE_NAMES_FILE = MODEL_READY_DIR / "final_feature_names.txt"

# Create directories
MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Feature selection configuration
TARGET_COLUMN = 'won'

# Columns that are NOT features (identifiers, metadata)
ID_COLS = ['race_id', 'horse', 'race_datetime', 'date_of_race', 'time', 'course', 'track']

# Known leakage columns (post-race information)
KNOWN_LEAKAGE_COLS = [
    # Position/result columns
    'pos', 'place', 'is_winner', 'placed', 'placed_in_betfair_market',
    
    # Return columns (calculated after race)
    'betfair_win_return', 'sp_win_return', 'ew_return', 'place_return',
    'betfair_lay_return', 'place_lay_return',
    
    # Post-race market data
    'ip_min', 'ip_max', 'tick_reduction', 'tick_inflation',
    'pct_bsp_reduction', 'pct_bsp_inflation',
    
    # Race outcome
    'winning_distance',
    
    # Betting columns that might leak
    'betfair_place_sp', 'num_places',
]

# Selection thresholds
VARIANCE_THRESHOLD = 1e-6  # Remove features with essentially no variance
CORR_THRESHOLD = 0.999      # Remove highly correlated features (>99.9% correlation)

# Data split ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# =============================================================================
# FEATURE SELECTOR CLASS
# =============================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for rule-based feature selection.
    
    Removes:
    1. Leakage columns (post-race information)
    2. Low-variance features (essentially constant)
    3. Highly correlated features (redundant information)
    
    Designed to prevent data leakage by fitting only on training data.
    """
    
    def __init__(
        self,
        leakage_cols: List[str],
        id_cols: List[str],
        target_col: str,
        variance_threshold: float = 1e-6,
        corr_threshold: float = 0.999
    ):
        self.leakage_cols = leakage_cols
        self.id_cols = id_cols
        self.target_col = target_col
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        
        # Will be set during fit
        self.features_to_keep_ = []
        self.features_removed_ = {}
        self.n_features_in_ = 0
        self.n_features_out_ = 0
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the selector on training data.
        
        Args:
            X: DataFrame with all columns (IDs, target, features)
            y: Ignored (for sklearn compatibility)
        
        Returns:
            self
        """
        logger.info("Fitting FeatureSelector...")
        
        # Get list of feature columns (exclude IDs and target)
        all_cols_to_exclude = set(self.id_cols + [self.target_col])
        initial_features = [col for col in X.columns if col not in all_cols_to_exclude]
        self.n_features_in_ = len(initial_features)
        
        logger.info(f"  Initial features: {self.n_features_in_}")
        
        # --- Step 1: Remove Leakage Features ---
        leaky = [col for col in self.leakage_cols if col in initial_features]
        self.features_removed_['leakage'] = leaky
        
        if leaky:
            logger.info(f"  Removing {len(leaky)} leakage features")
        
        # --- Step 2: Remove Low Variance Features ---
        # Only check numeric columns that aren't already removed
        remaining_features = [col for col in initial_features if col not in leaky]
        numeric_features = X[remaining_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_features:
            variances = X[numeric_features].var(ddof=0)
            low_variance = variances[variances < self.variance_threshold].index.tolist()
            self.features_removed_['low_variance'] = low_variance
            
            if low_variance:
                logger.info(f"  Removing {len(low_variance)} low-variance features")
        else:
            self.features_removed_['low_variance'] = []
        
        # --- Step 3: Remove Highly Correlated Features ---
        # Only check numeric columns that haven't been removed yet
        cols_removed_so_far = set(leaky + self.features_removed_['low_variance'])
        remaining_numeric = [col for col in numeric_features if col not in cols_removed_so_far]
        
        if len(remaining_numeric) > 1:
            try:
                corr_matrix = X[remaining_numeric].corr().abs()
                
                # Get upper triangle of correlation matrix
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find features with correlation > threshold
                highly_correlated = [
                    col for col in upper.columns 
                    if any(upper[col] > self.corr_threshold)
                ]
                
                self.features_removed_['highly_correlated'] = highly_correlated
                
                if highly_correlated:
                    logger.info(f"  Removing {len(highly_correlated)} highly correlated features")
            except Exception as e:
                logger.warning(f"  Could not compute correlations: {e}")
                self.features_removed_['highly_correlated'] = []
        else:
            self.features_removed_['highly_correlated'] = []
        
        # --- Step 4: Compute Final Feature List ---
        all_removed = set()
        for reason, cols in self.features_removed_.items():
            all_removed.update(cols)
        
        final_feature_cols = [col for col in initial_features if col not in all_removed]
        self.n_features_out_ = len(final_feature_cols)
        
        # Final columns to keep includes IDs, target, and selected features
        self.features_to_keep_ = self.id_cols + [self.target_col] + final_feature_cols
        
        logger.info(f"  Features removed: {len(all_removed)}")
        logger.info(f"  Features kept: {self.n_features_out_}")
        logger.info("✅ FeatureSelector fitted")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by keeping only selected features.
        
        Args:
            X: DataFrame with all columns
        
        Returns:
            DataFrame with only selected columns
        """
        if not self.features_to_keep_:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        
        # Only select columns that exist in X
        cols_to_select = [col for col in self.features_to_keep_ if col in X.columns]
        
        return X[cols_to_select].copy()
    
    def get_feature_names_out(self):
        """Returns list of selected feature names (excluding IDs and target)."""
        return [col for col in self.features_to_keep_ 
                if col not in self.id_cols + [self.target_col]]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_input_data(df: pd.DataFrame) -> None:
    """Validates that input data has required columns."""
    required_cols = ['race_datetime', 'won']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not pd.api.types.is_datetime64_any_dtype(df['race_datetime']):
        raise ValueError("'race_datetime' must be datetime type")
    
    if df['won'].dtype not in [np.int64, np.int32]:
        raise ValueError("'won' must be integer type")


def create_temporal_splits(df: pd.DataFrame, test_size: float, val_size: float):
    """
    Splits data chronologically into train/val/test sets.
    
    Args:
        df: DataFrame sorted by race_datetime
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
    
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    
    # Calculate split points
    test_start_idx = int(n * (1 - test_size))
    train_end_idx = int(test_start_idx * (1 - val_size))
    
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
    logger.info(f"    Date range: {train_df['race_datetime'].min()} to {train_df['race_datetime'].max()}")
    logger.info(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
    logger.info(f"    Date range: {val_df['race_datetime'].min()} to {val_df['race_datetime'].max()}")
    logger.info(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
    logger.info(f"    Date range: {test_df['race_datetime'].min()} to {test_df['race_datetime'].max()}")
    
    return train_df, val_df, test_df


def log_removal_report(selector: FeatureSelector) -> None:
    """Logs detailed report of removed features."""
    logger.info("\n" + "="*80)
    logger.info("FEATURE REMOVAL REPORT")
    logger.info("="*80)
    
    total_removed = 0
    for reason, cols in selector.features_removed_.items():
        if cols:
            logger.info(f"\n{reason.upper()}: {len(cols)} features removed")
            total_removed += len(cols)
            
            # Show first 10 examples
            for col in cols[:10]:
                logger.info(f"  - {col}")
            
            if len(cols) > 10:
                logger.info(f"  ... and {len(cols) - 10} more")
    
    logger.info("\n" + "-"*80)
    logger.info(f"Total features removed: {total_removed}")
    logger.info(f"Total features kept: {selector.n_features_out_}")
    logger.info("="*80)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main feature selection pipeline."""
    try:
        logger.info("="*80)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("="*80)
        
        # --- Step 1: Verify Input ---
        if not INPUT_FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Input file not found: {INPUT_FEATURES_PATH}\n"
                f"Run feature_engineering.py first."
            )
        
        # --- Step 2: Load Data ---
        logger.info(f"\nLoading feature-engineered data from: {INPUT_FEATURES_PATH}")
        df = pd.read_parquet(INPUT_FEATURES_PATH)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate
        validate_input_data(df)
        
        # Sort chronologically (critical for temporal split)
        df = df.sort_values('race_datetime').reset_index(drop=True)
        logger.info("✅ Data sorted chronologically")
        
        # --- Step 3: Create Temporal Splits ---
        logger.info("\n" + "="*80)
        logger.info("CREATING TEMPORAL SPLITS")
        logger.info("="*80)
        
        train_df, val_df, test_df = create_temporal_splits(df, TEST_SIZE, VAL_SIZE)
        
        # --- Step 4: Fit Selector on Training Data Only ---
        logger.info("\n" + "="*80)
        logger.info("FITTING FEATURE SELECTOR")
        logger.info("="*80)
        
        selector = FeatureSelector(
            leakage_cols=KNOWN_LEAKAGE_COLS,
            id_cols=ID_COLS,
            target_col=TARGET_COLUMN,
            variance_threshold=VARIANCE_THRESHOLD,
            corr_threshold=CORR_THRESHOLD
        )
        
        # CRITICAL: Fit only on training data to prevent leakage
        selector.fit(train_df)
        
        # Log removal report
        log_removal_report(selector)
        
        # --- Step 5: Transform All Splits ---
        logger.info("\n" + "="*80)
        logger.info("TRANSFORMING DATA SPLITS")
        logger.info("="*80)
        
        train_selected = selector.transform(train_df)
        val_selected = selector.transform(val_df)
        test_selected = selector.transform(test_df)
        
        logger.info(f"Train shape: {train_selected.shape}")
        logger.info(f"Val shape:   {val_selected.shape}")
        logger.info(f"Test shape:  {test_selected.shape}")
        
        # --- Step 6: Save Artifacts ---
        logger.info("\n" + "="*80)
        logger.info("SAVING ARTIFACTS")
        logger.info("="*80)
        
        # Save fitted selector
        logger.info(f"Saving selector to: {SELECTOR_PATH}")
        with open(SELECTOR_PATH, 'wb') as f:
            pickle.dump(selector, f)
        
        # Combine and save final dataset
        final_df = pd.concat([train_selected, val_selected, test_selected], ignore_index=True)
        final_df = final_df.sort_values('race_datetime').reset_index(drop=True)
        
        logger.info(f"Saving model-ready data to: {OUTPUT_DATA_PATH}")
        final_df.to_parquet(OUTPUT_DATA_PATH, index=False)
        
        # Save feature names
        feature_names = selector.get_feature_names_out()
        logger.info(f"Saving feature names to: {FEATURE_NAMES_FILE}")
        with open(FEATURE_NAMES_FILE, 'w') as f:
            f.write('\n'.join(sorted(feature_names)))
        
        # --- Step 7: Final Summary ---
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"\nFinal dataset:")
        logger.info(f"  Shape: {final_df.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  Rows: {len(final_df):,}")
        
        logger.info("\nOutput files:")
        logger.info(f"  1. {OUTPUT_DATA_PATH.name} - Model-ready data")
        logger.info(f"  2. {SELECTOR_PATH.name} - Fitted selector")
        logger.info(f"  3. {FEATURE_NAMES_FILE.name} - Feature list")
        
        logger.info("\nData splits saved:")
        logger.info(f"  Train: rows 0-{len(train_selected)-1}")
        logger.info(f"  Val:   rows {len(train_selected)}-{len(train_selected)+len(val_selected)-1}")
        logger.info(f"  Test:  rows {len(train_selected)+len(val_selected)}-{len(final_df)-1}")
        
        # Show sample
        logger.info("\nSample of final data:")
        print(final_df.head())
        
        logger.info("\n" + "="*80)
        logger.info("Next step: python src/train_model.py")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("❌ PIPELINE FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())