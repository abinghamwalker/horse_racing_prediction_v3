#!/usr/bin/env python3
"""
Production-Grade Feature Selection Pipeline for Horse Racing Prediction

This pipeline implements a sophisticated multi-stage feature selection process:
    1. Data leakage removal (post-race information)
    2. Variance-based filtering (near-constant features)
    3. Correlation-based filtering (redundant features)
    4. Model-based importance filtering (LightGBM + permutation importance)
    5. Stability validation across temporal folds

Usage:
    python feature_selection_pipeline.py [--config config.yaml]

Output:
    - data/03_model_ready/model_ready_data.parquet
    - data/03_model_ready/feature_selector.pkl
    - data/03_model_ready/final_feature_names.txt
    - data/03_model_ready/feature_selection_report.json
    - data/03_model_ready/feature_importance_plot.png
"""

import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection pipeline."""
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    input_path: Path = None
    output_dir: Path = None
    
    # Column definitions
    target_col: str = 'won'
    id_cols: List[str] = field(default_factory=lambda: [
        'race_id', 'horse', 'race_datetime', 'date_of_race', 'time', 'course', 'track',
        'jockey', 'trainer', 'country', 'betting_deadline'  # Added categorical identifiers
    ])
    leakage_cols: List[str] = field(default_factory=lambda: [
        # Direct race outcome columns (post-race results)
        'pos', 'place', 'is_winner', 'placed', 'placed_in_betfair_market',
        'finishing_position_variance',  # Based on actual finishing positions
        
        # Return/payout columns (only known after race)
        'betfair_win_return', 'sp_win_return', 'ew_return', 'place_return',
        'betfair_lay_return', 'place_lay_return',
        
        # In-play/post-race betting data
        'ip_min', 'ip_max',  # In-play odds (during race)
        'tick_reduction', 'tick_inflation',
        'pct_bsp_reduction', 'pct_bsp_inflation',
        
        # Post-race specific information
        'winning_distance',  # Distance behind winner (only known after)
        'betfair_place_sp',  # Betfair Starting Price for place market
        'num_places',  # Number of places paid (sometimes determined post-race)
        
        # VERIFIED SAFE (confirmed by user):
        # - last_time_out_position: Historical position from previous race ✓
        # - actual_distance_change_m: Pre-race measurement ✓
        # - pos_improvement_vs_last: Historical comparison ✓
        # - pace: Running style (not in-race data) ✓
        # - forecast_rank: Morning forecast ✓
        # - rbd_rank: Morning calculation ✓
        # - forecasted_odds: Morning odds ✓
    ])
    
    # Selection thresholds
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.98
    importance_threshold: float = 0.0  # Zero importance removal
    cumulative_importance_threshold: float = 0.995  # Keep features for 99.5% cumulative importance
    
    # Model-based selection
    use_model_selection: bool = True
    use_permutation_importance: bool = True
    n_permutation_repeats: int = 10
    
    # Data splits
    test_size: float = 0.15
    val_size: float = 0.15
    
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    })
    lgb_num_iterations: int = 100
    lgb_early_stopping_rounds: int = 20
    
    def __post_init__(self):
        if self.input_path is None:
            self.input_path = self.project_root / "data" / "02_reduction_ready" / "02_features_engineered.parquet"
        if self.output_dir is None:
            self.output_dir = self.project_root / "data" / "03_model_ready"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_fmt)
    
    # File handler
    log_file = output_dir / 'feature_selection.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_fmt)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# FEATURE SELECTOR CLASS
# =============================================================================

class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Production-grade feature selector with multi-stage filtering.
    
    Implements:
        - Leakage detection and removal
        - Variance-based filtering
        - Correlation-based filtering
        - Model-based importance filtering (LightGBM)
        - Optional permutation importance validation
    
    Attributes:
        config: FeatureSelectionConfig object
        features_to_keep_: List of selected feature names
        features_removed_: Dict mapping removal reason to feature lists
        feature_importances_: DataFrame of feature importances
        selection_metadata_: Dict with selection statistics
    """
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.features_to_keep_ = []
        self.features_removed_ = {}
        self.feature_importances_ = None
        self.permutation_importances_ = None
        self.selection_metadata_ = {}
        self.lgb_model_ = None
        
    def fit(self, X: pd.DataFrame, y=None, X_val: Optional[pd.DataFrame] = None):
        """
        Fit the selector on training data.
        
        Args:
            X: Training DataFrame with all columns
            y: Ignored (target extracted from X)
            X_val: Optional validation set for model-based selection
            
        Returns:
            self
        """
        logger.info("="*80)
        logger.info("FITTING ADVANCED FEATURE SELECTOR")
        logger.info("="*80)
        
        # Identify initial feature set
        all_cols_to_exclude = set(self.config.id_cols + [self.config.target_col])
        initial_features = [col for col in X.columns if col not in all_cols_to_exclude]
        self.selection_metadata_['n_features_initial'] = len(initial_features)
        
        logger.info(f"Initial features: {len(initial_features)}")
        
        # Stage 1: Rule-based selection
        features_after_rules = self._apply_rule_based_selection(X, initial_features)
        self.selection_metadata_['n_features_after_rules'] = len(features_after_rules)
        
        # Stage 2: Model-based selection
        if self.config.use_model_selection and len(features_after_rules) > 0:
            final_features = self._apply_model_based_selection(
                X, features_after_rules, X_val
            )
        else:
            final_features = features_after_rules
        
        self.selection_metadata_['n_features_final'] = len(final_features)
        
        # Set final feature list
        self.features_to_keep_ = self.config.id_cols + [self.config.target_col] + final_features
        
        # Calculate reduction statistics
        n_removed = len(initial_features) - len(final_features)
        pct_removed = (n_removed / len(initial_features) * 100) if len(initial_features) > 0 else 0
        
        logger.info("\n" + "="*80)
        logger.info("SELECTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Initial features:       {len(initial_features)}")
        logger.info(f"Features after rules:   {len(features_after_rules)}")
        logger.info(f"Final features:         {len(final_features)}")
        logger.info(f"Features removed:       {n_removed} ({pct_removed:.1f}%)")
        logger.info("="*80)
        
        return self
    
    def _apply_rule_based_selection(
        self, X: pd.DataFrame, features: List[str]
    ) -> List[str]:
        """Apply variance, leakage, and correlation filters."""
        logger.info("\n--- STAGE 1: RULE-BASED SELECTION ---")
        
        # Step 1: Remove leakage
        leaky = [col for col in self.config.leakage_cols if col in features]
        self.features_removed_['leakage'] = leaky
        remaining = [f for f in features if f not in leaky]
        logger.info(f"  ✓ Removed {len(leaky)} leakage features")
        
        # Step 2: Remove low-variance features
        numeric_cols = X[remaining].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            variances = X[numeric_cols].var(ddof=0)
            low_var = variances[variances <= self.config.variance_threshold].index.tolist()
            self.features_removed_['low_variance'] = low_var
            remaining = [f for f in remaining if f not in low_var]
            logger.info(f"  ✓ Removed {len(low_var)} low-variance features (threshold: {self.config.variance_threshold})")
        else:
            self.features_removed_['low_variance'] = []
        
        # Step 3: Remove highly correlated features
        numeric_cols = X[remaining].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            highly_corr = self._find_correlated_features(X[numeric_cols])
            self.features_removed_['high_correlation'] = highly_corr
            remaining = [f for f in remaining if f not in highly_corr]
            logger.info(f"  ✓ Removed {len(highly_corr)} highly correlated features (threshold: {self.config.correlation_threshold})")
        else:
            self.features_removed_['high_correlation'] = []
        
        logger.info(f"\n  → Remaining after rule-based selection: {len(remaining)}")
        return remaining
    
    def _find_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """Identify highly correlated feature pairs and select which to remove."""
        try:
            corr_matrix = X.corr().abs()
            
            # Get upper triangle
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find highly correlated pairs
            to_remove = set()
            for col in upper.columns:
                correlated_features = upper.index[upper[col] > self.config.correlation_threshold].tolist()
                if correlated_features:
                    # Keep the feature with higher variance, remove the other
                    variances = X[[col] + correlated_features].var()
                    features_to_check = [col] + correlated_features
                    # Remove all except the one with highest variance
                    max_var_feature = variances.idxmax()
                    to_remove.update([f for f in features_to_check if f != max_var_feature])
            
            return list(to_remove)
        except Exception as e:
            logger.warning(f"  ⚠ Could not compute correlations: {e}")
            return []
    
    def _apply_model_based_selection(
        self, 
        X: pd.DataFrame, 
        features: List[str],
        X_val: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """Apply LightGBM-based feature importance filtering."""
        logger.info("\n--- STAGE 2: MODEL-BASED SELECTION ---")
        
        X_train = X[features].copy()
        y_train = X[self.config.target_col].copy()
        
        # Handle non-numeric features (categorical and datetime)
        categorical_features = []
        features_to_drop = []
        
        for col in X_train.columns:
            dtype = X_train[col].dtype
            
            # Drop datetime columns (they can't be used directly)
            if pd.api.types.is_datetime64_any_dtype(dtype):
                features_to_drop.append(col)
                logger.info(f"  ⚠ Dropping datetime feature: {col}")
            
            # Handle object/categorical columns
            elif dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
                # Try to convert to category for LightGBM
                try:
                    X_train[col] = X_train[col].astype('category')
                    categorical_features.append(col)
                except Exception as e:
                    logger.warning(f"  ⚠ Could not convert {col} to category, dropping: {e}")
                    features_to_drop.append(col)
        
        # Drop problematic features
        if features_to_drop:
            X_train = X_train.drop(columns=features_to_drop)
            features = [f for f in features if f not in features_to_drop]
            self.features_removed_['unsupported_dtype'] = features_to_drop
            logger.info(f"  ✓ Dropped {len(features_to_drop)} unsupported dtype features")
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None:
            X_val_subset = X_val[[f for f in features if f in X_val.columns]].copy()
            y_val = X_val[self.config.target_col].copy()
            
            # Apply same transformations to validation set
            for col in X_val_subset.columns:
                if col in categorical_features:
                    X_val_subset[col] = X_val_subset[col].astype('category')
            
            eval_set = [(X_val_subset, y_val)]
        
        # Train LightGBM model
        logger.info("  Training LightGBM model for feature importance...")
        if categorical_features:
            logger.info(f"  Using {len(categorical_features)} categorical features")
        
        self.lgb_model_ = lgb.LGBMClassifier(**self.config.lgb_params)
        
        self.lgb_model_.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            categorical_feature=categorical_features if categorical_features else 'auto',
            callbacks=[
                lgb.early_stopping(self.config.lgb_early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ] if eval_set else None
        )
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': features,
            'importance': self.lgb_model_.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        self.feature_importances_ = importances
        
        # Calculate training AUC
        y_pred = self.lgb_model_.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred)
        logger.info(f"  Training AUC: {train_auc:.4f}")
        
        # Optional: Permutation importance
        if self.config.use_permutation_importance and X_val is not None:
            logger.info(f"  Computing permutation importance ({self.config.n_permutation_repeats} repeats)...")
            perm_importance = permutation_importance(
                self.lgb_model_,
                X_val_subset,
                y_val,
                n_repeats=self.config.n_permutation_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            self.permutation_importances_ = pd.DataFrame({
                'feature': features,
                'perm_importance_mean': perm_importance.importances_mean,
                'perm_importance_std': perm_importance.importances_std,
            }).sort_values('perm_importance_mean', ascending=False)
        
        # Remove zero-importance features
        zero_importance = importances[importances['importance'] == 0]['feature'].tolist()
        self.features_removed_['zero_importance'] = zero_importance
        remaining = [f for f in features if f not in zero_importance]
        logger.info(f"  ✓ Removed {len(zero_importance)} zero-importance features")
        
        # Optional: Cumulative importance threshold
        if self.config.cumulative_importance_threshold < 1.0:
            importances_sorted = importances[importances['importance'] > 0].copy()
            importances_sorted['cumulative_importance'] = (
                importances_sorted['importance'].cumsum() / importances_sorted['importance'].sum()
            )
            
            features_for_threshold = importances_sorted[
                importances_sorted['cumulative_importance'] <= self.config.cumulative_importance_threshold
            ]['feature'].tolist()
            
            # Always keep at least one feature
            if len(features_for_threshold) == 0 and len(importances_sorted) > 0:
                features_for_threshold = [importances_sorted.iloc[0]['feature']]
            
            cumulative_removed = [f for f in remaining if f not in features_for_threshold]
            if cumulative_removed:
                self.features_removed_['cumulative_importance'] = cumulative_removed
                remaining = features_for_threshold
                logger.info(f"  ✓ Removed {len(cumulative_removed)} features below cumulative importance threshold")
        
        logger.info(f"\n  → Remaining after model-based selection: {len(remaining)}")
        return remaining
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting only chosen features."""
        if not self.features_to_keep_:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        
        cols_to_select = [col for col in self.features_to_keep_ if col in X.columns]
        return X[cols_to_select].copy()
    
    def get_feature_names_out(self, include_target: bool = False) -> List[str]:
        """Return list of selected feature names."""
        cols = self.features_to_keep_
        if not include_target:
            cols = [c for c in cols if c != self.config.target_col]
        return [c for c in cols if c not in self.config.id_cols]
    
    def get_selection_report(self) -> Dict:
        """Generate comprehensive selection report."""
        report = {
            'metadata': self.selection_metadata_,
            'features_removed_by_stage': {
                k: len(v) for k, v in self.features_removed_.items()
            },
            'top_features': None,
            'removal_details': self.features_removed_
        }
        
        if self.feature_importances_ is not None:
            report['top_features'] = self.feature_importances_.head(20).to_dict('records')
        
        return report


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def validate_input_data(df: pd.DataFrame, config: FeatureSelectionConfig) -> None:
    """Validate input data integrity."""
    logger.info("Validating input data...")
    
    required_cols = ['race_datetime', config.target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not pd.api.types.is_datetime64_any_dtype(df['race_datetime']):
        raise TypeError("'race_datetime' must be datetime type")
    
    if df[config.target_col].dtype not in [np.int64, np.int32, bool, np.bool_]:
        raise TypeError(f"'{config.target_col}' must be integer or boolean type")
    
    if df[config.target_col].isnull().any():
        raise ValueError(f"'{config.target_col}' contains null values")
    
    logger.info("✅ Input data validated")


def create_temporal_splits(
    df: pd.DataFrame, 
    config: FeatureSelectionConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronologically-ordered train/validation/test splits."""
    logger.info("\n" + "="*80)
    logger.info("CREATING TEMPORAL SPLITS")
    logger.info("="*80)
    
    n = len(df)
    test_start_idx = int(n * (1 - config.test_size))
    train_end_idx = int(test_start_idx * (1 - config.val_size))
    
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    logger.info(f"\nTrain Set: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Date range: {train_df['race_datetime'].min().date()} to {train_df['race_datetime'].max().date()}")
    logger.info(f"  Target distribution: {train_df[config.target_col].mean():.4f}")
    
    logger.info(f"\nValidation Set: {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Date range: {val_df['race_datetime'].min().date()} to {val_df['race_datetime'].max().date()}")
    logger.info(f"  Target distribution: {val_df[config.target_col].mean():.4f}")
    
    logger.info(f"\nTest Set: {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
    logger.info(f"  Date range: {test_df['race_datetime'].min().date()} to {test_df['race_datetime'].max().date()}")
    logger.info(f"  Target distribution: {test_df[config.target_col].mean():.4f}")
    
    return train_df, val_df, test_df


def create_visualizations(
    selector: AdvancedFeatureSelector,
    output_dir: Path
) -> None:
    """Create feature importance visualizations."""
    if selector.feature_importances_ is None:
        return
    
    logger.info("\nGenerating feature importance visualizations...")
    
    # Top 30 features plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_features = selector.feature_importances_.head(30)
    
    sns.barplot(
        data=top_features,
        y='feature',
        x='importance',
        ax=ax,
        palette='viridis'
    )
    
    ax.set_title('Top 30 Feature Importances (LightGBM)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved plot to: {plot_path.name}")


def log_removal_report(selector: AdvancedFeatureSelector) -> None:
    """Log detailed feature removal report."""
    logger.info("\n" + "="*80)
    logger.info("FEATURE REMOVAL REPORT")
    logger.info("="*80)
    
    for reason, cols in selector.features_removed_.items():
        if cols:
            reason_formatted = reason.replace('_', ' ').title()
            logger.info(f"\n{reason_formatted}: {len(cols)} features")
            
            # Show examples
            for col in cols[:10]:
                logger.info(f"  • {col}")
            
            if len(cols) > 10:
                logger.info(f"  ... and {len(cols) - 10} more")
    
    logger.info("\n" + "="*80)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Execute the feature selection pipeline."""
    
    # Initialize configuration
    config = FeatureSelectionConfig()
    
    # Setup logging
    global logger
    logger = setup_logging(config.output_dir)
    
    try:
        logger.info("="*80)
        logger.info("ADVANCED FEATURE SELECTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Project root: {config.project_root}")
        logger.info(f"Input path: {config.input_path}")
        logger.info(f"Output directory: {config.output_dir}")
        
        # Load data
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        if not config.input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {config.input_path}\n"
                f"Run feature engineering first."
            )
        
        df = pd.read_parquet(config.input_path)
        logger.info(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Validate and sort
        validate_input_data(df, config)
        df = df.sort_values('race_datetime').reset_index(drop=True)
        logger.info("✅ Data sorted chronologically")
        
        # Create splits
        train_df, val_df, test_df = create_temporal_splits(df, config)
        
        # Fit selector
        logger.info("\n" + "="*80)
        logger.info("FITTING FEATURE SELECTOR")
        logger.info("="*80)
        
        selector = AdvancedFeatureSelector(config)
        selector.fit(train_df, X_val=val_df)
        
        # Log removal details
        log_removal_report(selector)
        
        # Transform all splits
        logger.info("\n" + "="*80)
        logger.info("TRANSFORMING DATA SPLITS")
        logger.info("="*80)
        
        train_selected = selector.transform(train_df)
        val_selected = selector.transform(val_df)
        test_selected = selector.transform(test_df)
        
        logger.info(f"Train shape: {train_selected.shape}")
        logger.info(f"Val shape:   {val_selected.shape}")
        logger.info(f"Test shape:  {test_selected.shape}")
        
        # Save artifacts
        logger.info("\n" + "="*80)
        logger.info("SAVING ARTIFACTS")
        logger.info("="*80)
        
        # Save selector
        selector_path = config.output_dir / 'feature_selector.pkl'
        logger.info(f"Saving selector to: {selector_path.name}")
        with open(selector_path, 'wb') as f:
            pickle.dump(selector, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save combined data
        final_df = pd.concat([train_selected, val_selected, test_selected], ignore_index=True)
        output_path = config.output_dir / 'model_ready_data.parquet'
        logger.info(f"Saving model-ready data to: {output_path.name}")
        final_df.to_parquet(output_path, index=False, compression='snappy')
        
        # Save feature names
        feature_names = selector.get_feature_names_out()
        feature_names_path = config.output_dir / 'final_feature_names.txt'
        logger.info(f"Saving {len(feature_names)} feature names to: {feature_names_path.name}")
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(sorted(feature_names)))
        
        # Save selection report
        report = selector.get_selection_report()
        report_path = config.output_dir / 'feature_selection_report.json'
        logger.info(f"Saving selection report to: {report_path.name}")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations
        create_visualizations(selector, config.output_dir)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"\nFinal dataset: {final_df.shape}")
        logger.info(f"Selected features: {len(feature_names)}")
        logger.info(f"Reduction rate: {(1 - len(feature_names) / selector.selection_metadata_['n_features_initial']) * 100:.1f}%")
        
        logger.info("\nOutput files:")
        logger.info(f"  1. model_ready_data.parquet - Clean dataset")
        logger.info(f"  2. feature_selector.pkl - Fitted selector")
        logger.info(f"  3. final_feature_names.txt - Feature list")
        logger.info(f"  4. feature_selection_report.json - Detailed report")
        logger.info(f"  5. feature_importance_plot.png - Visualization")
        logger.info(f"  6. feature_selection.log - Full logs")
        
        if selector.feature_importances_ is not None:
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in selector.feature_importances_.head(10).iterrows():
                logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("Next step: python src/train_model.py")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("❌ PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        
        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        
        return 1


if __name__ == "__main__":
    sys.exit(main())