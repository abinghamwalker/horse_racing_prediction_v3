# =============================================================================

# NOTEBOOK 04: MODEL TRAINING & HYPERPARAMETER TUNING (PRODUCTION VERSION)

# =============================================================================

# PURPOSE:

#   Production-ready training pipeline with:

#   - Proper temporal splits (no data leakage)

#   - Thread-safe metric calculations

#   - Correct probability handling per library

#   - Comprehensive logging and versioning

#   - Environment-based configuration

# =============================================================================



import logging

import pickle

import sys

import os

from pathlib import Path

from datetime import datetime

from typing import Dict, Any, List, Tuple, Callable, Optional

from dataclasses import dataclass



import numpy as np

import pandas as pd

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

import optuna

import mlflow

from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from mlflow.models.signature import infer_signature



# --- Logging Setup ---

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    datefmt='%Y-%m-%d %H:%M:%S'

)

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

pd.set_option('display.max_columns', 100)



logger.info("✅ Setup and imports complete.")





# =============================================================================

# CELL 2: CONFIGURATION (Environment-based, production-ready)

# =============================================================================



@dataclass

class TrainingConfig:

    """Centralized configuration for training pipeline."""

    

    # Model Selection

    train_separate_models: bool = True

    race_category: Optional[str] = 'Flat'  # 'Flat', 'Jumps', or None

    models_to_train: List[str] = None  # ['lgbm', 'xgb', 'catboost']

    

    # Hyperparameter Search

    n_optuna_trials: int = 1  # ✅ Actually do hyperparameter search

    random_state: int = 42

    

    # Temporal Split Ratios (chronological)

    train_ratio: float = 0.70

    val_ratio: float = 0.15

    test_ratio: float = 0.15

    

    # Custom Metric Parameters

    alpha_min: float = 0.5  # Weight for logloss in custom metric

    alpha_max: float = 0.95

    default_alpha: float = 0.7

    

    # Betting Strategy

    edge_thresholds: List[float] = None

    

    # Paths (use environment variables for production)

    project_root: Path = None

    model_ready_dir: Path = None

    model_artifacts_dir: Path = None

    mlflow_tracking_uri: str = None

    

    # Column Definitions

    target_column: str = 'won'

    odds_column: str = 'betfair_sp'

    datetime_column: str = 'race_datetime'

    id_cols: List[str] = None

    

    def __post_init__(self):

        """Initialize derived attributes."""

        if self.models_to_train is None:

            self.models_to_train = ['lgbm', 'xgb', 'catboost']

        

        if self.edge_thresholds is None:

            self.edge_thresholds = [0.0, 0.02, 0.05, 0.10]

        

        if self.id_cols is None:

            self.id_cols = ['race_id', 'horse', 'race_datetime']

        

        # ✅ Use environment variables for production

        if self.project_root is None:

            self.project_root = Path(os.getenv('PROJECT_ROOT', '../'))

        

        if self.model_ready_dir is None:

            self.model_ready_dir = self.project_root / "data" / "model_ready"

        

        if self.model_artifacts_dir is None:

            self.model_artifacts_dir = self.project_root / "models"

            self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)

        

        if self.mlflow_tracking_uri is None:

            # ✅ Production: use remote tracking server

            self.mlflow_tracking_uri = os.getenv(

                'MLFLOW_TRACKING_URI',

                f"sqlite:///{self.project_root / 'mlflow.db'}"

            )

        

        # Validate ratios

        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \

            "Split ratios must sum to 1.0"

        

        logger.info(f"✅ Configuration initialized: {self.race_category or 'All'} races")





# Initialize configuration

config = TrainingConfig()





# =============================================================================

# CELL 3: DATA LOADING WITH PROPER TEMPORAL SPLITTING (CORRECTED VERSION)

# =============================================================================



class TemporalDataSplitter:

    """Handles temporal data splitting with proper date-based cutoffs."""

    

    def __init__(self, config: TrainingConfig):

        self.config = config

    

    def load_and_split(self, filepath: Path) -> Dict[str, Any]:

        """

        Load data and perform temporal split.

        

        Returns:

            Dict containing all splits and metadata.

        """

        logger.info(f"Loading data from: {filepath}")

        df_full = pd.read_parquet(filepath)

        

        # 1. Handle missing odds to prevent CatBoost 'TypeError'.

        initial_rows = len(df_full)

        df_full.dropna(subset=[self.config.odds_column], inplace=True)

        if len(df_full) < initial_rows:

            logger.warning(f"Dropped {initial_rows - len(df_full):,} rows with missing odds.")



        # --- FIX START: Convert 'time' to a numerical feature ---

        # This resolves MLflow signature inference issues and is better for the model.

        # We convert to string first to robustly handle cases where it might be a category.

        time_dt = pd.to_datetime(df_full['time'].astype(str), format='%H:%M:%S', errors='coerce').dt

        df_full['minutes_from_midnight'] = time_dt.hour * 60 + time_dt.minute

        df_full.drop(columns=['time'], inplace=True) # Drop the original column

        logger.info("Converted 'time' column to 'minutes_from_midnight' numerical feature.")

        # --- FIX END ---

            

        # 2. Convert all remaining object columns to 'category' dtype.

        for col in df_full.select_dtypes(include=['object']).columns:

            df_full[col] = df_full[col].astype('category')

            

        # Ensure datetime column is datetime type for sorting

        df_full[self.config.datetime_column] = pd.to_datetime(

            df_full[self.config.datetime_column]

        )

        

        # Sort by datetime (critical for temporal split)

        df_full = df_full.sort_values(self.config.datetime_column).reset_index(drop=True)

        

        # Filter by race category if specified

        if self.config.race_category:

            logger.info(f"Filtering for race category: {self.config.race_category}")

            df = df_full[df_full['race_category'] == self.config.race_category].copy()

        else:

            df = df_full.copy()

        

        logger.info(f"Total samples after filtering: {len(df):,}")

        

        # PROPER TEMPORAL SPLIT

        train_cutoff = df[self.config.datetime_column].quantile(self.config.train_ratio)

        val_cutoff = df[self.config.datetime_column].quantile(

            self.config.train_ratio + self.config.val_ratio

        )

        

        train_mask = df[self.config.datetime_column] <= train_cutoff

        val_mask = (

            (df[self.config.datetime_column] > train_cutoff) & 

            (df[self.config.datetime_column] <= val_cutoff)

        )

        test_mask = df[self.config.datetime_column] > val_cutoff

        

        train_df = df[train_mask].copy()

        val_df = df[val_mask].copy()

        test_df = df[test_mask].copy()

        

        # 3. Robustly identify features, excluding unsupported dtypes and leakage columns.

        potential_feature_cols = [

            c for c in df.columns 

            if c not in self.config.id_cols + [self.config.target_column, self.config.odds_column]

        ]

        

        cols_to_exclude = ['race_category'] # Prevent data leakage

        for col in potential_feature_cols:

            if pd.api.types.is_datetime64_any_dtype(df[col]):

                cols_to_exclude.append(col)

                logger.warning(f"Excluding datetime column '{col}' from features.")

        

        feature_cols = [c for c in potential_feature_cols if c not in cols_to_exclude]

        

        categorical_features = df[feature_cols].select_dtypes(

            include=['category']

        ).columns.tolist()

        

        # Extract X, y, odds for each split

        X_train = train_df[feature_cols]

        y_train = train_df[self.config.target_column]

        odds_train = train_df[self.config.odds_column]

        

        X_val = val_df[feature_cols]

        y_val = val_df[self.config.target_column]

        odds_val = val_df[self.config.odds_column]

        

        X_test = test_df[feature_cols]

        y_test = test_df[self.config.target_column]

        odds_test = test_df[self.config.odds_column]

        

        # Log split information

        logger.info("\n" + "="*80)

        logger.info("✅ TEMPORAL SPLIT COMPLETE (NO DATA LEAKAGE)")

        logger.info("="*80)

        logger.info(f"Training set:   {len(train_df):,} rows | "

                   f"{train_df[self.config.datetime_column].min()} to "

                   f"{train_df[self.config.datetime_column].max()}")

        logger.info(f"Validation set: {len(val_df):,} rows | "

                   f"{val_df[self.config.datetime_column].min()} to "

                   f"{val_df[self.config.datetime_column].max()}")

        logger.info(f"Test set:       {len(test_df):,} rows | "

                   f"{test_df[self.config.datetime_column].min()} to "

                   f"{test_df[self.config.datetime_column].max()}")

        logger.info(f"Features: {len(feature_cols)} total, "

                   f"{len(categorical_features)} categorical")

        logger.info(f"Win rate: Train={y_train.mean():.3f}, "

                   f"Val={y_val.mean():.3f}, Test={y_test.mean():.3f}")

        

        return {

            'X_train': X_train,

            'y_train': y_train,

            'odds_train': odds_train,

            'X_val': X_val,

            'y_val': y_val,

            'odds_val': odds_val,

            'X_test': X_test,

            'y_test': y_test,

            'odds_test': odds_test,

            'feature_cols': feature_cols,

            'categorical_features': categorical_features,

            'train_cutoff': train_cutoff,

            'val_cutoff': val_cutoff

        }
# =============================================================================

# CELL 4: EVALUATION METRICS (Thread-safe, no global state)

# =============================================================================



class BettingMetrics:

    """Thread-safe betting metrics calculator."""

    

    @staticmethod

    def calculate_roi(

        y_true: np.ndarray,

        y_pred: np.ndarray,

        odds: np.ndarray,

        edge_threshold: float = 0.0

    ) -> Dict[str, float]:

        """

        Calculate betting ROI and related metrics.

        

        Args:

            y_true: Binary outcomes (1 for win, 0 for loss)

            y_pred: Predicted win probabilities

            odds: Decimal odds

            edge_threshold: Minimum edge required to place bet

        

        Returns:

            Dictionary of betting metrics

        """

        if len(y_true) == 0:

            return {

                'roi': 0.0,

                'profit': 0.0,

                'num_bets': 0,

                'win_rate': 0.0,

                'total_staked': 0.0,

                'avg_odds': 0.0

            }

        

        # Calculate edge

        implied_prob = 1.0 / odds

        edge = y_pred - implied_prob

        bets = edge > edge_threshold

        

        if not np.any(bets):

            return {

                'roi': 0.0,

                'profit': 0.0,

                'num_bets': 0,

                'win_rate': 0.0,

                'total_staked': 0.0,

                'avg_odds': 0.0

            }

        

        # Fixed stake betting

        stake = 1.0

        wins = y_true[bets] == 1

        profit = np.where(

            bets,

            np.where(y_true == 1, (odds - 1) * stake, -stake),

            0

        )

        

        total_staked = np.sum(bets) * stake

        total_profit = profit.sum()

        

        return {

            'roi': total_profit / total_staked if total_staked > 0 else 0.0,

            'profit': total_profit,

            'num_bets': int(np.sum(bets)),

            'win_rate': float(y_true[bets].mean()),

            'total_staked': total_staked,

            'avg_odds': float(odds[bets].mean())

        }

    

    @staticmethod

    def calculate_all_metrics(

        y_true: np.ndarray,

        y_pred: np.ndarray,

        odds: np.ndarray,

        edge_thresholds: List[float],

        prefix: str = ""

    ) -> Dict[str, float]:

        """

        Calculate all evaluation metrics.

        

        Args:

            y_true: True labels

            y_pred: Predicted probabilities

            odds: Betting odds

            edge_thresholds: List of edge thresholds to evaluate

            prefix: Prefix for metric names (e.g., 'train_', 'val_', 'test_')

        

        Returns:

            Dictionary of all metrics

        """

        metrics = {

            f'{prefix}logloss': log_loss(y_true, y_pred),

            f'{prefix}brier': brier_score_loss(y_true, y_pred),

            f'{prefix}auc': roc_auc_score(y_true, y_pred),

            f'{prefix}sample_count': len(y_true),

            f'{prefix}win_rate': float(y_true.mean())

        }

        

        # Calculate ROI metrics for each edge threshold

        for edge in edge_thresholds:

            roi_metrics = BettingMetrics.calculate_roi(

                y_true, y_pred, odds, edge_threshold=edge

            )

            edge_pct = int(edge * 100)

            for k, v in roi_metrics.items():

                metrics[f'{prefix}roi_{k}_edge{edge_pct}'] = v

        

        return metrics





# =============================================================================

# CELL 5: CUSTOM PROFIT METRIC (Thread-safe, library-specific)

# =============================================================================



class ProfitMetricCalculator:

    """

    Thread-safe profit metric calculator.

    ✅ No global variables

    ✅ Handles different probability formats per library

    """

    

    def __init__(self, alpha: float, edge_threshold: float = 0.01):

        """

        Initialize profit metric calculator.

        

        Args:

            alpha: Weight for logloss (1-alpha for ROI). Higher = more calibration focus.

            edge_threshold: Minimum edge required to place bet

        """

        self.alpha = alpha

        self.edge_threshold = edge_threshold

    

    def _convert_to_probs(

        self,

        y_pred_raw: np.ndarray,

        library: str

    ) -> np.ndarray:

        """

        Convert model output to probabilities.

        

        Args:

            y_pred_raw: Raw model output

            library: 'lgbm', 'xgb', or 'catboost'

        

        Returns:

            Probability array

        """

        if library == 'lgbm':

            # ✅ LightGBM returns probabilities directly

            return y_pred_raw

        elif library == 'xgb':

            # ✅ XGBoost with custom_metric returns logits

            return 1.0 / (1.0 + np.exp(-y_pred_raw))

        elif library == 'catboost':

            # ✅ CatBoost returns logits

            return 1.0 / (1.0 + np.exp(-y_pred_raw))

        else:

            raise ValueError(f"Unknown library: {library}")

    

    def calculate_lgbm(

        self,

        y_pred_raw: np.ndarray,

        dtrain: lgb.Dataset

    ) -> List[Tuple[str, float, bool]]:

        """LightGBM custom metric."""

        y_true = dtrain.get_label()

        probs = self._convert_to_probs(y_pred_raw, 'lgbm')

        

        # ✅ Get odds from separate storage (not from weights)

        odds = dtrain.odds if hasattr(dtrain, 'odds') else np.full_like(y_true, 10.0)

        

        # Calculate components

        logloss = log_loss(y_true, probs)

        roi = BettingMetrics.calculate_roi(

            y_true, probs, odds, edge_threshold=self.edge_threshold

        )['roi']

        

        # Combined score (minimize)

        score = self.alpha * logloss - (1 - self.alpha) * roi

        

        return [('profit_score', score, False)]  # False = minimize

    

    def calculate_xgb(

        self,

        y_pred_raw: np.ndarray,

        dtrain: xgb.DMatrix

    ) -> Tuple[str, float]:

        """XGBoost custom metric."""

        y_true = dtrain.get_label()

        probs = self._convert_to_probs(y_pred_raw, 'xgb')

        

        # ✅ Get odds from separate storage

        odds = dtrain.odds if hasattr(dtrain, 'odds') else np.full_like(y_true, 10.0)

        

        logloss = log_loss(y_true, probs)

        roi = BettingMetrics.calculate_roi(

            y_true, probs, odds, edge_threshold=self.edge_threshold

        )['roi']

        

        score = self.alpha * logloss - (1 - self.alpha) * roi

        

        return 'profit_score', score

    

    def calculate_catboost(

        self,

        y_pred_raw: np.ndarray,

        y_true: np.ndarray,

        odds: np.ndarray

    ) -> float:

        """CatBoost custom metric."""

        probs = self._convert_to_probs(y_pred_raw, 'catboost')

        

        logloss = log_loss(y_true, probs)

        roi = BettingMetrics.calculate_roi(

            y_true, probs, odds, edge_threshold=self.edge_threshold

        )['roi']

        

        score = self.alpha * logloss - (1 - self.alpha) * roi

        

        return score





class CatBoostProfitMetric:

    """Custom metric class for CatBoost."""

    

    def __init__(self, calculator: ProfitMetricCalculator):

        self.calculator = calculator

    

    def is_max_optimal(self):

        return False  # Minimize

    

    def get_final_error(self, error, weight):

        return error

    

    def evaluate(self, approxes, target, weight):

        y_pred_raw = np.array(approxes[0])

        y_true = np.array(target)

        odds = np.array(weight) if weight is not None else np.full_like(y_true, 10.0)

        

        score = self.calculator.calculate_catboost(y_pred_raw, y_true, odds)

        return score, 0.0





# =============================================================================

# CELL 6: MODEL-SPECIFIC DATA PREPARATION

# =============================================================================



class ModelDataPreparator:

    """Prepares data in library-specific formats."""

    

    @staticmethod

    def prepare_lgbm(

        X_train, y_train, odds_train,

        X_val, y_val, odds_val,

        categorical_features

    ):

        """Prepare LightGBM datasets with consistent categorical encoding."""



        # ---  Ensure both train and val categorical dtypes match ---

        X_train = X_train.copy()

        X_val = X_val.copy()



        for col in categorical_features:

            # Combine to ensure full category set

            combined = pd.concat([X_train[col], X_val[col]], axis=0).astype("category")

            categories = combined.cat.categories



            X_train[col] = pd.Categorical(X_train[col], categories=categories)

            X_val[col] = pd.Categorical(X_val[col], categories=categories)



        # --- Continue as before ---

        train_data = lgb.Dataset(

            X_train,

            y_train,

            categorical_feature=categorical_features,

            free_raw_data=False

        )

        train_data.odds = odds_train.values



        val_data = lgb.Dataset(

            X_val,

            y_val,

            categorical_feature=categorical_features,

            reference=train_data,

            free_raw_data=False

        )

        val_data.odds = odds_val.values



        return train_data, val_data



    

    @staticmethod

    def prepare_xgb(

        X_train, y_train, odds_train,

        X_val, y_val, odds_val,

        categorical_features

    ):

        """Prepare XGBoost DMatrix with consistent categorical encoding."""

        X_train = X_train.copy()

        X_val = X_val.copy()



        for col in categorical_features:

            combined = pd.concat([X_train[col], X_val[col]], axis=0).astype("category")

            categories = combined.cat.categories

            X_train[col] = pd.Categorical(X_train[col], categories=categories)

            X_val[col] = pd.Categorical(X_val[col], categories=categories)





        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

        train_data.odds = odds_train.values



        val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        val_data.odds = odds_val.values



        return train_data, val_data

    

    @staticmethod

    def prepare_catboost(

        X_train, y_train, odds_train,

        X_val, y_val, odds_val,

        categorical_features

    ):

        """Prepare CatBoost Pool."""

        # CatBoost uses odds in weight parameter for the metric

        train_data = cb.Pool(

            X_train, label=y_train,

            cat_features=categorical_features,

            weight=odds_train

        )

        

        val_data = cb.Pool(

            X_val, label=y_val,

            cat_features=categorical_features,

            weight=odds_val

        )

        

        return train_data, val_data





# =============================================================================

# CELL 7: OPTUNA OBJECTIVE FUNCTIONS (Thread-safe)

# =============================================================================



class OptunaObjectives:

    """Optuna objective functions for each model type."""

    

    def __init__(

        self,

        train_data,

        val_data,

        X_val,

        y_val,

        odds_val,

        config: TrainingConfig

    ):

        self.train_data = train_data

        self.val_data = val_data

        self.X_val = X_val

        self.y_val = y_val

        self.odds_val = odds_val

        self.config = config

    

    def lgbm_objective(self, trial: optuna.Trial) -> float:

        """LightGBM hyperparameter optimization."""

        # ✅ Thread-safe: alpha is trial-specific

        alpha = trial.suggest_float('alpha', self.config.alpha_min, self.config.alpha_max)

        

        params = {

            'objective': 'binary',

            'metric': 'logloss',

            'verbosity': -1,

            'seed': self.config.random_state,

            'n_jobs': 1,  # Important for Optuna parallelization

            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),

            'num_leaves': trial.suggest_int('num_leaves', 20, 200),

            'max_depth': trial.suggest_int('max_depth', 3, 15),

            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),

            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),

            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),

            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),

            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        }

        

        # ✅ Create metric calculator for this trial

        metric_calc = ProfitMetricCalculator(alpha=alpha)

        

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "profit_score")

        

        model = lgb.train(

            params,

            self.train_data,

            num_boost_round=1000,

            valid_sets=[self.val_data],

            callbacks=[

                lgb.early_stopping(50, verbose=False),

                pruning_callback

            ],

            feval=metric_calc.calculate_lgbm

        )

        

        trial.set_user_attr("best_iteration", model.best_iteration)

        

        # Also log standard metrics for comparison

        val_preds = model.predict(self.X_val)

        trial.set_user_attr("val_logloss", log_loss(self.y_val, val_preds))

        trial.set_user_attr("val_auc", roc_auc_score(self.y_val, val_preds))

        

        return model.best_score['valid_0']['profit_score']

    

    def xgb_objective(self, trial: optuna.Trial) -> float:

        """XGBoost hyperparameter optimization."""

        alpha = trial.suggest_float('alpha', self.config.alpha_min, self.config.alpha_max)

        

        params = {

            'objective': 'binary:logistic',

            'eval_metric': 'logloss',

            'verbosity': 0,

            'seed': self.config.random_state,

            'nthread': 1,

            'tree_method': 'hist',

            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),

            'alpha': trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),

            'subsample': trial.suggest_float('subsample', 0.5, 1.0),

            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

            'max_depth': trial.suggest_int('max_depth', 3, 10),

            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

            'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),

        }

        

        metric_calc = ProfitMetricCalculator(alpha=alpha)

        

        pruning_callback = optuna.integration.XGBoostPruningCallback(

            trial, "validation-profit_score"

        )

        

        model = xgb.train(

            params,

            self.train_data,

            num_boost_round=1000,

            evals=[(self.val_data, 'validation')],

            verbose_eval=False,

            early_stopping_rounds=50,

            custom_metric=metric_calc.calculate_xgb,

            callbacks=[pruning_callback]

        )

        

        trial.set_user_attr("best_iteration", model.best_iteration)

        

        val_preds = model.predict(self.val_data)

        trial.set_user_attr("val_logloss", log_loss(self.y_val, val_preds))

        trial.set_user_attr("val_auc", roc_auc_score(self.y_val, val_preds))

        

        return model.best_score

    

    def catboost_objective(self, trial: optuna.Trial) -> float:

        """CatBoost hyperparameter optimization."""

        alpha = trial.suggest_float('alpha', self.config.alpha_min, self.config.alpha_max)

        

        metric_calc = ProfitMetricCalculator(alpha=alpha)

        custom_metric = CatBoostProfitMetric(metric_calc)

        

        params = {

            "objective": "Logloss",

            "eval_metric": custom_metric,

            "iterations": 1000,

            "verbose": 0,

            "random_seed": self.config.random_state,

            "thread_count": 1,

            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

            "depth": trial.suggest_int("depth", 4, 10),

            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),

            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),

            "subsample": trial.suggest_float("subsample", 0.5, 1.0),

            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),

        }

        

        model = cb.CatBoostClassifier(**params)

        

        pruning_callback = optuna.integration.CatBoostPruningCallback(

            trial, "CatBoostProfitMetric"

        )

        

        model.fit(

            self.train_data,

            eval_set=self.val_data,

            early_stopping_rounds=50,

            callbacks=[pruning_callback]

        )

        

        trial.set_user_attr("best_iteration", model.get_best_iteration())

        

        val_preds = model.predict_proba(self.X_val)[:, 1]

        trial.set_user_attr("val_logloss", log_loss(self.y_val, val_preds))

        trial.set_user_attr("val_auc", roc_auc_score(self.y_val, val_preds))

        

        return model.get_best_score()['validation']['CatBoostProfitMetric']





# =============================================================================

# CELL 8: MODEL TRAINING ORCHESTRATOR (COMPLETE AND CORRECTED)

# =============================================================================



class ModelTrainer:

    """Orchestrates model training, evaluation, and logging."""

    

    def __init__(self, config: TrainingConfig, data: Dict[str, Any]):

        self.config = config

        self.data = data

        self.metrics_calculator = BettingMetrics()

        

        # Setup MLflow

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        experiment_name = f"{config.race_category}_WinPrediction" if config.race_category else "AllRaces_WinPrediction"

        mlflow.set_experiment(experiment_name)



    def _save_local_model_backup(self, model_type: str, model, run_id: str):

        """Saves a local pickle backup of the model."""

        try:

            filename = f"{model_type}_{self.config.race_category or 'All'}_{run_id}.pkl"

            filepath = self.config.model_artifacts_dir / filename

            with open(filepath, "wb") as f:

                pickle.dump(model, f)

            logger.info(f"✅ Successfully saved local model backup to: {filepath}")

            mlflow.log_artifact(str(filepath), "local_backup")

        except Exception as e:

            logger.error(f"Could not save local model backup: {e}")



    def train_model(self, model_type: str) -> Dict[str, Any]:

        """Trains a single model type."""

        logger.info(f"\n{'='*80}")

        logger.info(f"🚀 TRAINING {model_type.upper()} MODEL")

        logger.info(f"{'='*80}")

        

        with mlflow.start_run(run_name=f"{model_type}_{datetime.now():%Y%m%d_%H%M%S}") as run:

            run_id = run.info.run_id

            

            mlflow.log_params({

                'model_type': model_type, 'race_category': self.config.race_category or 'All',

                'n_trials': self.config.n_optuna_trials, 'random_state': self.config.random_state,

                'train_samples': len(self.data['X_train']), 'val_samples': len(self.data['X_val']),

                'test_samples': len(self.data['X_test']),

            })

            

            logger.info("Preparing model-specific data formats...")

            preparator = ModelDataPreparator()

            

            if model_type == 'lgbm':

                train_data, val_data = preparator.prepare_lgbm(

                    self.data['X_train'], self.data['y_train'], self.data['odds_train'],

                    self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                    self.data['categorical_features']

                )

                objective_func = 'lgbm_objective'

            elif model_type == 'xgb':

                train_data, val_data = preparator.prepare_xgb(

                    self.data['X_train'], self.data['y_train'], self.data['odds_train'],

                    self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                    self.data['categorical_features']

                )

                objective_func = 'xgb_objective'

            elif model_type == 'catboost':

                train_data, val_data = preparator.prepare_catboost(

                    self.data['X_train'], self.data['y_train'], self.data['odds_train'],

                    self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                    self.data['categorical_features']

                )

                objective_func = 'catboost_objective'

            else:

                raise ValueError(f"Unknown model type: {model_type}")

            

            logger.info(f"Running Optuna optimization ({self.config.n_optuna_trials} trials)...")

            objectives = OptunaObjectives(

                train_data, val_data, self.data['X_val'], self.data['y_val'], self.data['odds_val'], self.config

            )

            study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

            study.optimize(getattr(objectives, objective_func), n_trials=self.config.n_optuna_trials, show_progress_bar=True, n_jobs=1)

            

            best_params = study.best_params

            best_iteration = study.best_trial.user_attrs['best_iteration']

            

            logger.info(f"Best trial value: {study.best_value:.6f}")

            logger.info(f"Best iteration: {best_iteration}")

            

            mlflow.log_params(best_params)

            mlflow.log_metric("best_optuna_score", study.best_value)

            mlflow.log_metric("best_iteration", best_iteration)

            

            logger.info("Training final model on train+val data...")

            final_model, X_train_full = self._train_final_model(

                model_type, best_params, best_iteration

            )

            

            self._save_local_model_backup(model_type, final_model, run_id)

            

            logger.info("Evaluating model on all splits...")

            all_metrics = self._evaluate_all_splits(model_type, final_model)

            mlflow.log_metrics(all_metrics)

            

            self._log_feature_importance(model_type, final_model)

            self._log_model(model_type, final_model, X_train_full)

            

            logger.info(f"\n{'='*80}")

            logger.info(f"✅ {model_type.upper()} TRAINING COMPLETE")

            logger.info(f"MLflow Run ID: {run_id}")

            self._print_metrics_summary(all_metrics)

            

            return {

                'run_id': run_id, 'model': final_model, 'metrics': all_metrics, 'best_params': best_params

            }

    

    def _train_final_model(self, model_type: str, best_params: Dict, best_iteration: int):

        """Train final model on full training data and return model and data."""

        X_train_full = pd.concat([self.data['X_train'], self.data['X_val']])

        y_train_full = pd.concat([self.data['y_train'], self.data['y_val']])

        odds_train_full = pd.concat([self.data['odds_train'], self.data['odds_val']])

        

        preparator = ModelDataPreparator()

        

        if model_type == 'lgbm':

            full_data, _ = preparator.prepare_lgbm(

                X_train_full, y_train_full, odds_train_full,

                self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                self.data['categorical_features']

            )

            params = {**best_params, 'objective': 'binary', 'metric': 'logloss', 'seed': self.config.random_state, 'verbosity': -1}

            params.pop('alpha', None)

            model = lgb.train(params, full_data, num_boost_round=best_iteration)

        

        elif model_type == 'xgb':

            full_data, _ = preparator.prepare_xgb(

                X_train_full, y_train_full, odds_train_full,

                self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                self.data['categorical_features']

            )

            params = {**best_params, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': self.config.random_state, 'verbosity': 0}

            params.pop('alpha', None)

            if 'xgb_alpha' in params: params['alpha'] = params.pop('xgb_alpha')

            model = xgb.train(params, full_data, num_boost_round=best_iteration)

        

        elif model_type == 'catboost':

            full_data, _ = preparator.prepare_catboost(

                X_train_full, y_train_full, odds_train_full,

                self.data['X_val'], self.data['y_val'], self.data['odds_val'],

                self.data['categorical_features']

            )

            params = {**best_params, 'objective': 'Logloss', 'iterations': best_iteration, 'random_seed': self.config.random_state, 'verbose': 0}

            params.pop('alpha', None)

            model = cb.CatBoostClassifier(**params)

            model.fit(full_data)

        

        return model, X_train_full

    

    def _evaluate_all_splits(self, model_type: str, model) -> Dict[str, float]:

        """Evaluate model on train, val, and test sets."""

        all_metrics = {}

        for split_name, X, y, odds in [

            ('train', self.data['X_train'], self.data['y_train'], self.data['odds_train']),

            ('val', self.data['X_val'], self.data['y_val'], self.data['odds_val']),

            ('test', self.data['X_test'], self.data['y_test'], self.data['odds_test'])

        ]:

            if model_type == 'lgbm':

                preds = model.predict(X)

            elif model_type == 'xgb':

                X_cat = X.copy()

                for col in self.data['categorical_features']:

                    X_cat[col] = X_cat[col].astype('category')

                dmatrix = xgb.DMatrix(X_cat, enable_categorical=True)

                preds = model.predict(dmatrix)

            elif model_type == 'catboost':

                preds = model.predict_proba(X)[:, 1]

            

            split_metrics = self.metrics_calculator.calculate_all_metrics(

                y.values, preds, odds.values, self.config.edge_thresholds, prefix=f'{split_name}_'

            )

            all_metrics.update(split_metrics)

        return all_metrics

    

    def _log_feature_importance(self, model_type: str, model):

        """Log feature importance to MLflow."""

        try:

            if model_type == 'lgbm':

                importance = model.feature_importance(importance_type='gain')

                feature_names = model.feature_name()

            elif model_type == 'xgb':

                importance = list(model.get_score(importance_type='gain').values())

                feature_names = list(model.get_score(importance_type='gain').keys())

            elif model_type == 'catboost':

                importance = model.get_feature_importance()

                feature_names = model.feature_names_

            

            importance_df = pd.DataFrame({

                'feature': feature_names, 'importance': importance

            }).sort_values('importance', ascending=False)

            

            importance_path = "feature_importance.csv"

            importance_df.to_csv(importance_path, index=False)

            mlflow.log_artifact(importance_path)

            

            # FIX for NameError: Use importance_df directly.

            top_5_features = importance_df.head(5)['feature'].tolist()

            logger.info(f"✅ Logged feature importance artifact (top 5): {', '.join(top_5_features)}")

            

        except Exception as e:

            logger.warning(f"Could not log feature importance: {e}")

    

    

    def _log_model(self, model_type: str, model, X_train_full: pd.DataFrame):

        """Log model safely to MLflow with dtype and framework-specific handling."""

        import mlflow

        from mlflow.models.signature import infer_signature

        import pandas as pd

        import numpy as np

        import logging



        logger = logging.getLogger(__name__)



        # 🧹 Sample input (small batch for signature)

        sample_input = X_train_full.sample(min(5, len(X_train_full))).copy()



        # 🔧 Convert object → string or category depending on model

        if model_type in ["xgb", "xgboost"]:

            # For XGBoost: convert object to category

            for col in sample_input.select_dtypes(include="object").columns:

                sample_input[col] = sample_input[col].astype("category")



        elif model_type in ["lgb", "lightgbm"]:

            # For LightGBM: convert object to string (keeps consistency)

            for col in sample_input.select_dtypes(include="object").columns:

                sample_input[col] = sample_input[col].astype("string")



        elif model_type in ["cb", "catboost"]:

            # CatBoost handles strings fine, but ensure no NaN in categorical columns

            sample_input.fillna("missing", inplace=True)



        # Try to infer signature safely

        try:

            y_pred_sample = model.predict(sample_input)

            signature = infer_signature(sample_input, y_pred_sample)

        except Exception as e:

            logger.warning(f"⚠️ Could not infer signature automatically: {e}")

            signature = None



        # 🚀 Log model safely

        with mlflow.start_run(run_name=f"{model_type}_training", nested=True):

            if model_type in ["lgb", "lightgbm"]:

                mlflow.lightgbm.log_model(model, "model", signature=signature)

            elif model_type in ["xgb", "xgboost"]:

                mlflow.xgboost.log_model(model, "model", signature=signature)

            elif model_type in ["cb", "catboost"]:

                mlflow.catboost.log_model(model, "model", signature=signature)

            else:

                mlflow.sklearn.log_model(model, "model", signature=signature)



        logger.info(f"✅ Logged {model_type.upper()} model to MLflow successfully.")



    

    def _print_metrics_summary(self, metrics: Dict[str, float]):

        """Print formatted metrics summary."""

        logger.info("\n📊 METRICS SUMMARY:")

        for split in ['train', 'val', 'test']:

            logger.info(f"\n{split.upper()} SET:")

            logger.info(f"  LogLoss: {metrics.get(f'{split}_logloss', 0):.4f}")

            logger.info(f"  AUC:     {metrics.get(f'{split}_auc', 0):.4f}")

            logger.info(f"  Brier:   {metrics.get(f'{split}_brier', 0):.4f}")

            

            roi_key = f'{split}_roi_roi_edge2'

            num_bets_key = f'{split}_roi_num_bets_edge2'

            if roi_key in metrics:

                logger.info(f"  ROI (2% edge): {metrics[roi_key]:.2%} "

                          f"({metrics[num_bets_key]:.0f} bets)")
# =============================================================================

# CELL 9: SETUP FOR TRAINING

# =============================================================================



logger.info("\n" + "="*80)

logger.info("🚀 INITIALIZING TRAINING SETUP")

logger.info("="*80)



# 1. Load and split data

logger.info("Loading and splitting data...")

splitter = TemporalDataSplitter(config)

data = splitter.load_and_split(config.model_ready_dir / "04_model_ready_data.parquet")



# 2. Initialize the ModelTrainer

logger.info("Initializing ModelTrainer...")

trainer = ModelTrainer(config, data)



logger.info("\n✅ Setup complete. Ready to train models in the cells below.")
# =============================================================================

# CELL 10: TRAIN LIGHTGBM

# =============================================================================

lgbm_results = None

try:

    lgbm_results = trainer.train_model('lgbm')

except Exception as e:

    logger.error(f"❌ Failed to train LightGBM: {e}", exc_info=True)
# =============================================================================

# CELL 11: TRAIN XGBOOST

# =============================================================================

xgb_results = None

try:

    xgb_results = trainer.train_model('xgb')

except Exception as e:

    logger.error(f"❌ Failed to train XGBoost: {e}", exc_info=True)
# =============================================================================

# CELL 12: TRAIN CATBOOST

# =============================================================================

catboost_results = None

try:

    catboost_results = trainer.train_model('catboost')

except Exception as e:

    logger.error(f"❌ Failed to train CatBoost: {e}", exc_info=True)
