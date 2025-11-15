#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Horse Racing Prediction

Usage:
    python feature_engineering.py

This script:
1. Loads processed data.
2. Creates temporally-safe chronological features by importing the v2 module.
3. Creates derived features (interactions, ratios).
4. Creates advanced market features.
5. Saves feature-engineered data and state trackers for incremental updates.

Output:
    - data/02_reduction_ready/02_features_engineered.parquet
    - data/02_reduction_ready/state_trackers.pkl
    - data/02_reduction_ready/feature_names.txt
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import sys
from pathlib import Path
import warnings
import pickle
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd

# Import the feature engineering module and its configuration classes
from chronological_features_v2 import create_chronological_features, EloConfig, HistoryConfig, FeatureConfig

# Suppress warnings & configure pandas
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(".").resolve()

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / '01_processed'
REDUCTION_READY_DIR = PROJECT_ROOT / 'data' / '02_reduction_ready'

OUTPUT_DATA_FILE = REDUCTION_READY_DIR / "02_features_engineered.parquet"
OUTPUT_TRACKERS_FILE = REDUCTION_READY_DIR / "state_trackers.pkl"
FEATURE_NAMES_FILE = REDUCTION_READY_DIR / "feature_names.txt"

FILE_PATTERN = "processed_race_data_*.parquet"

REDUCTION_READY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS for Derived Features
# (ELO and History constants are now managed by the imported module's configs)
# =============================================================================

RECENT_DAYS_WINDOW = 14
LONG_DRY_SPELL_THRESHOLD = 10
DISTANCE_SIMILARITY_THRESHOLD_M = 200
SPECIALIST_RUNS_THRESHOLD = 3
FRESH_HORSE_DAYS_THRESHOLD = 60
QUICK_RETURN_DAYS_THRESHOLD = 7
PRIME_AGE_RANGE = (3, 5)
WEAK_FAVORITE_BSP_THRESHOLD = 4.0

# =============================================================================
# HELPER & FEATURE FUNCTIONS
# =============================================================================

def find_latest_file(directory: Path, pattern: str) -> Path:
    """Finds the most recently created file matching a pattern in a directory."""
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(
            f"No processed data files found in {directory} matching pattern '{pattern}'."
        )
    return sorted(matching_files)[-1]

INPUT_FILE = find_latest_file(PROCESSED_DATA_DIR, FILE_PATTERN)

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived and interaction features."""
    logger.info("\n" + "="*80)
    logger.info("DERIVED FEATURE ENGINEERING")
    logger.info("="*80)
    
    df_final = df.copy()

    # Market features
    if 'betfair_sp' in df_final.columns:
        df_final['bsp_rank'] = df_final.groupby('race_id')['betfair_sp'].rank(method='min')
        df_final['implied_prob_bsp'] = 1 / df_final['betfair_sp'].replace(0, np.nan)
        df_final['market_book_percentage'] = df_final.groupby('race_id')['implied_prob_bsp'].transform('sum')
        df_final['log_betfair_sp'] = np.log1p(df_final['betfair_sp'])
        df_final['is_favorite'] = (df_final['bsp_rank'] == 1).astype(int)
        df_final['is_second_favorite'] = (df_final['bsp_rank'] == 2).astype(int)
        df_final['is_outsider'] = (df_final['bsp_rank'] > df_final.get('runners', 10) * 0.75).astype(int)
        df_final['weak_favorite'] = ((df_final['is_favorite'] == 1) & (df_final['betfair_sp'] > WEAK_FAVORITE_BSP_THRESHOLD)).astype(int)
        if 'elo_implied_prob' in df_final.columns:
            df_final['value_signal_elo_vs_bsp'] = df_final['elo_implied_prob'] - df_final['implied_prob_bsp']

    # Form features
    df_final['place_to_win_ratio'] = np.where(df_final['win_rate_last_5'] > 0, df_final['place_rate_last_5'] / df_final['win_rate_last_5'], np.nan)
    df_final['on_winning_streak'] = (df_final['win_streak'] > 0).astype(int)
    df_final['long_dry_spell'] = (df_final['races_since_win'] > LONG_DRY_SPELL_THRESHOLD).astype(int)

    # Specialization
    df_final['course_specialist'] = (df_final['course_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    df_final['distance_specialist'] = (df_final['distance_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    df_final['surface_specialist'] = (df_final['surface_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    df_final['double_specialist'] = ((df_final['course_specialist'] == 1) & (df_final['distance_specialist'] == 1)).astype(int)

    # Field context
    df_final['or_vs_field_quality'] = df_final['official_rating'] - df_final['field_quality_avg_or']
    df_final['above_average_runner'] = (df_final['or_vs_field_quality'] > 0).astype(int)

    # Jockey/Trainer strike rates
    df_final['jockey_strike_rate_14d'] = np.where(df_final['jockey_rides_14d'] > 0, df_final['jockey_form_wins_14d'] / df_final['jockey_rides_14d'], 0.0)
    df_final['trainer_strike_rate_14d'] = np.where(df_final['trainer_runners_14d'] > 0, df_final['trainer_form_wins_14d'] / df_final['trainer_runners_14d'], 0.0)

    # Freshness & Age
    if 'days_since_last_time_out' in df_final.columns:
        df_final['fresh_horse'] = (df_final['days_since_last_time_out'] >= FRESH_HORSE_DAYS_THRESHOLD).astype(int)
        df_final['quick_return'] = (df_final['days_since_last_time_out'] <= QUICK_RETURN_DAYS_THRESHOLD).astype(int)
        df_final['log_days_since_last'] = np.log1p(df_final['days_since_last_time_out'])
    if 'age' in df_final.columns:
        df_final['prime_age'] = df_final['age'].between(PRIME_AGE_RANGE[0], PRIME_AGE_RANGE[1]).astype(int)
        df_final['age_squared'] = df_final['age'] ** 2

    # Temporal
    df_final['month'] = df_final['date_of_race'].dt.month
    df_final['day_of_week'] = df_final['date_of_race'].dt.dayofweek
    df_final['is_weekend'] = df_final['day_of_week'].isin([5, 6]).astype(int)

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info("✅ Derived features complete")
    return df_final


def create_advanced_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates advanced market-based features."""
    logger.info("\n" + "="*80)
    logger.info("ADVANCED MARKET FEATURES")
    logger.info("="*80)
    
    df_adv = df.copy()
    
    if 'betfair_sp' in df_adv.columns and 'forecasted_odds' in df_adv.columns:
        df_adv['odds_drift_forecast_to_bsp'] = df_adv['betfair_sp'] - df_adv['forecasted_odds']
        df_adv['odds_drift_pct'] = (df_adv['betfair_sp'] - df_adv['forecasted_odds']) / df_adv['forecasted_odds'].replace(0, np.nan)
        df_adv['odds_shortened'] = (df_adv['betfair_sp'] < df_adv['forecasted_odds']).fillna(False).astype(int)
        df_adv['odds_drifted'] = (df_adv['betfair_sp'] > df_adv['forecasted_odds']).fillna(False).astype(int)
    
    if 'implied_prob_bsp' in df_adv.columns:
        df_adv['fav_prob_concentration'] = df_adv.groupby('race_id')['implied_prob_bsp'].transform(
            lambda x: x.max() / x.sum() if x.sum() > 0 else 0
        )
    
    if 'betfair_sp' in df_adv.columns:
        fav_odds_per_race = df_adv.groupby('race_id')['betfair_sp'].transform('min')
        df_adv['odds_gap_to_favorite'] = df_adv['betfair_sp'] - fav_odds_per_race
    
    df_adv.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info("✅ Advanced market features complete")
    return df_adv


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates data integrity and removes leaky columns."""
    logger.info("\n" + "="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)
    
    df_validated = df.copy()
    
    critical_cols = ['race_id', 'date_of_race', 'horse', 'won']
    if any(df_validated[col].isnull().any() for col in critical_cols if col in df_validated.columns):
        raise ValueError(f"Critical columns have NaNs.")
    
    numeric_cols = df_validated.select_dtypes(include=np.number)
    if np.isinf(numeric_cols).any().any():
        raise ValueError(f"Infinite values found.")
    
    leakage_keywords = ['winning_distance', 'sp_win_return', 'e_w_return', 'betfair_win_return',
                        'place_return', 'betfair_lay_return', 'ip_min', 'ip_max']
    found_leakage = [col for col in df_validated.columns if any(key in col.lower() for key in leakage_keywords)]
    
    if found_leakage:
        logger.warning(f"Removing {len(found_leakage)} potentially leaky columns: {found_leakage}")
        df_validated.drop(columns=found_leakage, inplace=True)
    
    logger.info("✅ Validation passed")
    return df_validated


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline execution."""
    try:
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE (using v2 module)")
        logger.info("="*80)
        
        logger.info(f"\nLoading data from: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        logger.info(f"Loaded {len(df):,} records")

        logger.info("\nPreparing data for feature engineering module...")

        # 1. Handle Weight Conversion (weight -> weight_lbs)
        if 'weight_lbs' not in df.columns and 'weight' in df.columns:
            logger.info("Found 'weight' column. Converting to 'weight_lbs'...")
            def weight_to_lbs(w):
                if pd.isna(w): return np.nan
                try:
                    if isinstance(w, (int, float)): return float(w)
                    if '-' in str(w):
                        s, p = str(w).split('-')
                        return int(s) * 14 + int(p)
                    return float(w)
                except (ValueError, TypeError): return np.nan
            df['weight_lbs'] = df['weight'].apply(weight_to_lbs)
        elif 'weight_lbs' not in df.columns:
            logger.warning("Neither 'weight' nor 'weight_lbs' found. Creating default column.")
            df['weight_lbs'] = np.nan

        # 2. Handle Distance Conversion (distance_f -> distance_m)
        FURLONGS_TO_METERS = 201.168
        if 'distance_m' not in df.columns and 'distance_f' in df.columns:
            logger.info("Found 'distance_f' column. Converting to 'distance_m'...")
            df['distance_m'] = df['distance_f'] * FURLONGS_TO_METERS
        elif 'distance_m' not in df.columns:
            logger.warning("Neither 'distance_f' nor 'distance_m' found. Creating default column.")
            df['distance_m'] = np.nan
            
        # 3. Handle Course Name (track -> course)
        if 'course' not in df.columns and 'track' in df.columns:
            logger.info("Found 'track' column. Renaming to 'course'...")
            df = df.rename(columns={'track': 'course'})
        elif 'course' not in df.columns:
            logger.warning("Neither 'track' nor 'course' found. Creating default column.")
            df['course'] = 'Unknown'

        logger.info("✅ Data preparation complete.")
        
        # Load the previous state for incremental updates, if it exists.
        existing_state = None
        if OUTPUT_TRACKERS_FILE.exists():
            try:
                with open(OUTPUT_TRACKERS_FILE, 'rb') as f:
                    existing_state = pickle.load(f)
                logger.info(f"Successfully loaded existing state from {OUTPUT_TRACKERS_FILE}")
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Could not load state file, starting fresh. Error: {e}")

        # Instantiate configuration objects for the feature engineering module
        elo_config = EloConfig() # Uses default values from the module
        history_config = HistoryConfig(recent_form_days=RECENT_DAYS_WINDOW)
        feature_config = FeatureConfig(distance_similarity_threshold_m=DISTANCE_SIMILARITY_THRESHOLD_M)
        
        # Call the imported function to create chronological features
        df_chrono, new_state_trackers = create_chronological_features(
            df,
            state_trackers=existing_state,
            elo_config=elo_config,
            history_config=history_config,
            feature_config=feature_config
        )
        
        # Continue with the rest of the feature creation
        df_derived = create_derived_features(df_chrono)
        df_advanced = create_advanced_market_features(df_derived)
        df_final = validate_data(df_advanced)
        
        # Save the final dataframe
        logger.info(f"\nSaving feature-engineered data to: {OUTPUT_DATA_FILE}")
        df_final.to_parquet(OUTPUT_DATA_FILE, index=False)
        
        # Save the new state for the next run
        logger.info(f"Saving new state trackers to: {OUTPUT_TRACKERS_FILE}")
        with open(OUTPUT_TRACKERS_FILE, 'wb') as f:
            pickle.dump(new_state_trackers, f)
        
        # Save feature names
        feature_cols = [col for col in df_final.columns if col not in ['won', 'date_of_race', 'race_id', 'horse']]
        with open(FEATURE_NAMES_FILE, 'w') as f:
            f.write('\n'.join(feature_cols))
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Final shape: {df_final.shape}")
        logger.info(f"Features created: {len(feature_cols)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*80}\n❌ PIPELINE FAILED\n{'='*80}")
        logger.exception("An error occurred:") # This will print the full traceback
        return 1

if __name__ == "__main__":
    sys.exit(main())