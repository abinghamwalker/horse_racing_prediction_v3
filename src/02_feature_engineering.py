#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Horse Racing Prediction

Usage:
    python feature_engineering.py

This script:
1. Loads processed data.
2. Creates temporally-safe chronological features module.
3. Creates derived features (interactions, ratios).
4. Creates advanced market features.
5. Saves feature-engineered data and state trackers for incremental updates.

Output:
    - data/02_reduction_ready/02_features_engineered.parquet
    - data/02_reduction_ready/state_trackers.pkl
    - data/02_reduction_ready/feature_names.txt

    this file may need a weight and distance fix because it is missing gemini can do it
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
import sys
from pathlib import Path
import warnings
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from chronological_features import create_chronological_features, EloConfig, HistoryConfig, FeatureConfig

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

CHECKPOINT_DF_FILE = REDUCTION_READY_DIR / "02_chrono_features_checkpoint.parquet" 
CHECKPOINT_TRACKERS_FILE = REDUCTION_READY_DIR / "02_chrono_trackers_checkpoint.pkl" 

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
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction and polynomial features."""
    logger.info("\n" + "="*80)
    logger.info("INTERACTION FEATURES")
    logger.info("="*80)
    
    df_int = df.copy()
    
    # ELO interactions
    if 'horse_elo_pre_race' in df_int.columns:
        df_int['elo_squared'] = df_int['horse_elo_pre_race'] ** 2
        
        if 'course_experience' in df_int.columns:
            df_int['elo_times_course_exp'] = df_int['horse_elo_pre_race'] * df_int['course_experience']
        
        if 'distance_experience' in df_int.columns:
            df_int['elo_times_distance_exp'] = df_int['horse_elo_pre_race'] * df_int['distance_experience']
        
        if 'days_since_last_time_out' in df_int.columns:
            df_int['elo_per_day_since_last'] = df_int['horse_elo_pre_race'] / (df_int['days_since_last_time_out'] + 1)
    
    # Field quality interactions
    if 'field_quality_avg_or' in df_int.columns:
        df_int['field_quality_squared'] = df_int['field_quality_avg_or'] ** 2
        
        if 'official_rating' in df_int.columns:
            df_int['or_times_field_quality'] = df_int['official_rating'] * df_int['field_quality_avg_or']
    
    # Rating interactions
    if 'official_rating' in df_int.columns:
        df_int['or_squared'] = df_int['official_rating'] ** 2
        
        if 'surface_win_rate' in df_int.columns:
            df_int['or_times_surface_wr'] = df_int['official_rating'] * df_int['surface_win_rate']
        
        if 'weight_lbs' in df_int.columns:
            df_int['or_per_weight'] = df_int['official_rating'] / (df_int['weight_lbs'] + 1)
    
    # Form interactions
    if 'win_rate_last_5' in df_int.columns and 'total_career_runs' in df_int.columns:
        df_int['wr_times_experience'] = df_int['win_rate_last_5'] * df_int['total_career_runs']
    
    # Age interactions
    if 'age' in df_int.columns:
        if 'total_career_runs' in df_int.columns:
            df_int['age_times_career_runs'] = df_int['age'] * df_int['total_career_runs']
        
        if 'official_rating' in df_int.columns:
            df_int['age_times_rating'] = df_int['age'] * df_int['official_rating']
    
    # Specialization interactions
    if all(col in df_int.columns for col in ['course_experience', 'distance_experience']):
        df_int['course_times_distance_exp'] = df_int['course_experience'] * df_int['distance_experience']
    
    # Jockey/Trainer interaction
    if all(col in df_int.columns for col in ['jockey_strike_rate_14d', 'trainer_strike_rate_14d']):
        df_int['jockey_trainer_combined_form'] = (df_int['jockey_strike_rate_14d'] + 
                                                   df_int['trainer_strike_rate_14d']) / 2
        df_int['jockey_times_trainer_form'] = df_int['jockey_strike_rate_14d'] * df_int['trainer_strike_rate_14d']
    
    # Distance change interactions
    if 'actual_distance_change_m' in df_int.columns:
        df_int['distance_change_squared'] = df_int['actual_distance_change_m'] ** 2
        df_int['distance_change_abs'] = np.abs(df_int['actual_distance_change_m'])
    
    logger.info("✅ Interaction features complete")
    return df_int


def create_elo_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived features from ELO ratings."""
    logger.info("\n" + "="*80)
    logger.info("ELO DERIVED FEATURES")
    logger.info("="*80)
    
    df_elo = df.copy()
    
    if 'horse_elo_pre_race' in df_elo.columns:
        # ELO field comparisons
        df_elo['elo_vs_field_avg'] = df_elo['horse_elo_pre_race'] - df_elo.groupby('race_id')['horse_elo_pre_race'].transform('mean')
        df_elo['elo_vs_field_median'] = df_elo['horse_elo_pre_race'] - df_elo.groupby('race_id')['horse_elo_pre_race'].transform('median')
        df_elo['elo_vs_field_max'] = df_elo['horse_elo_pre_race'] - df_elo.groupby('race_id')['horse_elo_pre_race'].transform('max')
        df_elo['elo_percentile_in_race'] = df_elo.groupby('race_id')['horse_elo_pre_race'].rank(pct=True)
        
        # ELO rankings
        df_elo['elo_rank_in_race'] = df_elo.groupby('race_id')['horse_elo_pre_race'].rank(ascending=False, method='min')
        df_elo['is_elo_favorite'] = (df_elo['elo_rank_in_race'] == 1).astype(int)
        df_elo['is_elo_top3'] = (df_elo['elo_rank_in_race'] <= 3).astype(int)
        
        # ELO advantage over second
        def calc_elo_advantage(group):
            sorted_elos = group.sort_values(ascending=False).values
            if len(sorted_elos) >= 2:
                return sorted_elos[0] - sorted_elos[1]
            return 0
        
        elo_advantage = df_elo.groupby('race_id')['horse_elo_pre_race'].transform(
            lambda x: x.max() - x.nlargest(2).iloc[-1] if len(x) >= 2 else 0
        )
        df_elo['elo_advantage_over_second'] = np.where(
            df_elo['is_elo_favorite'] == 1,
            elo_advantage,
            0
        )
        
        # ELO categories
        df_elo['elo_high'] = (df_elo['horse_elo_pre_race'] >= 1600).astype(int)
        df_elo['elo_mid'] = df_elo['horse_elo_pre_race'].between(1400, 1600).astype(int)
        df_elo['elo_low'] = (df_elo['horse_elo_pre_race'] < 1400).astype(int)
    
    if 'elo_implied_prob' in df_elo.columns:
        df_elo['log_elo_prob'] = np.log1p(df_elo['elo_implied_prob'])
        df_elo['elo_prob_strong'] = (df_elo['elo_implied_prob'] > 0.25).astype(int)
        df_elo['elo_prob_weak'] = (df_elo['elo_implied_prob'] < 0.05).astype(int)
    
    logger.info("✅ ELO derived features complete")
    return df_elo


def create_rating_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates official rating momentum and trend features."""
    logger.info("\n" + "="*80)
    logger.info("RATING MOMENTUM FEATURES")
    logger.info("="*80)
    
    df_rating = df.copy()
    
    if 'or_change_vs_last_race' in df_rating.columns:
        df_rating['or_improving'] = (df_rating['or_change_vs_last_race'] > 0).astype(int)
        df_rating['or_declining'] = (df_rating['or_change_vs_last_race'] < 0).astype(int)
        df_rating['or_stable'] = (df_rating['or_change_vs_last_race'] == 0).astype(int)
        df_rating['or_big_jump'] = (df_rating['or_change_vs_last_race'] > 5).astype(int)
        df_rating['or_big_drop'] = (df_rating['or_change_vs_last_race'] < -5).astype(int)
    
    if 'or_trend_last_5' in df_rating.columns:
        df_rating['or_uptrend'] = (df_rating['or_trend_last_5'] > 0.5).astype(int)
        df_rating['or_downtrend'] = (df_rating['or_trend_last_5'] < -0.5).astype(int)
    
    if 'or_momentum_short_term' in df_rating.columns:
        df_rating['or_momentum_positive'] = (df_rating['or_momentum_short_term'] > 0).astype(int)
        df_rating['or_momentum_negative'] = (df_rating['or_momentum_short_term'] < 0).astype(int)
    
    logger.info("✅ Rating momentum features complete")
    return df_rating


def create_headgear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates headgear-related features."""
    logger.info("\n" + "="*80)
    logger.info("HEADGEAR FEATURES")
    logger.info("="*80)
    
    df_hg = df.copy()
    
    if 'headgear' in df_hg.columns:
        df_hg['has_headgear'] = (df_hg['headgear'].notna() & (df_hg['headgear'] != 'None')).astype(int)
        df_hg['has_blinkers'] = df_hg['headgear'].str.contains('Blinker', case=False, na=False).astype(int)
        df_hg['has_visor'] = df_hg['headgear'].str.contains('Visor', case=False, na=False).astype(int)
        df_hg['has_cheekpieces'] = df_hg['headgear'].str.contains('Cheek', case=False, na=False).astype(int)
        df_hg['has_hood'] = df_hg['headgear'].str.contains('Hood', case=False, na=False).astype(int)
        df_hg['has_tongue_tie'] = df_hg['headgear'].str.contains('Tongue', case=False, na=False).astype(int)
    
    if 'headgear_first_time' in df_hg.columns:
        # FIX: Fill NaN with 0 before converting to int. If status is unknown, assume False.
        df_hg['first_time_gear'] = df_hg['headgear_first_time'].fillna(0).astype(int)
    
    if 'headgear_change' in df_hg.columns:
        # FIX: Fill NaN with 0 before converting to int. If status is unknown, assume False.
        df_hg['gear_changed'] = df_hg['headgear_change'].fillna(0).astype(int)
    
    logger.info("✅ Headgear features complete")
    return df_hg

def create_class_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates class movement features."""
    logger.info("\n" + "="*80)
    logger.info("CLASS MOVEMENT FEATURES")
    logger.info("="*80)
    
    df_class = df.copy()
    
    if 'moving_up_in_class' in df_class.columns:
        # FIX: Fill NaN with 0 before converting. If unknown, assume not moving up.
        df_class['up_in_class'] = df_class['moving_up_in_class'].fillna(0).astype(int)
    
    if 'moving_down_in_class' in df_class.columns:
        # FIX: Fill NaN with 0 before converting. If unknown, assume not moving down.
        df_class['down_in_class'] = df_class['moving_down_in_class'].fillna(0).astype(int)
    
    if 'class' in df_class.columns:
        df_class['class_vs_field_avg'] = df_class['class'] - df_class.groupby('race_id')['class'].transform('mean')
    
    logger.info("✅ Class movement features complete")
    return df_class

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates data integrity and removes leaky columns."""
    logger.info("\n" + "="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)
    
    df_validated = df.copy()
    
    # Check critical columns
    critical_cols = ['race_id', 'date_of_race', 'horse', 'won']
    for col in critical_cols:
        if col in df_validated.columns and df_validated[col].isnull().any():
            logger.warning(f"Critical column {col} has NaN values")
    
    # Check for infinite values
    numeric_cols = df_validated.select_dtypes(include=np.number)
    if np.isinf(numeric_cols).any().any():
        logger.warning("Infinite values found, replacing with NaN")
        df_validated.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # =========================================================================
    # START OF MODIFIED SECTION
    # =========================================================================
    
    # Remove known leaky columns by substring matching
    # NOTE: We've removed 'pos' from this list to avoid being too aggressive.
    leakage_keywords = [
        'winning_distance', 'sp_win_return', 'ew_return', 'betfair_win_return',
        'place_return', 'betfair_lay_return', 'place_lay_return',
        'ip_min', 'ip_max', 'pre_min', 'pre_max',
        'betfair_sp', 'industry_sp', 'sp_fav',
        'betfair_rank', 'betfair_place_sp',
        'tick_reduction', 'tick_inflation', 'pct_bsp_reduction', 'pct_bsp_inflation',
        'placed_in_betfair_market'
    ]
    
    # Add a separate list for columns we want to remove by EXACT name match.
    # This is much safer and prevents accidental removal of good features.
    exact_leakage_columns_to_remove = [
        'pos', 
        'position' # In case you have another column name for finishing position
    ]
    
    # Find columns to remove based on the two lists
    found_by_keyword = [col for col in df_validated.columns 
                        if any(key in col.lower() for key in leakage_keywords)]
    
    found_by_exact_name = [col for col in df_validated.columns 
                           if col.lower() in exact_leakage_columns_to_remove]

    # Combine the lists and remove any duplicates
    found_leakage = sorted(list(set(found_by_keyword + found_by_exact_name)))

    # =========================================================================
    # END OF MODIFIED SECTION
    # =========================================================================
    
    if found_leakage:
        logger.warning(f"Removing {len(found_leakage)} potentially leaky columns: {found_leakage}")
        df_validated.drop(columns=found_leakage, inplace=True)
    
    logger.info("✅ Validation passed")
    return df_validated

def find_latest_file(directory: Path, pattern: str) -> Path:
    """Finds the most recently created file matching a pattern in a directory."""
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(
            f"No processed data files found in {directory} matching pattern '{pattern}'."
        )
    return sorted(matching_files)[-1]


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates comprehensive temporal features."""
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL FEATURES")
    logger.info("="*80)
    
    df_temp = df.copy()
    
    # Date decomposition
    df_temp['day_of_week'] = df_temp['date_of_race'].dt.dayofweek
    df_temp['day_of_month'] = df_temp['date_of_race'].dt.day
    df_temp['month'] = df_temp['date_of_race'].dt.month
    df_temp['quarter'] = df_temp['date_of_race'].dt.quarter
    df_temp['year'] = df_temp['date_of_race'].dt.year
    df_temp['week_of_year'] = df_temp['date_of_race'].dt.isocalendar().week
    df_temp['day_of_year'] = df_temp['date_of_race'].dt.dayofyear
    
    # Categorical time features
    df_temp['is_weekend'] = df_temp['day_of_week'].isin([5, 6]).astype(int)
    df_temp['is_monday'] = (df_temp['day_of_week'] == 0).astype(int)
    df_temp['is_friday'] = (df_temp['day_of_week'] == 4).astype(int)
    
    # Seasonal features
    df_temp['is_summer'] = df_temp['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df_temp['is_winter'] = df_temp['month'].isin([11, 12, 1, 2]).astype(int)
    df_temp['is_spring'] = df_temp['month'].isin([3, 4, 5]).astype(int)
    df_temp['is_autumn'] = df_temp['month'].isin([9, 10, 11]).astype(int)
    
    # Racing season indicators
    df_temp['flat_season'] = df_temp['month'].isin([3, 4, 5, 6, 7, 8, 9, 10]).astype(int)
    df_temp['jumps_season'] = df_temp['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    
    # If race_datetime exists, extract time of day
    if 'race_datetime' in df_temp.columns:
        df_temp['hour_of_day'] = df_temp['race_datetime'].dt.hour
        df_temp['is_afternoon'] = df_temp['hour_of_day'].between(12, 17).astype(int)
        df_temp['is_evening'] = (df_temp['hour_of_day'] >= 17).astype(int)
        df_temp['is_morning'] = (df_temp['hour_of_day'] < 12).astype(int)
    
    # Race sequence features (position on card)
    df_temp['race_number_on_card'] = df_temp.groupby(['track', 'date_of_race']).cumcount() + 1
    df_temp['total_races_on_card'] = df_temp.groupby(['track', 'date_of_race'])['race_id'].transform('count')
    df_temp['is_first_race'] = (df_temp['race_number_on_card'] == 1).astype(int)
    df_temp['is_last_race'] = (df_temp['race_number_on_card'] == df_temp['total_races_on_card']).astype(int)
    df_temp['race_position_pct'] = df_temp['race_number_on_card'] / df_temp['total_races_on_card']
    
    logger.info("✅ Temporal features complete")
    return df_temp


def create_track_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates track and going condition features."""
    logger.info("\n" + "="*80)
    logger.info("TRACK & CONDITION FEATURES")
    logger.info("="*80)
    
    df_track = df.copy()
    
    # Track encoding - frequency-based
    track_counts = df_track['track'].value_counts()
    df_track['track_frequency'] = df_track['track'].map(track_counts)
    df_track['track_rarity'] = 1 / (df_track['track_frequency'] + 1)
    
    # Going encoding with numerical scale
    going_firmness = {
        'Heavy': 1, 'Soft': 2, 'Yielding': 3, 'Good to Soft': 4,
        'Good to Yielding': 4.5, 'Good': 5, 'Good to Firm': 6, 'Firm': 7,
        'Standard': 5, 'Standard to Slow': 3, 'Slow': 2, 'Fast': 6
    }
    df_track['going_firmness_index'] = df_track['going'].map(going_firmness).fillna(5)
    
    # Going categories
    df_track['is_soft_ground'] = df_track['going'].str.contains('Soft|Heavy|Yielding', case=False, na=False).astype(int)
    df_track['is_good_ground'] = df_track['going'].str.contains('Good', case=False, na=False).astype(int)
    df_track['is_firm_ground'] = df_track['going'].str.contains('Firm', case=False, na=False).astype(int)
    
    # Distance categories
    if 'distance_m' in df_track.columns:
        df_track['is_sprint'] = (df_track['distance_m'] < 1400).astype(int)
        df_track['is_mile'] = df_track['distance_m'].between(1400, 1800).astype(int)
        df_track['is_middle_distance'] = df_track['distance_m'].between(1800, 2400).astype(int)
        df_track['is_staying'] = (df_track['distance_m'] >= 2400).astype(int)
        df_track['distance_km'] = df_track['distance_m'] / 1000
        df_track['distance_furlongs'] = df_track['distance_m'] / 201.168
    
    # Race type encoding
    if 'type' in df_track.columns:
        df_track['is_handicap'] = df_track['type'].str.contains('Handicap', case=False, na=False).astype(int)
        df_track['is_maiden'] = df_track['type'].str.contains('Maiden', case=False, na=False).astype(int)
        df_track['is_stakes'] = df_track['type'].str.contains('Stakes|Grade|Group|Listed', case=False, na=False).astype(int)
        df_track['is_novice'] = df_track['type'].str.contains('Novice', case=False, na=False).astype(int)
    
    # Class features
    if 'class' in df_track.columns:
        df_track['class_filled'] = df_track['class'].fillna(df_track['class'].median())
        df_track['is_high_class'] = (df_track['class_filled'] >= 5).astype(int)
        df_track['is_low_class'] = (df_track['class_filled'] <= 2).astype(int)
    
    logger.info("✅ Track & condition features complete")
    return df_track


def create_age_weight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates age and weight-related features."""
    logger.info("\n" + "="*80)
    logger.info("AGE & WEIGHT FEATURES")
    logger.info("="*80)
    
    df_aw = df.copy()
    
    # Age features
    if 'age' in df_aw.columns:
        df_aw['age_squared'] = df_aw['age'] ** 2
        df_aw['age_cubed'] = df_aw['age'] ** 3
        df_aw['is_young_horse'] = (df_aw['age'] < 4).astype(int)
        df_aw['is_prime_age'] = df_aw['age'].between(3, 6).astype(int)
        df_aw['is_veteran'] = (df_aw['age'] > 8).astype(int)
        df_aw['is_aged_3'] = (df_aw['age'] == 3).astype(int)
        df_aw['is_aged_4'] = (df_aw['age'] == 4).astype(int)
        df_aw['is_aged_5_plus'] = (df_aw['age'] >= 5).astype(int)
    
    # Weight features (relative to field)
    if 'weight_lbs' in df_aw.columns:
        df_aw['weight_vs_field_median'] = df_aw['weight_lbs'] - df_aw.groupby('race_id')['weight_lbs'].transform('median')
        df_aw['weight_vs_field_min'] = df_aw['weight_lbs'] - df_aw.groupby('race_id')['weight_lbs'].transform('min')
        df_aw['weight_vs_field_max'] = df_aw['weight_lbs'] - df_aw.groupby('race_id')['weight_lbs'].transform('max')
        df_aw['weight_percentile_in_race'] = df_aw.groupby('race_id')['weight_lbs'].rank(pct=True)
        df_aw['is_top_weight'] = (df_aw['weight_lbs'] == df_aw.groupby('race_id')['weight_lbs'].transform('max')).astype(int)
        df_aw['is_bottom_weight'] = (df_aw['weight_lbs'] == df_aw.groupby('race_id')['weight_lbs'].transform('min')).astype(int)
    
    # Official rating features (relative to field)
    if 'official_rating' in df_aw.columns:
        df_aw['or_percentile_in_race'] = df_aw.groupby('race_id')['official_rating'].rank(pct=True)
        df_aw['or_vs_field_median'] = df_aw['official_rating'] - df_aw.groupby('race_id')['official_rating'].transform('median')
        df_aw['or_vs_field_min'] = df_aw['official_rating'] - df_aw.groupby('race_id')['official_rating'].transform('min')
        df_aw['or_vs_field_max'] = df_aw['official_rating'] - df_aw.groupby('race_id')['official_rating'].transform('max')
        df_aw['is_top_rated'] = (df_aw['official_rating'] == df_aw.groupby('race_id')['official_rating'].transform('max')).astype(int)
        df_aw['is_bottom_rated'] = (df_aw['official_rating'] == df_aw.groupby('race_id')['official_rating'].transform('min')).astype(int)
        
        # Weight per rating unit
        if 'weight_lbs' in df_aw.columns:
            df_aw['weight_per_or_unit'] = df_aw['weight_lbs'] / (df_aw['official_rating'] + 1)
    
    # Age-experience interaction
    if 'age' in df_aw.columns and 'total_career_runs' in df_aw.columns:
        df_aw['age_times_experience'] = df_aw['age'] * df_aw['total_career_runs']
        df_aw['runs_per_year_of_age'] = df_aw['total_career_runs'] / (df_aw['age'] + 1)
    
    logger.info("✅ Age & weight features complete")
    return df_aw


def create_field_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features about the composition of the field."""
    logger.info("\n" + "="*80)
    logger.info("FIELD COMPOSITION FEATURES")
    logger.info("="*80)
    
    df_field = df.copy()
    
    # Field size features
    if 'runners' in df_field.columns:
        df_field['field_size_small'] = (df_field['runners'] < 8).astype(int)
        df_field['field_size_medium'] = df_field['runners'].between(8, 14).astype(int)
        df_field['field_size_large'] = (df_field['runners'] >= 15).astype(int)
        df_field['log_field_size'] = np.log1p(df_field['runners'])
    
    # Field quality variance
    if 'official_rating' in df_field.columns:
        df_field['field_or_range'] = df_field.groupby('race_id')['official_rating'].transform('max') - \
                                      df_field.groupby('race_id')['official_rating'].transform('min')
        df_field['field_competitiveness'] = df_field.groupby('race_id')['official_rating'].transform('std') / \
                                            (df_field.groupby('race_id')['official_rating'].transform('mean') + 1)
    
    # Age composition of field
    if 'age' in df_field.columns:
        df_field['avg_age_of_field'] = df_field.groupby('race_id')['age'].transform('mean')
        df_field['age_vs_field_avg'] = df_field['age'] - df_field['avg_age_of_field']
        df_field['is_youngest_in_race'] = (df_field['age'] == df_field.groupby('race_id')['age'].transform('min')).astype(int)
        df_field['is_oldest_in_race'] = (df_field['age'] == df_field.groupby('race_id')['age'].transform('max')).astype(int)
    
    # Experience composition of field
    if 'total_career_runs' in df_field.columns:
        df_field['avg_experience_of_field'] = df_field.groupby('race_id')['total_career_runs'].transform('mean')
        df_field['experience_vs_field_avg'] = df_field['total_career_runs'] - df_field['avg_experience_of_field']
        df_field['is_most_experienced'] = (df_field['total_career_runs'] == df_field.groupby('race_id')['total_career_runs'].transform('max')).astype(int)
        df_field['is_least_experienced'] = (df_field['total_career_runs'] == df_field.groupby('race_id')['total_career_runs'].transform('min')).astype(int)
    
    # Stall position features
    if 'stall' in df_field.columns:
        df_field['stall_filled'] = df_field['stall'].fillna(df_field.groupby('race_id')['stall'].transform('median'))
        df_field['is_low_stall'] = (df_field['stall_filled'] <= 5).astype(int)
        df_field['is_high_stall'] = (df_field['stall_filled'] > 10).astype(int)
        df_field['stall_vs_field_size'] = df_field['stall_filled'] / (df_field['runners'] + 1)
    
    logger.info("✅ Field composition features complete")
    return df_field


def create_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates market-based features (SAFE - forecasted odds only)."""
    logger.info("\n" + "="*80)
    logger.info("MARKET FEATURES (SAFE)")
    logger.info("="*80)
    
    df_market = df.copy()
    
    # Forecasted odds features (SAFE - tissue price)
    if 'forecasted_odds' in df_market.columns:
        df_market['log_forecasted_odds'] = np.log1p(df_market['forecasted_odds'])
        df_market['implied_prob_forecast'] = 1 / (df_market['forecasted_odds'] + 0.001)
        df_market['forecast_rank'] = df_market.groupby('race_id')['forecasted_odds'].rank(method='min')
        df_market['is_forecast_favorite'] = (df_market['forecast_rank'] == 1).fillna(False).astype(int)
        df_market['is_forecast_outsider'] = (df_market['forecast_rank'] > df_market['runners'] * 0.75).fillna(False).astype(int)
    
    logger.info("✅ Market features complete")
    return df_market

def create_specialization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates specialization indicator features."""
    logger.info("\n" + "="*80)
    logger.info("SPECIALIZATION FEATURES")
    logger.info("="*80)
    
    df_spec = df.copy()
    
    # Course specialist
    if 'course_experience' in df_spec.columns:
        df_spec['course_specialist'] = (df_spec['course_experience'] >= 3).astype(int)
        df_spec['course_novice'] = (df_spec['course_experience'] == 0).astype(int)
        df_spec['has_course_win'] = (df_spec.get('course_win_rate', 0) > 0).astype(int)
    
    # Distance specialist
    if 'distance_experience' in df_spec.columns:
        df_spec['distance_specialist'] = (df_spec['distance_experience'] >= 3).astype(int)
        df_spec['distance_novice'] = (df_spec['distance_experience'] == 0).astype(int)
        df_spec['has_distance_win'] = (df_spec.get('distance_win_rate', 0) > 0).astype(int)
    
    # Surface specialist
    if 'surface_experience' in df_spec.columns:
        df_spec['surface_specialist'] = (df_spec['surface_experience'] >= 3).astype(int)
        df_spec['surface_novice'] = (df_spec['surface_experience'] == 0).astype(int)
        df_spec['has_surface_win'] = (df_spec.get('surface_win_rate', 0) > 0).astype(int)
    
    # Multiple specializations
    if all(col in df_spec.columns for col in ['course_specialist', 'distance_specialist', 'surface_specialist']):
        df_spec['double_specialist'] = ((df_spec['course_specialist'] == 1) & 
                                        (df_spec['distance_specialist'] == 1)).astype(int)
        df_spec['triple_specialist'] = ((df_spec['course_specialist'] == 1) & 
                                        (df_spec['distance_specialist'] == 1) & 
                                        (df_spec['surface_specialist'] == 1)).astype(int)
    
    logger.info("✅ Specialization features complete")
    return df_spec


def create_form_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates form trend and momentum features."""
    logger.info("\n" + "="*80)
    logger.info("FORM TREND FEATURES")
    logger.info("="*80)
    
    df_form = df.copy()
    
    # Win/place rate features
    if 'win_rate_last_5' in df_form.columns:
        # FIX: A boolean comparison on a column with NaNs results in pd.NA. Fill these with False.
        df_form['has_recent_win'] = (df_form['win_rate_last_5'] > 0).fillna(False).astype(int)
        df_form['has_recent_place'] = (df_form.get('place_rate_last_5', 0) > 0).fillna(False).astype(int)
        
    if 'place_rate_last_5' in df_form.columns and 'win_rate_last_5' in df_form.columns:
        df_form['place_to_win_ratio'] = np.where(
            df_form['win_rate_last_5'] > 0,
            df_form['place_rate_last_5'] / df_form['win_rate_last_5'],
            df_form['place_rate_last_5'] * 5
        )
    
    # Streak features
    if 'win_streak' in df_form.columns:
        # FIX: Handle potential NaNs
        df_form['on_winning_streak'] = (df_form['win_streak'] > 0).fillna(False).astype(int)
        df_form['long_winning_streak'] = (df_form['win_streak'] >= 2).fillna(False).astype(int)
    
    if 'races_since_win' in df_form.columns:
        # FIX: Handle potential NaNs
        df_form['long_dry_spell'] = (df_form['races_since_win'] > 10).fillna(False).astype(int)
        df_form['very_long_dry_spell'] = (df_form['races_since_win'] > 20).fillna(False).astype(int)
        df_form['recent_winner'] = (df_form['races_since_win'] <= 3).fillna(False).astype(int)
    
    # Consistency features
    if 'pos_consistency' in df_form.columns:
        # FIX: Handle potential NaNs
        df_form['is_consistent'] = (df_form['pos_consistency'] < 2).fillna(False).astype(int)
        df_form['is_inconsistent'] = (df_form['pos_consistency'] > 4).fillna(False).astype(int)
    
    # Form improvement indicators
    if 'improving_form' in df_form.columns:
        # FIX: The original source of the error. Fill NaN with 0 before converting.
        df_form['improving_form_binary'] = df_form['improving_form'].fillna(0).astype(int)
    
    logger.info("✅ Form trend features complete")
    return df_form

def create_recency_freshness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates recency and freshness features."""
    logger.info("\n" + "="*80)
    logger.info("RECENCY & FRESHNESS FEATURES")
    logger.info("="*80)
    
    df_rec = df.copy()
    
    if 'days_since_last_time_out' in df_rec.columns:
        df_rec['log_days_since_last'] = np.log1p(df_rec['days_since_last_time_out'])
        df_rec['fresh_horse'] = (df_rec['days_since_last_time_out'] >= 60).astype(int)
        df_rec['very_fresh'] = (df_rec['days_since_last_time_out'] >= 90).astype(int)
        df_rec['quick_return'] = (df_rec['days_since_last_time_out'] <= 7).astype(int)
        df_rec['very_quick_return'] = (df_rec['days_since_last_time_out'] <= 3).astype(int)
        df_rec['optimal_rest'] = df_rec['days_since_last_time_out'].between(14, 30).astype(int)
        
        # Layoff categories
        df_rec['layoff_0_14_days'] = (df_rec['days_since_last_time_out'] <= 14).astype(int)
        df_rec['layoff_15_30_days'] = df_rec['days_since_last_time_out'].between(15, 30).astype(int)
        df_rec['layoff_31_60_days'] = df_rec['days_since_last_time_out'].between(31, 60).astype(int)
        df_rec['layoff_61_plus_days'] = (df_rec['days_since_last_time_out'] > 60).astype(int)
    
    # Running frequency
    if 'runs_last_30_days' in df_rec.columns:
        df_rec['frequent_runner'] = (df_rec['runs_last_30_days'] >= 3).astype(int)
        df_rec['infrequent_runner'] = (df_rec['runs_last_30_days'] == 0).astype(int)
    
    if 'avg_days_between_races' in df_rec.columns:
        df_rec['log_avg_days_between'] = np.log1p(df_rec['avg_days_between_races'])
    
    logger.info("✅ Recency & freshness features complete")
    return df_rec


def create_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates jockey and trainer performance features."""
    logger.info("\n" + "="*80)
    logger.info("JOCKEY & TRAINER FEATURES")
    logger.info("="*80)
    
    df_jt = df.copy()
    
    # Jockey strike rates
    if 'jockey_rides_14d' in df_jt.columns and 'jockey_form_wins_14d' in df_jt.columns:
        df_jt['jockey_strike_rate_14d'] = np.where(
            df_jt['jockey_rides_14d'] > 0,
            df_jt['jockey_form_wins_14d'] / df_jt['jockey_rides_14d'],
            0.0
        )
        df_jt['jockey_in_form'] = (df_jt['jockey_strike_rate_14d'] > 0.15).astype(int)
        df_jt['jockey_hot_streak'] = (df_jt['jockey_strike_rate_14d'] > 0.25).astype(int)
        df_jt['jockey_cold_streak'] = (df_jt['jockey_strike_rate_14d'] < 0.05).astype(int)
    
    # Trainer strike rates
    if 'trainer_runners_14d' in df_jt.columns and 'trainer_form_wins_14d' in df_jt.columns:
        df_jt['trainer_strike_rate_14d'] = np.where(
            df_jt['trainer_runners_14d'] > 0,
            df_jt['trainer_form_wins_14d'] / df_jt['trainer_runners_14d'],
            0.0
        )
        df_jt['trainer_in_form'] = (df_jt['trainer_strike_rate_14d'] > 0.15).astype(int)
        df_jt['trainer_hot_streak'] = (df_jt['trainer_strike_rate_14d'] > 0.25).astype(int)
        df_jt['trainer_cold_streak'] = (df_jt['trainer_strike_rate_14d'] < 0.05).astype(int)
    
    # Combination features
    if 'jockey_horse_combo_runs' in df_jt.columns and 'jockey_horse_combo_wins' in df_jt.columns:
        df_jt['jockey_horse_combo_rate'] = np.where(
            df_jt['jockey_horse_combo_runs'] > 0,
            df_jt['jockey_horse_combo_wins'] / df_jt['jockey_horse_combo_runs'],
            0.0
        )
        df_jt['proven_combination'] = (df_jt['jockey_horse_combo_runs'] >= 3).astype(int)
        df_jt['winning_combination'] = ((df_jt['jockey_horse_combo_wins'] > 0) & 
                                        (df_jt['jockey_horse_combo_runs'] >= 2)).astype(int)
    
    # Jockey/trainer activity levels
    if 'jockey_rides_14d' in df_jt.columns:
        df_jt['jockey_busy'] = (df_jt['jockey_rides_14d'] > 20).astype(int)
        df_jt['jockey_quiet'] = (df_jt['jockey_rides_14d'] < 5).astype(int)
    
    if 'trainer_runners_14d' in df_jt.columns:
        df_jt['trainer_busy'] = (df_jt['trainer_runners_14d'] > 10).astype(int)
        df_jt['trainer_quiet'] = (df_jt['trainer_runners_14d'] < 3).astype(int)
    
    logger.info("✅ Jockey & trainer features complete")
    return df_jt


def save_feature_names(df: pd.DataFrame, output_path: Path) -> None:
    """Save list of all engineered feature names to a text file."""
    logger.info("\nSaving feature names...")
    
    # Exclude non-feature columns
    exclude_cols = [
        'race_id', 'date_of_race', 'race_datetime', 'betting_deadline',
        'horse', 'jockey', 'trainer', 'won', 'placed', 'track', 'course',
        'going', 'type', 'headgear', 'num_places'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = sorted(feature_cols)
    
    with open(output_path, 'w') as f:
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        for col in feature_cols:
            f.write(f"{col}\n")
    
    logger.info(f"✅ Saved {len(feature_cols)} feature names to {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("\n" + "="*80)
    logger.info("ENHANCED FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    INPUT_FILE = find_latest_file(PROCESSED_DATA_DIR, FILE_PATTERN)
    
    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Output file: {OUTPUT_DATA_FILE}")
    logger.info(f"Trackers file: {OUTPUT_TRACKERS_FILE}")
    
    # =========================================================================
    # CHECKPOINT LOGIC: Load from checkpoint or run the slow step
    # =========================================================================
    if CHECKPOINT_DF_FILE.exists() and CHECKPOINT_TRACKERS_FILE.exists():
        logger.info(f"\nFound checkpoint files. Loading...")
        
        logger.info(f"Loading dataframe from {CHECKPOINT_DF_FILE}...")
        df_with_chrono = pd.read_parquet(CHECKPOINT_DF_FILE)
        
        logger.info(f"Loading state trackers from {CHECKPOINT_TRACKERS_FILE}...")
        with open(CHECKPOINT_TRACKERS_FILE, 'rb') as f:
            state_trackers = pickle.load(f)

        logger.info("✅ Checkpoint loaded successfully.")

    else:
        logger.info("\nCheckpoint files not found. Running full chronological feature engineering...")
        
        # Load initial raw data
        logger.info("\nLoading initial data...")
        df = pd.read_parquet(INPUT_FILE)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df['date_of_race'].min()} to {df['date_of_race'].max()}")
        
        # Step 1: Create chronological features (THE SLOW PART)
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CHRONOLOGICAL FEATURES")
        logger.info("="*80)
        
        df_with_chrono, state_trackers = create_chronological_features(
            df,
            state_trackers=None,
            elo_config=EloConfig(),
            history_config=HistoryConfig(),
            feature_config=FeatureConfig()
        )
        
        # SAVE THE CHECKPOINT FILES
        logger.info(f"\nSaving checkpoint dataframe to {CHECKPOINT_DF_FILE}...")
        df_with_chrono.to_parquet(CHECKPOINT_DF_FILE, index=False, compression='snappy')
        
        logger.info(f"Saving checkpoint trackers to {CHECKPOINT_TRACKERS_FILE}...")
        with open(CHECKPOINT_TRACKERS_FILE, 'wb') as f:
            pickle.dump(state_trackers, f)
        logger.info("✅ Checkpoint files saved.")

    # =========================================================================
    # Pipeline continues from here, using the loaded or newly-created data
    # =========================================================================
    
    logger.info(f"After chronological features: {len(df_with_chrono.columns)} columns")
    
    # Step 2: Temporal features
    df_with_temporal = create_temporal_features(df_with_chrono)
    logger.info(f"After temporal features: {len(df_with_temporal.columns)} columns")
    
    # Step 3: Track & condition features
    df_with_track = create_track_condition_features(df_with_temporal)
    logger.info(f"After track features: {len(df_with_track.columns)} columns")
    
    # Step 4: Age & weight features
    df_with_age_weight = create_age_weight_features(df_with_track)
    logger.info(f"After age/weight features: {len(df_with_age_weight.columns)} columns")
    
    # Step 5: Field composition features
    df_with_field = create_field_composition_features(df_with_age_weight)
    logger.info(f"After field features: {len(df_with_field.columns)} columns")
    
    # Step 6: Market features (safe)
    df_with_market = create_market_features(df_with_field)
    logger.info(f"After market features: {len(df_with_market.columns)} columns")
    
    # Step 7: Specialization features
    df_with_spec = create_specialization_features(df_with_market)
    logger.info(f"After specialization features: {len(df_with_spec.columns)} columns")
    
    # Step 8: Form trend features
    df_with_form = create_form_trend_features(df_with_spec)
    logger.info(f"After form features: {len(df_with_form.columns)} columns")
    
    # Step 9: Recency & freshness features
    df_with_recency = create_recency_freshness_features(df_with_form)
    logger.info(f"After recency features: {len(df_with_recency.columns)} columns")
    
    # Step 10: Jockey & trainer features
    df_with_jt = create_jockey_trainer_features(df_with_recency)
    logger.info(f"After J/T features: {len(df_with_jt.columns)} columns")
    
    # Step 11: ELO derived features
    df_with_elo_derived = create_elo_derived_features(df_with_jt)
    logger.info(f"After ELO derived features: {len(df_with_elo_derived.columns)} columns")
    
    # Step 12: Rating momentum features
    df_with_rating_momentum = create_rating_momentum_features(df_with_elo_derived)
    logger.info(f"After rating momentum features: {len(df_with_rating_momentum.columns)} columns")
    
    # Step 13: Headgear features
    df_with_headgear = create_headgear_features(df_with_rating_momentum)
    logger.info(f"After headgear features: {len(df_with_headgear.columns)} columns")
    
    # Step 14: Class movement features
    df_with_class = create_class_movement_features(df_with_headgear)
    logger.info(f"After class features: {len(df_with_class.columns)} columns")
    
    # Step 15: Interaction features
    df_with_interactions = create_interaction_features(df_with_class)
    logger.info(f"After interaction features: {len(df_with_interactions.columns)} columns")
    
    # Step 16: Validate and clean
    df_final = validate_data(df_with_interactions)
    logger.info(f"After validation: {len(df_final.columns)} columns")
    
    # =========================================================================
    # Save FINAL outputs
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("SAVING FINAL OUTPUTS")
    logger.info("="*80)
    
    # Save feature-engineered data to its final destination
    logger.info(f"Saving data to {OUTPUT_DATA_FILE}...")
    df_final.to_parquet(OUTPUT_DATA_FILE, index=False, compression='snappy')
    logger.info(f"✅ Saved {len(df_final):,} rows, {len(df_final.columns)} columns")
    
    # Save state trackers to their final destination
    logger.info(f"Saving state trackers to {OUTPUT_TRACKERS_FILE}...")
    with open(OUTPUT_TRACKERS_FILE, 'wb') as f:
        pickle.dump(state_trackers, f)
    logger.info("✅ Final state trackers saved")
    
    # Save feature names
    save_feature_names(df_final, FEATURE_NAMES_FILE)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total rows: {len(df_final):,}")
    logger.info(f"Total columns: {len(df_final.columns)}")
    logger.info(f"Total races: {df_final['race_id'].nunique():,}")
    logger.info(f"Date range: {df_final['date_of_race'].min()} to {df_final['date_of_race'].max()}")
    
    # Memory usage
    memory_mb = df_final.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory usage: {memory_mb:.1f} MB")
    
    # Missing value summary
    missing_pct = (df_final.isnull().sum() / len(df_final) * 100).round(2)
    high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
    if len(high_missing) > 0:
        logger.info(f"\nColumns with >10% missing values: {len(high_missing)}")
        for col, pct in high_missing.head(10).items():
            logger.info(f"  {col}: {pct}%")
    
    logger.info("\n" + "="*80)
    logger.info("ALL DONE! ✅")
    logger.info("="*80)

if __name__ == "__main__":
    main()