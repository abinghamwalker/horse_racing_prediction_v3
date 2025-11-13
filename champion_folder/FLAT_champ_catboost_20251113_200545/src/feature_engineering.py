#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Horse Racing Prediction

Usage:
    python feature_engineering.py

This script:
1. Loads processed data from Notebook 01
2. Creates chronological features (ELO, historical performance)
3. Creates derived features (interactions, ratios)
4. Creates advanced market features
5. Saves feature-engineered data and state trackers

Output:
    - data/reduction_ready/03_features_engineered.parquet
    - data/reduction_ready/state_trackers.pkl
    - data/reduction_ready/feature_names.txt
"""

# Standard library imports
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path
import warnings
import pickle
from typing import Dict, Any, List, Deque
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use __file__ to dynamically find the project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(".").resolve()

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
REDUCTION_READY_DIR = PROJECT_ROOT / 'data' / 'reduction_ready'

OUTPUT_DATA_FILE = REDUCTION_READY_DIR / "03_features_engineered.parquet"
OUTPUT_TRACKERS_FILE = REDUCTION_READY_DIR / "state_trackers.pkl"
FEATURE_NAMES_FILE = REDUCTION_READY_DIR / "feature_names.txt"

FILE_PATTERN = "processed_race_data_*.parquet"

def find_latest_file(directory: Path, pattern: str) -> Path:
    """Finds the most recently created file matching a pattern in a directory."""
    
    # List all files matching the pattern
    matching_files = list(directory.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(
            f"No processed data files found in {directory} matching pattern '{pattern}'."
        )

    # Sort the files alphabetically. Since the pattern includes YYYYMMDD_HHMMSS, 
    # the last item in the sorted list will be the newest file.
    latest_file = sorted(matching_files)[-1]
    
    return latest_file

# Dynamically determine the input file path

INPUT_FILE = find_latest_file(PROCESSED_DATA_DIR, FILE_PATTERN)

# Create output directory§
REDUCTION_READY_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# ELO Rating System
ELO_DEFAULT_RATING = 1500
ELO_K_FACTOR = 32
ELO_REGRESSION_FACTOR = 0.005
ELO_FLOOR = 1000
ELO_CEILING = 2500

# History Tracking
HORSE_HISTORY_LEN = 20
JOCKEY_HISTORY_LEN = 50
TRAINER_HISTORY_LEN = 100

# Feature Engineering Thresholds
DEFAULT_AVG_POS = 10.0
DEFAULT_AVG_DAYS_BETWEEN_RACES = 21.0
RECENT_DAYS_WINDOW = 14
LONG_DRY_SPELL_THRESHOLD = 10
DISTANCE_SIMILARITY_THRESHOLD_M = 200
SPECIALIST_RUNS_THRESHOLD = 3
STRIKE_RATE_IN_FORM_THRESHOLD = 0.15
FRESH_HORSE_DAYS_THRESHOLD = 60
QUICK_RETURN_DAYS_THRESHOLD = 7
PRIME_AGE_RANGE = (3, 5)
INEXPERIENCED_RUNS_THRESHOLD = 5
COMPETITIVE_RACE_STD_DEV_THRESHOLD = 5.0
WEAK_FAVORITE_BSP_THRESHOLD = 4.0

GOING_ORDINAL_MAP = {
    'Firm': 5, 'Good to Firm': 4, 'Good': 3, 'Good to Yielding': 2.5,
    'Good to Soft': 2, 'Yielding': 1.5, 'Yielding to Soft': 1, 'Soft': 0,
    'Soft to Heavy': -0.5, 'Heavy': -1, 'Standard': 3, 'Standard to Slow': 1, 'Slow': 0
}
DEFAULT_GOING_VALUE = 3

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_num_places(runners: int) -> int:
    """Determines the number of place positions based on the number of runners."""
    if runners >= 8:
        return 3
    if runners >= 5:
        return 2
    return 0


def get_first_time_runner_features(row: pd.Series) -> Dict[str, Any]:
    """Generates default features for a first-time runner."""
    return {
        'win_rate_last_5': 0.0, 'place_rate_last_5': 0.0, 'avg_pos_last_5': DEFAULT_AVG_POS,
        'win_streak': 0, 'runs_last_30_days': 0, 'races_since_win': 0, 'races_since_place': 0,
        'avg_official_rating_last_5': row.get('official_rating', 60), 'or_change_vs_last_race': 0.0,
        'or_trend_last_5': 0.0, 'last_race_won': 0, 'improving_form': 0, 'moving_up_in_class': 0,
        'moving_down_in_class': 0, 'headgear_first_time': 0, 'headgear_change': 0,
        'avg_days_between_races': DEFAULT_AVG_DAYS_BETWEEN_RACES, 'pos_consistency': 0.0,
        'or_momentum_short_term': 0.0, 'best_pos_last_5': DEFAULT_AVG_POS,
        'worst_pos_last_5': DEFAULT_AVG_POS, 'pos_improvement_vs_last': 0,
        'surface_win_rate': 0.0, 'surface_place_rate': 0.0, 'surface_experience': 0,
        'going_change': 0, 'going_deterioration': 0, 'going_improvement': 0,
        'distance_experience': 0, 'distance_win_rate': 0.0, 'actual_distance_change_m': 0,
        'course_experience': 0, 'course_win_rate': 0.0,
    }


def calculate_historical_horse_features(h_hist: List[Dict[str, Any]], current_row: pd.Series) -> Dict[str, Any]:
    """Calculates features based on a horse's past performance history."""
    features = {}
    last_race = h_hist[-1]
    last_5 = h_hist[-5:]
    
    pos_last_5 = [r['pos'] for r in last_5 if r['pos'] != 99] or [DEFAULT_AVG_POS]
    features['win_rate_last_5'] = sum(r['won'] for r in last_5) / len(last_5)
    features['place_rate_last_5'] = sum(r['placed'] for r in last_5) / len(last_5)
    features['avg_pos_last_5'] = np.mean(pos_last_5)
    features['best_pos_last_5'] = np.min(pos_last_5)
    features['worst_pos_last_5'] = np.max(pos_last_5)
    features['pos_consistency'] = np.std(pos_last_5)
    features['last_race_won'] = 1 if last_race['pos'] == 1 else 0
    features['pos_improvement_vs_last'] = last_race['pos'] - features['avg_pos_last_5']

    win_streak_len = 0
    for race in reversed(h_hist):
        if race['won'] == 1:
            win_streak_len += 1
        else:
            break
    features['win_streak'] = win_streak_len

    wins = [i for i, r in enumerate(h_hist) if r['won']]
    places = [i for i, r in enumerate(h_hist) if r['placed']]
    features['races_since_win'] = len(h_hist) - wins[-1] - 1 if wins else len(h_hist)
    features['races_since_place'] = len(h_hist) - places[-1] - 1 if places else len(h_hist)
    
    features['avg_official_rating_last_5'] = np.mean([r['or'] for r in last_5])
    features['or_change_vs_last_race'] = current_row.get('official_rating', 60) - last_race['or']
    features['or_momentum_short_term'] = current_row.get('official_rating', 60) - features['avg_official_rating_last_5']
    
    ors = [r['or'] for r in last_5]
    features['or_trend_last_5'] = np.polyfit(range(len(ors)), ors, 1)[0] if len(ors) > 1 else 0.0

    if len(h_hist) >= 6:
        recent_pos = [r['pos'] for r in h_hist[-3:] if r['pos'] != 99]
        older_pos = [r['pos'] for r in h_hist[-6:-3] if r['pos'] != 99]
        features['improving_form'] = 1 if recent_pos and older_pos and np.mean(recent_pos) < np.mean(older_pos) else 0
    else:
        features['improving_form'] = 0

    last_3_class = [r['class'] for r in h_hist[-3:] if r.get('class') is not None]
    current_class = current_row.get('class')
    
    if last_3_class and not pd.isna(current_class):
        avg_class_last_3 = np.mean(last_3_class)
        features['moving_up_in_class'] = 1 if current_class > avg_class_last_3 else 0
        features['moving_down_in_class'] = 1 if current_class < avg_class_last_3 else 0
    else:
        features['moving_up_in_class'] = 0
        features['moving_down_in_class'] = 0

    features['runs_last_30_days'] = sum(1 for r in h_hist if (current_row['date_of_race'] - r['date']).days <= 30)
    
    days_between = [(h_hist[i]['date'] - h_hist[i-1]['date']).days for i in range(1, len(h_hist))]
    features['avg_days_between_races'] = np.mean(days_between) if days_between else DEFAULT_AVG_DAYS_BETWEEN_RACES
    
    current_headgear = current_row.get('headgear', 'None')
    last_headgear = last_race.get('headgear', 'None')
    
    features['headgear_change'] = 1 if current_headgear != last_headgear else 0
    features['headgear_first_time'] = 1 if (current_headgear != 'None' and last_headgear == 'None') else 0

    current_going = current_row.get('going', 'Good')
    going_hist = [r for r in h_hist if r['going'] == current_going]
    
    features['surface_experience'] = len(going_hist)
    features['surface_win_rate'] = sum(r['won'] for r in going_hist) / len(going_hist) if going_hist else 0.0
    features['surface_place_rate'] = sum(r['placed'] for r in going_hist) / len(going_hist) if going_hist else 0.0
    
    last_going_val = GOING_ORDINAL_MAP.get(last_race['going'], DEFAULT_GOING_VALUE)
    curr_going_val = GOING_ORDINAL_MAP.get(current_going, DEFAULT_GOING_VALUE)
    
    features['going_deterioration'] = 1 if curr_going_val < last_going_val else 0
    features['going_improvement'] = 1 if curr_going_val > last_going_val else 0
    features['going_change'] = 1 if last_race['going'] != current_going else 0

    current_dist = current_row.get('distance_m', 2000)
    dist_hist = [r for r in h_hist if abs(r.get('distance_m', 2000) - current_dist) < DISTANCE_SIMILARITY_THRESHOLD_M]
    
    features['distance_experience'] = len(dist_hist)
    features['distance_win_rate'] = sum(r['won'] for r in dist_hist) / len(dist_hist) if dist_hist else 0.0
    features['actual_distance_change_m'] = current_dist - last_race.get('distance_m', current_dist)

    current_course = current_row.get('course')
    course_hist = [r for r in h_hist if r.get('course') == current_course]
    
    features['course_experience'] = len(course_hist)
    features['course_win_rate'] = sum(r['won'] for r in course_hist) / len(course_hist) if course_hist else 0.0
    
    return features


# =============================================================================
# CHRONOLOGICAL FEATURE ENGINEERING
# =============================================================================

def create_chronological_features(df: pd.DataFrame, state_trackers: Dict = None) -> tuple:
    """Generates chronological features with ELO ratings."""
    logger.info("\n" + "="*80)
    logger.info("CHRONOLOGICAL FEATURE ENGINEERING")
    logger.info("="*80)

    if state_trackers:
        logger.info("Loading existing state trackers...")
        horse_elo = state_trackers['horse_elo']
        horse_history = state_trackers['horse_history']
        jockey_history = state_trackers['jockey_history']
        trainer_history = state_trackers['trainer_history']
    else:
        logger.info("Initializing new state trackers...")
        horse_elo = defaultdict(lambda: ELO_DEFAULT_RATING)
        horse_history = defaultdict(lambda: deque(maxlen=HORSE_HISTORY_LEN))
        jockey_history = defaultdict(lambda: deque(maxlen=JOCKEY_HISTORY_LEN))
        trainer_history = defaultdict(lambda: deque(maxlen=TRAINER_HISTORY_LEN))

    df_copy = df.copy()
    
    if 'is_winner' not in df_copy.columns:
        df_copy['is_winner'] = (df_copy['won'] == 1).astype(int)
    
    if 'num_places' not in df_copy.columns:
        df_copy['num_places'] = df_copy['runners'].apply(get_num_places)
    
    if 'placed' not in df_copy.columns:
        df_copy['placed'] = (
            (df_copy['won'] == 1) |
            ((df_copy.get('pos', 99) <= df_copy['num_places']) & (df_copy.get('pos', 99) < 99))
        ).astype(int)
    
    df_sorted = df_copy.sort_values(by=['date_of_race', 'race_datetime', 'race_id']).reset_index(drop=True)
    race_groups = df_sorted.groupby('race_id', sort=False)
    feature_list = []
    
    logger.info(f"Processing {len(race_groups):,} races chronologically...")

    for race_id, race_df in tqdm(race_groups, desc="Building Features"):
        num_runners = len(race_df)
        if num_runners < 2:
            continue

        pre_race_horse_elos = [horse_elo[h] for h in race_df['horse']]
        
        elo_strengths = np.array([10**(elo/400) for elo in pre_race_horse_elos])
        elo_total = elo_strengths.sum()
        elo_probs = elo_strengths / elo_total if elo_total > 0 else np.zeros(num_runners)

        race_sum_or = race_df['official_rating'].sum()
        race_sum_weight = race_df['weight_lbs'].sum()

        for i, (idx, row) in enumerate(race_df.iterrows()):
            if pd.isna(row['horse']) or pd.isna(row['jockey']) or pd.isna(row['trainer']):
                continue

            features = {'index': idx}
            
            features['horse_elo_pre_race'] = pre_race_horse_elos[i]
            features['elo_implied_prob'] = elo_probs[i]
            
            num_opponents = num_runners - 1
            if num_opponents > 0:
                field_avg_or_excl_self = (race_sum_or - row['official_rating']) / num_opponents
                features['field_quality_avg_or'] = field_avg_or_excl_self
                features['weight_vs_field_avg'] = row['weight_lbs'] - ((race_sum_weight - row['weight_lbs']) / num_opponents)
            else:
                features['field_quality_avg_or'] = row['official_rating']
                features['weight_vs_field_avg'] = 0
            
            features['field_quality_std_or'] = race_df['official_rating'].std()

            h_hist = list(horse_history[row['horse']])
            if not h_hist:
                features.update(get_first_time_runner_features(row))
            else:
                features.update(calculate_historical_horse_features(h_hist, row))

            j_hist = list(jockey_history[row['jockey']])
            t_hist = list(trainer_history[row['trainer']])
            j_recent = [r for r in j_hist if (row['date_of_race'] - r['date']).days <= RECENT_DAYS_WINDOW]
            t_recent = [r for r in t_hist if (row['date_of_race'] - r['date']).days <= RECENT_DAYS_WINDOW]
            
            features['jockey_form_wins_14d'] = sum(r['won'] for r in j_recent)
            features['jockey_rides_14d'] = len(j_recent)
            features['trainer_form_wins_14d'] = sum(r['won'] for r in t_recent)
            features['trainer_runners_14d'] = len(t_recent)
            
            combo_runs = [r for r in h_hist if r.get('jockey') == row['jockey']]
            features['jockey_horse_combo_runs'] = len(combo_runs)
            features['jockey_horse_combo_wins'] = sum(r['won'] for r in combo_runs)
            
            feature_list.append(features)

        # Post-race ELO updates
        num_opponents = num_runners - 1
        
        if num_opponents > 0:
            elo_changes = []
            
            for i, (idx, row) in enumerate(race_df.iterrows()):
                current_pos = row.get('pos', 99)
                if pd.isna(current_pos):
                    current_pos = 99
                current_pos = int(current_pos) if current_pos != 99 else num_runners + 1
                
                pos_column = race_df['pos'].fillna(99)
                opponents_beaten = (pos_column > current_pos).sum()
                non_finishers = (pos_column == 99).sum()
                opponents_beaten += non_finishers
                
                if current_pos > num_runners:
                    opponents_beaten -= 1
                
                actual_score = opponents_beaten / num_opponents

                horse_expected_score = sum(
                    1 / (1 + 10**((opp_elo - pre_race_horse_elos[i]) / 400)) 
                    for j, opp_elo in enumerate(pre_race_horse_elos) if i != j
                ) / num_opponents
                
                elo_change = ELO_K_FACTOR * (actual_score - horse_expected_score)
                elo_changes.append((row['horse'], elo_change))
            
            for horse, elo_change in elo_changes:
                old_elo = horse_elo[horse]
                new_elo = old_elo + elo_change
                new_elo = new_elo + (ELO_DEFAULT_RATING - new_elo) * ELO_REGRESSION_FACTOR
                new_elo = max(ELO_FLOOR, min(ELO_CEILING, new_elo))
                horse_elo[horse] = new_elo
        
        for idx, row in race_df.iterrows():
            history_record = {
                'date': row['date_of_race'],
                'pos': row.get('pos', 99),
                'won': row['is_winner'],
                'placed': row['placed'],
                'class': row.get('class'),
                'or': row.get('official_rating', 60),
                'going': row.get('going', 'Good'),
                'jockey': row.get('jockey'),
                'headgear': row.get('headgear', 'None'),
                'distance_m': row.get('distance_m', 2000),
                'course': row.get('course'),
            }
            horse_history[row['horse']].append(history_record)
            jockey_history[row['jockey']].append(history_record)
            trainer_history[row['trainer']].append(history_record)

    chrono_features_df = pd.DataFrame(feature_list).set_index('index')
    df_with_chrono = df_sorted.join(chrono_features_df)
    
    final_state_trackers = {
        'horse_elo': horse_elo,
        'horse_history': horse_history,
        'jockey_history': jockey_history,
        'trainer_history': trainer_history
    }
    
    elo_values = list(horse_elo.values())
    logger.info(f"\nELO Statistics:")
    logger.info(f"  Mean: {np.mean(elo_values):.1f}")
    logger.info(f"  Std: {np.std(elo_values):.1f}")
    logger.info(f"  Range: [{np.min(elo_values):.1f}, {np.max(elo_values):.1f}]")
    
    return df_with_chrono, final_state_trackers


# =============================================================================
# DERIVED FEATURES
# =============================================================================

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
        
        if 'sp_fav' in df_final.columns:
            df_final['is_favorite'] = (df_final['sp_fav'] == 1).astype(int)
        else:
            df_final['is_favorite'] = (df_final['bsp_rank'] == 1).astype(int)
        
        df_final['is_second_favorite'] = (df_final['bsp_rank'] == 2).astype(int)
        df_final['is_outsider'] = (df_final['bsp_rank'] > df_final['runners'] * 0.75).astype(int)
        df_final['weak_favorite'] = ((df_final['is_favorite'] == 1) & (df_final['betfair_sp'] > WEAK_FAVORITE_BSP_THRESHOLD)).astype(int)
        
        if 'elo_implied_prob' in df_final.columns:
            df_final['value_signal_elo_vs_bsp'] = df_final['elo_implied_prob'] - df_final['implied_prob_bsp']

    # Form features
    if 'win_rate_last_5' in df_final.columns and 'place_rate_last_5' in df_final.columns:
        df_final['place_to_win_ratio'] = np.where(
            df_final['win_rate_last_5'] > 0,
            df_final['place_rate_last_5'] / df_final['win_rate_last_5'],
            np.nan
        )
    
    if 'win_streak' in df_final.columns:
        df_final['on_winning_streak'] = (df_final['win_streak'] > 0).astype(int)
    
    if 'races_since_win' in df_final.columns:
        df_final['long_dry_spell'] = (df_final['races_since_win'] > LONG_DRY_SPELL_THRESHOLD).astype(int)

    # Specialization
    if 'course_experience' in df_final.columns:
        df_final['course_specialist'] = (df_final['course_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    
    if 'distance_experience' in df_final.columns:
        df_final['distance_specialist'] = (df_final['distance_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    
    if 'surface_experience' in df_final.columns:
        df_final['surface_specialist'] = (df_final['surface_experience'] >= SPECIALIST_RUNS_THRESHOLD).astype(int)
    
    if 'course_specialist' in df_final.columns and 'distance_specialist' in df_final.columns:
        df_final['double_specialist'] = ((df_final['course_specialist'] == 1) & (df_final['distance_specialist'] == 1)).astype(int)

    # Field context
    if 'official_rating' in df_final.columns and 'field_quality_avg_or' in df_final.columns:
        df_final['or_vs_field_quality'] = df_final['official_rating'] - df_final['field_quality_avg_or']
        df_final['above_average_runner'] = (df_final['or_vs_field_quality'] > 0).astype(int)

    # Jockey/Trainer strike rates
    if 'jockey_form_wins_14d' in df_final.columns and 'jockey_rides_14d' in df_final.columns:
        df_final['jockey_strike_rate_14d'] = np.where(
            df_final['jockey_rides_14d'] > 0,
            df_final['jockey_form_wins_14d'] / df_final['jockey_rides_14d'],
            0.0
        )
    
    if 'trainer_form_wins_14d' in df_final.columns and 'trainer_runners_14d' in df_final.columns:
        df_final['trainer_strike_rate_14d'] = np.where(
            df_final['trainer_runners_14d'] > 0,
            df_final['trainer_form_wins_14d'] / df_final['trainer_runners_14d'],
            0.0
        )

    # Freshness
    if 'days_since_last_time_out' in df_final.columns:
        df_final['fresh_horse'] = (df_final['days_since_last_time_out'] >= FRESH_HORSE_DAYS_THRESHOLD).astype(int)
        df_final['quick_return'] = (df_final['days_since_last_time_out'] <= QUICK_RETURN_DAYS_THRESHOLD).astype(int)
        df_final['log_days_since_last'] = np.log1p(df_final['days_since_last_time_out'])
    
    if 'age' in df_final.columns:
        df_final['prime_age'] = df_final['age'].between(PRIME_AGE_RANGE[0], PRIME_AGE_RANGE[1]).astype(int)
        df_final['age_squared'] = df_final['age'] ** 2

    # Temporal
    if 'month' not in df_final.columns and 'date_of_race' in df_final.columns:
        df_final['month'] = df_final['date_of_race'].dt.month
    
    if 'day_of_week' not in df_final.columns and 'date_of_race' in df_final.columns:
        df_final['day_of_week'] = df_final['date_of_race'].dt.dayofweek
        df_final['is_weekend'] = df_final['day_of_week'].isin([5, 6]).astype(int)

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    logger.info("✅ Derived features complete")
    
    return df_final


# =============================================================================
# ADVANCED MARKET FEATURES
# =============================================================================

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


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame) -> None:
    """Validates data integrity."""
    logger.info("\n" + "="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)
    
    # Check critical columns
    critical_cols = ['race_id', 'date_of_race', 'horse', 'won']
    critical_nulls = [col for col in critical_cols if col in df.columns and df[col].isnull().any()]
    
    if critical_nulls:
        raise ValueError(f"Critical columns have NaNs: {critical_nulls}")
    
    # Check infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {col: np.isinf(df[col]).sum() for col in numeric_cols}
    inf_counts = {k: v for k, v in inf_counts.items() if v > 0}
    
    if inf_counts:
        raise ValueError(f"Infinite values found: {list(inf_counts.keys())}")
    
    # Check leakage columns
    leakage_keywords = ['winning_distance', 'sp_win_return', 'e_w_return', 'betfair_win_return',
                        'place_return', 'betfair_lay_return', 'ip_min', 'ip_max']
    
    found_leakage = []
    for keyword in leakage_keywords:
        found_leakage.extend([col for col in df.columns if keyword in col.lower()])
    
    if found_leakage:
        logger.warning(f"Removing {len(found_leakage)} leakage columns: {found_leakage}")
        df.drop(columns=found_leakage, inplace=True)
    
    logger.info("✅ Validation passed")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline execution."""
    try:
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        # Verify input file
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}\nRun data_processing.py first.")
        
        # Load data
        logger.info(f"\nLoading data from: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        logger.info(f"Loaded {len(df):,} records")
        
        # Prepare data
        logger.info("\nPreparing data...")
        
        # Convert dates to datetime
        if df['date_of_race'].dtype == 'object':
            df['date_of_race'] = pd.to_datetime(df['date_of_race'])
        if 'race_datetime' in df.columns and df['race_datetime'].dtype == 'object':
            df['race_datetime'] = pd.to_datetime(df['race_datetime'])
        
        # Convert weight
        if 'weight_lbs' not in df.columns:
            def weight_to_lbs(w):
                if pd.isna(w): return np.nan
                try:
                    if isinstance(w, (int, float)): return float(w)
                    if '-' in str(w):
                        s, p = str(w).split('-')
                        return int(s) * 14 + int(p)
                    return float(w)
                except: return np.nan
            df['weight_lbs'] = df['weight'].apply(weight_to_lbs) if 'weight' in df.columns else 140
        
        # Ensure distance_m
        if 'distance_m' not in df.columns:
            df['distance_m'] = 2000
        
        # Rename track to course
        if 'course' not in df.columns and 'track' in df.columns:
            df['course'] = df['track']
        
        logger.info("✅ Data ready")
        
        # Create features
        df_chrono, state_trackers = create_chronological_features(df)
        df_derived = create_derived_features(df_chrono)
        df_final = create_advanced_market_features(df_derived)
        
        # Validate
        validate_data(df_final)
        
        # Save data
        logger.info(f"\nSaving feature-engineered data to: {OUTPUT_DATA_FILE}")
        df_final.to_parquet(OUTPUT_DATA_FILE, index=False)
        
        # Save state trackers
        logger.info(f"Saving state trackers to: {OUTPUT_TRACKERS_FILE}")
        state_trackers_serializable = {
            'horse_elo': dict(state_trackers['horse_elo']),
            'horse_history': dict(state_trackers['horse_history']),
            'jockey_history': dict(state_trackers['jockey_history']),
            'trainer_history': dict(state_trackers['trainer_history']),
            'config': {
                'ELO_DEFAULT_RATING': ELO_DEFAULT_RATING,
                'ELO_K_FACTOR': ELO_K_FACTOR,
                'ELO_REGRESSION_FACTOR': ELO_REGRESSION_FACTOR,
                'ELO_FLOOR': ELO_FLOOR,
                'ELO_CEILING': ELO_CEILING,
            }
        }
        with open(OUTPUT_TRACKERS_FILE, 'wb') as f:
            pickle.dump(state_trackers_serializable, f)
        
        # Save feature names
        feature_cols = [col for col in df_final.columns if col not in ['won', 'date_of_race', 'race_id', 'horse']]
        with open(FEATURE_NAMES_FILE, 'w') as f:
            f.write('\n'.join(feature_cols))
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Final shape: {df_final.shape}")
        logger.info(f"Features created: {len(feature_cols)}")
        logger.info("\nOutput files:")
        logger.info(f"  1. {OUTPUT_DATA_FILE.name}")
        logger.info(f"  2. {OUTPUT_TRACKERS_FILE.name}")
        logger.info(f"  3. {FEATURE_NAMES_FILE.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("❌ PIPELINE FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())