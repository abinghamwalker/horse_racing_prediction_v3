#!/usr/bin/env python3
"""
Enhanced Temporally-Safe Chronological Feature Engineering

This enhanced version includes:
- Expanded win/place rate calculations (5, 10, 15, 20 races)
- Position statistics (median, variance, trends)
- Consistency metrics
- Rating volatility and peaks
- Extended specialization features
- Jockey/Trainer activity windows (30d, 90d)
- All features remain temporally safe
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class EloConfig:
    """Configuration for ELO rating system."""
    default_rating: float = 1500.0
    k_factor: float = 32.0
    regression_factor: float = 0.005
    floor: float = 1000.0
    ceiling: float = 2500.0
    base: float = 400.0


@dataclass(frozen=True)
class HistoryConfig:
    """Configuration for history tracking."""
    horse_history_length: int = 30  # Increased to support last_20 features
    jockey_history_length: int = 100
    trainer_history_length: int = 150
    recent_form_days: int = 14


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering thresholds."""
    default_avg_position: float = 10.0
    default_avg_days_between_races: float = 21.0
    distance_similarity_threshold_m: float = 200.0
    betting_deadline_offset_minutes: int = 5


@dataclass
class RaceHistoryRecord:
    """Immutable record of a single race performance."""
    race_datetime: datetime
    date_of_race: datetime
    position: int
    won: bool
    placed: bool
    official_rating: float
    race_class: Optional[int]
    going: str
    distance_m: float
    course: str
    jockey: str
    trainer: str
    headgear: str
    weight_lbs: float
    num_runners: int
    
    def __hash__(self):
        return hash((self.race_datetime, self.date_of_race))


@dataclass
class EntityState:
    """Tracks the historical state of an entity (horse/jockey/trainer)."""
    elo_rating: float = 1500.0
    history: Deque[RaceHistoryRecord] = field(default_factory=deque)
    last_updated: Optional[datetime] = None
    peak_rating: float = 1500.0  # Track peak OR
    
    def add_race(self, record: RaceHistoryRecord, max_length: Optional[int] = None):
        """Add a race record to history."""
        if max_length:
            while len(self.history) >= max_length:
                self.history.popleft()
        self.history.append(record)
        self.last_updated = record.race_datetime
        
        # Update peak rating
        if record.official_rating > self.peak_rating:
            self.peak_rating = record.official_rating


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """Manages entity states with temporal guarantees."""
    
    def __init__(
        self,
        elo_config: EloConfig,
        history_config: HistoryConfig,
        existing_state: Optional[Dict[str, Any]] = None
    ):
        self.elo_config = elo_config
        self.history_config = history_config
        
        if existing_state:
            logger.info("Loading existing state...")
            self.horse_states = existing_state.get('horse_states', defaultdict(EntityState))
            self.jockey_states = existing_state.get('jockey_states', defaultdict(EntityState))
            self.trainer_states = existing_state.get('trainer_states', defaultdict(EntityState))
        else:
            logger.info("Initializing new state...")
            self.horse_states: Dict[str, EntityState] = defaultdict(
                lambda: EntityState(elo_rating=elo_config.default_rating)
            )
            self.jockey_states: Dict[str, EntityState] = defaultdict(
                lambda: EntityState(elo_rating=elo_config.default_rating)
            )
            self.trainer_states: Dict[str, EntityState] = defaultdict(
                lambda: EntityState(elo_rating=elo_config.default_rating)
            )
    
    def get_horse_state_at_time(
        self, 
        horse_id: str, 
        cutoff_datetime: datetime
    ) -> EntityState:
        """Get horse state as it existed at a specific point in time."""
        state = self.horse_states[horse_id]
        
        valid_history = deque(
            [record for record in state.history if record.race_datetime < cutoff_datetime],
            maxlen=self.history_config.horse_history_length
        )
        
        snapshot = EntityState(
            elo_rating=state.elo_rating if not valid_history else state.elo_rating,
            history=valid_history,
            last_updated=state.last_updated,
            peak_rating=state.peak_rating
        )
        
        return snapshot
    
    def get_jockey_state_at_time(
        self,
        jockey_id: str,
        cutoff_datetime: datetime
    ) -> EntityState:
        """Get jockey state as it existed at cutoff time."""
        state = self.jockey_states[jockey_id]
        valid_history = deque(
            [record for record in state.history if record.race_datetime < cutoff_datetime],
            maxlen=self.history_config.jockey_history_length
        )
        return EntityState(
            elo_rating=state.elo_rating,
            history=valid_history,
            last_updated=state.last_updated
        )
    
    def get_trainer_state_at_time(
        self,
        trainer_id: str,
        cutoff_datetime: datetime
    ) -> EntityState:
        """Get trainer state as it existed at cutoff time."""
        state = self.trainer_states[trainer_id]
        valid_history = deque(
            [record for record in state.history if record.race_datetime < cutoff_datetime],
            maxlen=self.history_config.trainer_history_length
        )
        return EntityState(
            elo_rating=state.elo_rating,
            history=valid_history,
            last_updated=state.last_updated
        )
    
    def update_after_race(
        self,
        race_results: pd.DataFrame,
        elo_updates: Dict[str, float]
    ):
        """Update entity states after a race completes."""
        for idx, row in race_results.iterrows():
            record = self._create_race_record(row)
            
            # Update horse
            horse_id = row['horse']
            self.horse_states[horse_id].add_race(
                record, 
                max_length=self.history_config.horse_history_length
            )
            if horse_id in elo_updates:
                self.horse_states[horse_id].elo_rating = elo_updates[horse_id]
            
            # Update jockey
            jockey_id = row['jockey']
            self.jockey_states[jockey_id].add_race(
                record,
                max_length=self.history_config.jockey_history_length
            )
            
            # Update trainer
            trainer_id = row['trainer']
            self.trainer_states[trainer_id].add_race(
                record,
                max_length=self.history_config.trainer_history_length
            )
    
    def _create_race_record(self, row: pd.Series) -> RaceHistoryRecord:
        """Create an immutable race record from a dataframe row."""
        return RaceHistoryRecord(
            race_datetime=row['race_datetime'],
            date_of_race=row['date_of_race'],
            position=int(row.get('pos', 99)) if not pd.isna(row.get('pos')) else 99,
            won=bool(row.get('won', 0)),
            placed=bool(row.get('placed', 0)),
            official_rating=float(row.get('official_rating', 60)),
            race_class=int(row['class']) if 'class' in row and not pd.isna(row['class']) else None,
            going=str(row.get('going', 'Good')),
            distance_m=float(row.get('distance_m', 2000)),
            course=str(row.get('course', 'Unknown')),
            jockey=str(row.get('jockey', 'Unknown')),
            trainer=str(row.get('trainer', 'Unknown')),
            headgear=str(row.get('headgear', 'None')),
            weight_lbs=float(row.get('weight_lbs', 140)),
            num_runners=int(row.get('runners', 10))
        )
    
    def export_state(self) -> Dict[str, Any]:
        """Export state for persistence."""
        return {
            'horse_states': dict(self.horse_states),
            'jockey_states': dict(self.jockey_states),
            'trainer_states': dict(self.trainer_states),
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'num_horses': len(self.horse_states),
                'num_jockeys': len(self.jockey_states),
                'num_trainers': len(self.trainer_states),
            }
        }


# =============================================================================
# ELO CALCULATOR
# =============================================================================

class EloCalculator:
    """Calculates ELO ratings for horse racing with proper field adjustment."""
    
    def __init__(self, config: EloConfig):
        self.config = config
    
    def calculate_expected_score(
        self,
        rating: float,
        opponent_ratings: List[float]
    ) -> float:
        """Calculate expected score against multiple opponents."""
        if not opponent_ratings:
            return 0.5
        
        expected = sum(
            1 / (1 + 10 ** ((opp_rating - rating) / self.config.base))
            for opp_rating in opponent_ratings
        )
        
        return expected / len(opponent_ratings)
    
    def calculate_actual_score(
        self,
        position: int,
        num_runners: int,
        opponent_positions: List[int]
    ) -> float:
        """Calculate actual score based on finishing position."""
        if num_runners <= 1:
            return 0.5
        
        beaten = sum(1 for opp_pos in opponent_positions if opp_pos > position)
        non_finishers = sum(1 for opp_pos in opponent_positions if opp_pos >= 99)
        beaten += non_finishers
        
        if position >= 99:
            beaten = 0
        
        total_opponents = num_runners - 1
        return beaten / total_opponents if total_opponents > 0 else 0.5
    
    def calculate_rating_change(
        self,
        current_rating: float,
        expected_score: float,
        actual_score: float
    ) -> float:
        """Calculate ELO rating change."""
        return self.config.k_factor * (actual_score - expected_score)
    
    def apply_regression(self, rating: float) -> float:
        """Apply regression toward mean to prevent rating inflation."""
        adjustment = (self.config.default_rating - rating) * self.config.regression_factor
        return rating + adjustment
    
    def constrain_rating(self, rating: float) -> float:
        """Constrain rating within floor and ceiling."""
        return max(self.config.floor, min(self.config.ceiling, rating))
    
    def update_race_ratings(
        self,
        race_df: pd.DataFrame,
        pre_race_ratings: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate post-race ELO ratings for all horses in a race."""
        num_runners = len(race_df)
        if num_runners < 2:
            return pre_race_ratings
        
        new_ratings = {}
        
        for idx, row in race_df.iterrows():
            horse_id = row['horse']
            current_rating = pre_race_ratings.get(horse_id, self.config.default_rating)

            pos_val = row.get('pos')
            if pd.isna(pos_val):
                current_position = 99
            else:
                current_position = int(pos_val)

            opponent_ratings = [
                pre_race_ratings.get(opp_horse, self.config.default_rating)
                for opp_horse in race_df['horse']
                if opp_horse != horse_id
            ]
            
            opponent_positions = []
            for _, opp_row in race_df.iterrows():
                if opp_row['horse'] != horse_id:
                    opp_pos_val = opp_row.get('pos')
                    if pd.isna(opp_pos_val):
                        opponent_positions.append(99)
                    else:
                        opponent_positions.append(int(opp_pos_val))

            expected = self.calculate_expected_score(current_rating, opponent_ratings)
            actual = self.calculate_actual_score(current_position, num_runners, opponent_positions)
            
            rating_change = self.calculate_rating_change(current_rating, expected, actual)
            new_rating = current_rating + rating_change
            new_rating = self.apply_regression(new_rating)
            new_rating = self.constrain_rating(new_rating)
            
            new_ratings[horse_id] = new_rating
        
        return new_ratings


# =============================================================================
# ENHANCED FEATURE EXTRACTORS
# =============================================================================

class HorseFeatureExtractor:
    """Extracts comprehensive features from horse history."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_first_time_features(self, current_row: pd.Series) -> Dict[str, float]:
        """Features for horses with no prior history."""
        default_or = current_row.get('official_rating', 60)
        return {
            # Win/place rates - multiple windows
            'win_rate_last_5': 0.0,
            'win_rate_last_10': 0.0,
            'win_rate_last_15': 0.0,
            'win_rate_last_20': 0.0,
            'win_rate_career': 0.0,
            'place_rate_last_5': 0.0,
            'place_rate_last_10': 0.0,
            'place_rate_last_15': 0.0,
            'place_rate_career': 0.0,
            'top3_finish_rate_last_5': 0.0,
            'top3_finish_rate_last_10': 0.0,
            
            # Position statistics
            'avg_pos_last_5': self.config.default_avg_position,
            'avg_pos_last_10': self.config.default_avg_position,
            'avg_pos_last_15': self.config.default_avg_position,
            'median_pos_last_5': self.config.default_avg_position,
            'median_pos_last_10': self.config.default_avg_position,
            'best_pos_last_5': self.config.default_avg_position,
            'best_pos_last_10': self.config.default_avg_position,
            'worst_pos_last_5': self.config.default_avg_position,
            'worst_pos_last_10': self.config.default_avg_position,
            'pos_consistency': 0.0,
            'finishing_position_variance': 0.0,
            
            # Streaks and recency
            'win_streak': 0,
            'runs_last_30_days': 0,
            'runs_last_60_days': 0,
            'runs_last_90_days': 0,
            'races_since_win': 0,
            'races_since_place': 0,
            'races_since_top3': 0,
            
            # Rating features
            'avg_official_rating_last_5': default_or,
            'avg_official_rating_last_10': default_or,
            'or_rolling_avg_3': default_or,
            'or_change_vs_last_race': 0.0,
            'or_trend_last_5': 0.0,
            'or_momentum_short_term': 0.0,
            'or_volatility_last_5': 0.0,
            'or_peak_in_career': default_or,
            'or_decline_from_peak': 0.0,
            
            # Form indicators
            'last_race_won': 0,
            'last_race_placed': 0,
            'improving_form': 0,
            'recent_form_trend': 0.0,
            'consistency_score': 0.0,
            
            # Class movement
            'moving_up_in_class': 0,
            'moving_down_in_class': 0,
            'avg_class_last_5': np.nan,
            'class_change_vs_last': 0.0,
            
            # Equipment
            'headgear_first_time': 0,
            'headgear_change': 0,
            
            # Recency
            'avg_days_between_races': self.config.default_avg_days_between_races,
            'pos_improvement_vs_last': 0,
            
            # Specialization
            'surface_win_rate': 0.0,
            'surface_place_rate': 0.0,
            'surface_experience': 0,
            'surface_avg_position': self.config.default_avg_position,
            'surface_consistency': 0.0,
            'going_change': 0,
            'going_deterioration': 0,
            'going_improvement': 0,
            'distance_experience': 0,
            'distance_win_rate': 0.0,
            'distance_place_rate': 0.0,
            'avg_pos_at_distance': self.config.default_avg_position,
            'actual_distance_change_m': 0,
            'distance_change_pct': 0.0,
            'course_experience': 0,
            'course_win_rate': 0.0,
            'course_place_rate': 0.0,
            'course_avg_position': self.config.default_avg_position,
            'course_best_position': self.config.default_avg_position,
            'course_consistency': 0.0,
            
            # Career
            'total_career_runs': 0,
            'career_win_pct': 0.0,
            'career_place_pct': 0.0,
        }
    
    def extract_features(
        self,
        history: Deque[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract comprehensive features from horse history."""
        if not history:
            return self.extract_first_time_features(current_row)
        
        features = {}
        history_list = list(history)
        
        # Extract all feature categories
        features.update(self._extract_form_features(history_list, current_row))
        features.update(self._extract_rating_features(history_list, current_row))
        features.update(self._extract_class_features(history_list, current_row))
        features.update(self._extract_recency_features(history_list, current_row))
        features.update(self._extract_equipment_features(history_list, current_row))
        features.update(self._extract_specialization_features(history_list, current_row))
        features.update(self._extract_trend_features(history_list))
        features.update(self._extract_consistency_features(history_list))
        
        # Career stats
        features['total_career_runs'] = len(history_list)
        features['career_win_pct'] = sum(r.won for r in history_list) / len(history_list)
        features['career_place_pct'] = sum(r.placed for r in history_list) / len(history_list)
        
        return features
    
    def _extract_form_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract form-based features with multiple windows."""
        last_race = history[-1]
        
        features = {}
        
        # Win/place rates for multiple windows
        for window in [5, 10, 15, 20]:
            window_history = history[-window:] if len(history) >= window else history
            features[f'win_rate_last_{window}'] = sum(r.won for r in window_history) / len(window_history)
            
            if window <= 15:  # Only do place rates for smaller windows
                features[f'place_rate_last_{window}'] = sum(r.placed for r in window_history) / len(window_history)
            
            if window <= 10:  # Only do top3 for smaller windows
                top3 = sum(1 for r in window_history if r.position <= 3 and r.position < 99)
                features[f'top3_finish_rate_last_{window}'] = top3 / len(window_history)
        
        # Career rates
        features['win_rate_career'] = sum(r.won for r in history) / len(history)
        features['place_rate_career'] = sum(r.placed for r in history) / len(history)
        
        # Position statistics for multiple windows
        for window in [5, 10, 15]:
            window_history = history[-window:] if len(history) >= window else history
            positions = [r.position for r in window_history if r.position < 99]
            
            if positions:
                features[f'avg_pos_last_{window}'] = float(np.mean(positions))
                if window <= 10:
                    features[f'median_pos_last_{window}'] = float(np.median(positions))
                    features[f'best_pos_last_{window}'] = float(np.min(positions))
                    features[f'worst_pos_last_{window}'] = float(np.max(positions))
            else:
                features[f'avg_pos_last_{window}'] = self.config.default_avg_position
                if window <= 10:
                    features[f'median_pos_last_{window}'] = self.config.default_avg_position
                    features[f'best_pos_last_{window}'] = self.config.default_avg_position
                    features[f'worst_pos_last_{window}'] = self.config.default_avg_position
        
        # Position consistency
        last_10_pos = [r.position for r in history[-10:] if r.position < 99]
        if len(last_10_pos) > 1:
            features['pos_consistency'] = float(np.std(last_10_pos))
            features['finishing_position_variance'] = float(np.var(last_10_pos))
        else:
            features['pos_consistency'] = 0.0
            features['finishing_position_variance'] = 0.0
        
        # Streaks and recency
        features['last_race_won'] = 1 if last_race.won else 0
        features['last_race_placed'] = 1 if last_race.placed else 0
        features['win_streak'] = self._calculate_win_streak(history)
        
        # Races since events
        wins = [i for i, r in enumerate(history) if r.won]
        places = [i for i, r in enumerate(history) if r.placed]
        top3 = [i for i, r in enumerate(history) if r.position <= 3 and r.position < 99]
        
        features['races_since_win'] = len(history) - wins[-1] - 1 if wins else len(history)
        features['races_since_place'] = len(history) - places[-1] - 1 if places else len(history)
        features['races_since_top3'] = len(history) - top3[-1] - 1 if top3 else len(history)
        
        # Position improvement
        last_5_avg = features['avg_pos_last_5']
        features['pos_improvement_vs_last'] = last_race.position - last_5_avg
        
        # Form trend
        if len(history) >= 6:
            recent_pos = [r.position for r in history[-3:] if r.position < 99]
            older_pos = [r.position for r in history[-6:-3] if r.position < 99]
            features['improving_form'] = (
                1 if recent_pos and older_pos and np.mean(recent_pos) < np.mean(older_pos)
                else 0
            )
        else:
            features['improving_form'] = 0
        
        return features
    
    def _extract_rating_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract official rating features."""
        current_or = current_row.get('official_rating', 60)
        
        features = {}
        
        # Average ratings for multiple windows
        for window in [3, 5, 10]:
            window_history = history[-window:] if len(history) >= window else history
            features[f'avg_official_rating_last_{window}' if window != 3 else 'or_rolling_avg_3'] = \
                float(np.mean([r.official_rating for r in window_history]))
        
        # Rating changes
        last_race = history[-1]
        last_5 = history[-5:] if len(history) >= 5 else history
        avg_or_last_5 = np.mean([r.official_rating for r in last_5])
        
        features['or_change_vs_last_race'] = current_or - last_race.official_rating
        features['or_momentum_short_term'] = current_or - avg_or_last_5
        
        # Rating trend
        if len(last_5) > 1:
            ors = [r.official_rating for r in last_5]
            features['or_trend_last_5'] = float(np.polyfit(range(len(ors)), ors, 1)[0])
        else:
            features['or_trend_last_5'] = 0.0
        
        # Rating volatility
        if len(last_5) > 1:
            features['or_volatility_last_5'] = float(np.std([r.official_rating for r in last_5]))
        else:
            features['or_volatility_last_5'] = 0.0
        
        # Peak rating
        all_ratings = [r.official_rating for r in history]
        features['or_peak_in_career'] = float(np.max(all_ratings))
        features['or_decline_from_peak'] = current_or - features['or_peak_in_career']
        
        return features
    
    def _extract_class_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract class movement features."""
        last_5_with_class = [r for r in history[-5:] if r.race_class is not None]
        current_class = current_row.get('class')
        
        features = {}
        
        if last_5_with_class:
            avg_class = np.mean([r.race_class for r in last_5_with_class])
            features['avg_class_last_5'] = float(avg_class)
            
            if not pd.isna(current_class):
                features['moving_up_in_class'] = 1 if current_class > avg_class else 0
                features['moving_down_in_class'] = 1 if current_class < avg_class else 0
                features['class_change_vs_last'] = current_class - last_5_with_class[-1].race_class
            else:
                features['moving_up_in_class'] = 0
                features['moving_down_in_class'] = 0
                features['class_change_vs_last'] = 0.0
        else:
            features['avg_class_last_5'] = np.nan
            features['moving_up_in_class'] = 0
            features['moving_down_in_class'] = 0
            features['class_change_vs_last'] = 0.0
        
        return features
    
    def _extract_recency_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract recency and frequency features."""
        current_date = current_row['date_of_race']
        
        features = {}
        
        # Recent runs in multiple windows
        for days in [30, 60, 90]:
            runs = sum(
                1 for r in history
                if (current_date - r.date_of_race).days <= days
            )
            features[f'runs_last_{days}_days'] = runs
        
        # Average days between races
        if len(history) >= 2:
            gaps = [
                (history[i].date_of_race - history[i-1].date_of_race).days
                for i in range(1, len(history))
            ]
            features['avg_days_between_races'] = float(np.mean(gaps))
        else:
            features['avg_days_between_races'] = self.config.default_avg_days_between_races
        
        return features
    
    def _extract_equipment_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract headgear change features."""
        last_race = history[-1]
        current_headgear = current_row.get('headgear', 'None')
        last_headgear = last_race.headgear
        
        changed = 1 if current_headgear != last_headgear else 0
        first_time = (
            1 if current_headgear != 'None' and last_headgear == 'None'
            else 0
        )
        
        return {
            'headgear_change': changed,
            'headgear_first_time': first_time,
        }
    
    def _extract_specialization_features(
        self,
        history: List[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract course/distance/going specialization features."""
        current_going = current_row.get('going', 'Good')
        current_distance = current_row.get('distance_m', 2000)
        current_course = current_row.get('course', 'Unknown')
        last_race = history[-1]
        
        features = {}
        
        # Going analysis
        going_history = [r for r in history if r.going == current_going]
        features['surface_experience'] = len(going_history)
        features['surface_win_rate'] = (
            sum(r.won for r in going_history) / len(going_history)
            if going_history else 0.0
        )
        features['surface_place_rate'] = (
            sum(r.placed for r in going_history) / len(going_history)
            if going_history else 0.0
        )
        
        # Average position on surface
        going_positions = [r.position for r in going_history if r.position < 99]
        features['surface_avg_position'] = (
            float(np.mean(going_positions)) if going_positions 
            else self.config.default_avg_position
        )
        
        # Surface consistency
        if len(going_positions) > 1:
            features['surface_consistency'] = float(np.std(going_positions))
        else:
            features['surface_consistency'] = 0.0
        
        # Going change
        going_map = {
            'Firm': 5, 'Good to Firm': 4, 'Good': 3, 'Good to Yielding': 2.5,
            'Good to Soft': 2, 'Yielding': 1.5, 'Yielding to Soft': 1, 'Soft': 0,
            'Soft to Heavy': -0.5, 'Heavy': -1, 'Standard': 3, 'Standard to Slow': 1, 'Slow': 0
        }
        last_going_val = going_map.get(last_race.going, 3)
        curr_going_val = going_map.get(current_going, 3)
        
        features['going_change'] = 1 if last_race.going != current_going else 0
        features['going_deterioration'] = 1 if curr_going_val < last_going_val else 0
        features['going_improvement'] = 1 if curr_going_val > last_going_val else 0
        
        # Distance analysis
        distance_history = [
            r for r in history
            if abs(r.distance_m - current_distance) < self.config.distance_similarity_threshold_m
        ]
        features['distance_experience'] = len(distance_history)
        features['distance_win_rate'] = (
            sum(r.won for r in distance_history) / len(distance_history)
            if distance_history else 0.0
        )
        features['distance_place_rate'] = (
            sum(r.placed for r in distance_history) / len(distance_history)
            if distance_history else 0.0
        )
        
        # Average position at distance
        distance_positions = [r.position for r in distance_history if r.position < 99]
        features['avg_pos_at_distance'] = (
            float(np.mean(distance_positions)) if distance_positions
            else self.config.default_avg_position
        )
        
        # Distance change
        features['actual_distance_change_m'] = current_distance - last_race.distance_m
        features['distance_change_pct'] = (
            (current_distance - last_race.distance_m) / last_race.distance_m 
            if last_race.distance_m > 0 else 0.0
        )
        
        # Course analysis
        course_history = [r for r in history if r.course == current_course]
        features['course_experience'] = len(course_history)
        features['course_win_rate'] = (
            sum(r.won for r in course_history) / len(course_history)
            if course_history else 0.0
        )
        features['course_place_rate'] = (
            sum(r.placed for r in course_history) / len(course_history)
            if course_history else 0.0
        )
        
        # Course position statistics
        course_positions = [r.position for r in course_history if r.position < 99]
        if course_positions:
            features['course_avg_position'] = float(np.mean(course_positions))
            features['course_best_position'] = float(np.min(course_positions))
            if len(course_positions) > 1:
                features['course_consistency'] = float(np.std(course_positions))
            else:
                features['course_consistency'] = 0.0
        else:
            features['course_avg_position'] = self.config.default_avg_position
            features['course_best_position'] = self.config.default_avg_position
            features['course_consistency'] = 0.0
        
        return features
    
    def _extract_trend_features(
        self,
        history: List[RaceHistoryRecord]
    ) -> Dict[str, float]:
        """Extract performance trend features."""
        features = {}
        
        if len(history) >= 6:
            # Position momentum (comparing recent vs older)
            recent_3 = [r.position for r in history[-3:] if r.position < 99]
            older_3 = [r.position for r in history[-6:-3] if r.position < 99]
            
            if recent_3 and older_3:
                features['position_momentum_3_races'] = np.mean(older_3) - np.mean(recent_3)
            else:
                features['position_momentum_3_races'] = 0.0
        else:
            features['position_momentum_3_races'] = 0.0
        
        if len(history) >= 10:
            recent_5 = [r.position for r in history[-5:] if r.position < 99]
            older_5 = [r.position for r in history[-10:-5] if r.position < 99]
            
            if recent_5 and older_5:
                features['position_momentum_5_races'] = np.mean(older_5) - np.mean(recent_5)
            else:
                features['position_momentum_5_races'] = 0.0
        else:
            features['position_momentum_5_races'] = 0.0
        
        # Recent form trend (weighted average - more recent = higher weight)
        last_5 = history[-5:] if len(history) >= 5 else history
        positions = [r.position for r in last_5 if r.position < 99]
        if positions:
            weights = np.arange(1, len(positions) + 1)
            features['recent_form_trend'] = float(np.average(positions, weights=weights))
        else:
            features['recent_form_trend'] = self.config.default_avg_position
        
        return features
    
    def _extract_consistency_features(
        self,
        history: List[RaceHistoryRecord]
    ) -> Dict[str, float]:
        """Extract consistency score features."""
        features = {}
        
        # Consistency score (inverse coefficient of variation)
        last_10 = history[-10:] if len(history) >= 10 else history
        positions = [r.position for r in last_10 if r.position < 99]
        
        if len(positions) > 1:
            mean_pos = np.mean(positions)
            std_pos = np.std(positions)
            cv = std_pos / mean_pos if mean_pos > 0 else 0
            features['consistency_score'] = 1 - min(cv, 1.0)  # Cap at 1
        else:
            features['consistency_score'] = 0.0
        
        return features
    
    def _calculate_win_streak(self, history: List[RaceHistoryRecord]) -> int:
        """Calculate current winning streak."""
        streak = 0
        for record in reversed(history):
            if record.won:
                streak += 1
            else:
                break
        return streak


class JockeyTrainerFeatureExtractor:
    """Extracts jockey and trainer features with extended windows."""
    
    def __init__(self, config: HistoryConfig):
        self.config = config
    
    def extract_jockey_features(
        self,
        jockey_history: Deque[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract jockey form features for multiple time windows."""
        history_list = list(jockey_history)
        current_date = current_row['date_of_race']
        
        features = {}
        
        # Multiple time windows
        for days in [14, 30, 90]:
            window_races = [
                r for r in history_list
                if (current_date - r.date_of_race).days <= days
            ]
            features[f'jockey_form_wins_{days}d'] = sum(r.won for r in window_races)
            features[f'jockey_rides_{days}d'] = len(window_races)
            features[f'jockey_places_{days}d'] = sum(r.placed for r in window_races)
        
        return features
    
    def extract_trainer_features(
        self,
        trainer_history: Deque[RaceHistoryRecord],
        current_row: pd.Series
    ) -> Dict[str, float]:
        """Extract trainer form features for multiple time windows."""
        history_list = list(trainer_history)
        current_date = current_row['date_of_race']
        
        features = {}
        
        # Multiple time windows
        for days in [14, 30, 90]:
            window_races = [
                r for r in history_list
                if (current_date - r.date_of_race).days <= days
            ]
            features[f'trainer_form_wins_{days}d'] = sum(r.won for r in window_races)
            features[f'trainer_runners_{days}d'] = len(window_races)
            features[f'trainer_places_{days}d'] = sum(r.placed for r in window_races)
        
        return features
    
    def extract_combination_features(
        self,
        horse_history: Deque[RaceHistoryRecord],
        current_jockey: str,
        current_trainer: str
    ) -> Dict[str, float]:
        """Extract jockey-horse and trainer-horse combination features."""
        history_list = list(horse_history)
        
        # Jockey-horse combo
        jockey_combo_races = [r for r in history_list if r.jockey == current_jockey]
        
        # Trainer-horse combo
        trainer_combo_races = [r for r in history_list if r.trainer == current_trainer]
        
        return {
            'jockey_horse_combo_runs': len(jockey_combo_races),
            'jockey_horse_combo_wins': sum(r.won for r in jockey_combo_races),
            'jockey_horse_combo_places': sum(r.placed for r in jockey_combo_races),
            'trainer_horse_combo_runs': len(trainer_combo_races),
            'trainer_horse_combo_wins': sum(r.won for r in trainer_combo_races),
            'trainer_horse_combo_places': sum(r.placed for r in trainer_combo_races),
        }


class FieldQualityExtractor:
    """Extracts race field quality features."""
    
    @staticmethod
    def extract_features(race_df: pd.DataFrame, runner_idx: int) -> Dict[str, float]:
        """Extract field quality features for a specific runner."""
        current_runner = race_df.iloc[runner_idx]
        num_runners = len(race_df)
        
        # Field statistics excluding current runner
        other_runners = race_df.drop(race_df.index[runner_idx])
        
        features = {
            'field_quality_avg_or': float(other_runners['official_rating'].mean()),
            'field_quality_median_or': float(other_runners['official_rating'].median()),
            'field_quality_std_or': float(race_df['official_rating'].std()),
            'field_quality_range': float(
                race_df['official_rating'].max() - race_df['official_rating'].min()
            ),
            'weight_vs_field_avg': float(
                current_runner['weight_lbs'] - other_runners['weight_lbs'].mean()
            ),
            'weight_vs_field_median': float(
                current_runner['weight_lbs'] - other_runners['weight_lbs'].median()
            ),
        }
        
        return features


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

class ChronologicalFeatureEngineer:
    """Main class for chronological feature engineering with temporal guarantees."""
    
    def __init__(
        self,
        elo_config: Optional[EloConfig] = None,
        history_config: Optional[HistoryConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        existing_state: Optional[Dict[str, Any]] = None
    ):
        self.elo_config = elo_config or EloConfig()
        self.history_config = history_config or HistoryConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        self.state_manager = StateManager(
            self.elo_config,
            self.history_config,
            existing_state
        )
        self.elo_calculator = EloCalculator(self.elo_config)
        self.horse_extractor = HorseFeatureExtractor(self.feature_config)
        self.jt_extractor = JockeyTrainerFeatureExtractor(self.history_config)
        self.field_extractor = FieldQualityExtractor()
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main feature engineering pipeline with temporal guarantees."""
        logger.info("\n" + "="*80)
        logger.info("ENHANCED CHRONOLOGICAL FEATURE ENGINEERING")
        logger.info("="*80)
        
        # Validate and prepare data
        df_prepared = self._prepare_data(df)
        
        # Sort chronologically (CRITICAL)
        df_sorted = df_prepared.sort_values(
            by=['race_datetime', 'race_id']
        ).reset_index(drop=True)
        
        logger.info(f"Processing {len(df_sorted):,} rows across {df_sorted['race_id'].nunique():,} races")
        
        # Process races chronologically
        feature_list = []
        race_groups = df_sorted.groupby('race_id', sort=False)
        
        for race_id, race_df in tqdm(race_groups, desc="Building Features"):
            race_features = self._process_single_race(race_df)
            feature_list.extend(race_features)
        
        # Merge features back
        features_df = pd.DataFrame(feature_list).set_index('index')
        df_with_features = df_sorted.join(features_df)
        
        # Export final state
        final_state = self.state_manager.export_state()
        
        self._log_statistics(df_with_features, final_state)
        
        return df_with_features, final_state
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate input data."""
        df_prep = df.copy()
        
        # Ensure datetime columns
        if df_prep['date_of_race'].dtype != 'datetime64[ns]':
            df_prep['date_of_race'] = pd.to_datetime(df_prep['date_of_race'])
        
        if 'race_datetime' not in df_prep.columns:
            logger.warning("race_datetime not found, using date_of_race as proxy")
            df_prep['race_datetime'] = df_prep['date_of_race']
        elif df_prep['race_datetime'].dtype != 'datetime64[ns]':
            df_prep['race_datetime'] = pd.to_datetime(df_prep['race_datetime'])
        
        # Create betting deadline (info cutoff time)
        df_prep['betting_deadline'] = df_prep['race_datetime'] - pd.Timedelta(
            minutes=self.feature_config.betting_deadline_offset_minutes
        )
        
        # Ensure required columns
        required_cols = ['race_id', 'horse', 'jockey', 'trainer', 'won']
        missing = [col for col in required_cols if col not in df_prep.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create placed column if needed
        if 'placed' not in df_prep.columns:
            df_prep['num_places'] = df_prep.groupby('race_id')['horse'].transform('count').apply(
                lambda x: 3 if x >= 8 else (2 if x >= 5 else 0)
            )
            df_prep['placed'] = (
                (df_prep['won'] == 1) |
                ((df_prep.get('pos', 99) <= df_prep['num_places']) & (df_prep.get('pos', 99) < 99))
            ).astype(int)
        
        # Fill missing values with sensible defaults
        df_prep['official_rating'] = df_prep['official_rating'].fillna(60)
        df_prep['weight_lbs'] = df_prep['weight_lbs'].fillna(140)
        df_prep['distance_m'] = df_prep['distance_m'].fillna(2000)
        df_prep['going'] = df_prep['going'].fillna('Good')
        df_prep['headgear'] = df_prep['headgear'].fillna('None')
        df_prep['course'] = df_prep['course'].fillna('Unknown')
        
        return df_prep
    
    def _process_single_race(self, race_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a single race and extract features for all runners."""
        if len(race_df) < 2:
            return []
        
        # Get betting deadline for this race
        betting_deadline = race_df['betting_deadline'].iloc[0]
        
        # Pre-race: Extract features using only historical data
        race_features = []
        pre_race_elos = {}
        
        for i, (idx, row) in enumerate(race_df.iterrows()):
            # Validate entities
            if pd.isna(row['horse']) or pd.isna(row['jockey']) or pd.isna(row['trainer']):
                continue
            
            # Get point-in-time states (only data BEFORE betting deadline)
            horse_state = self.state_manager.get_horse_state_at_time(
                row['horse'],
                betting_deadline
            )
            jockey_state = self.state_manager.get_jockey_state_at_time(
                row['jockey'],
                betting_deadline
            )
            trainer_state = self.state_manager.get_trainer_state_at_time(
                row['trainer'],
                betting_deadline
            )
            
            # Store pre-race ELO
            pre_race_elos[row['horse']] = horse_state.elo_rating
            
            # Extract features
            features = {'index': idx}
            
            # ELO features
            features['horse_elo_pre_race'] = horse_state.elo_rating
            
            # Horse historical features
            horse_features = self.horse_extractor.extract_features(
                horse_state.history,
                row
            )
            features.update(horse_features)
            
            # Jockey/Trainer features
            jockey_features = self.jt_extractor.extract_jockey_features(
                jockey_state.history,
                row
            )
            features.update(jockey_features)
            
            trainer_features = self.jt_extractor.extract_trainer_features(
                trainer_state.history,
                row
            )
            features.update(trainer_features)
            
            # Combination features
            combo_features = self.jt_extractor.extract_combination_features(
                horse_state.history,
                row['jockey'],
                row['trainer']
            )
            features.update(combo_features)
            
            race_features.append(features)
        
        # Add field-level features (calculated from pre-race data only)
        for i, feature_dict in enumerate(race_features):
            field_features = self.field_extractor.extract_features(race_df, i)
            feature_dict.update(field_features)
        
        # Calculate ELO-implied probabilities
        elo_ratings = [pre_race_elos.get(row['horse'], self.elo_config.default_rating) 
                      for _, row in race_df.iterrows()]
        elo_probs = self._calculate_elo_probabilities(elo_ratings)
        
        for i, feature_dict in enumerate(race_features):
            feature_dict['elo_implied_prob'] = elo_probs[i]
        
        # Post-race: Update states (AFTER feature extraction)
        new_elos = self.elo_calculator.update_race_ratings(race_df, pre_race_elos)
        self.state_manager.update_after_race(race_df, new_elos)
        
        return race_features
    
    def _calculate_elo_probabilities(self, ratings: List[float]) -> List[float]:
        """Calculate probability of winning based on ELO ratings."""
        if not ratings:
            return []
        
        strengths = np.array([10 ** (r / self.elo_config.base) for r in ratings])
        total_strength = strengths.sum()
        
        if total_strength == 0:
            return [1.0 / len(ratings)] * len(ratings)
        
        return list(strengths / total_strength)
    
    def _log_statistics(self, df: pd.DataFrame, state: Dict[str, Any]) -> None:
        """Log summary statistics."""
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*80)
        
        # ELO statistics
        horse_elos = [s.elo_rating for s in self.state_manager.horse_states.values()]
        if horse_elos:
            logger.info(f"\nELO Statistics:")
            logger.info(f"  Mean: {np.mean(horse_elos):.1f}")
            logger.info(f"  Std: {np.std(horse_elos):.1f}")
            logger.info(f"  Min: {np.min(horse_elos):.1f}")
            logger.info(f"  Max: {np.max(horse_elos):.1f}")
        
        # State statistics
        metadata = state.get('metadata', {})
        logger.info(f"\nState Tracking:")
        logger.info(f"  Horses: {metadata.get('num_horses', 0):,}")
        logger.info(f"  Jockeys: {metadata.get('num_jockeys', 0):,}")
        logger.info(f"  Trainers: {metadata.get('num_trainers', 0):,}")
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col.startswith(
            ('horse_', 'elo_', 'win_', 'place_', 'avg_', 'or_', 'jockey_', 'trainer_', 'field_',
             'surface_', 'distance_', 'course_', 'top3_', 'median_', 'races_since_', 'runs_last_',
             'career_', 'position_', 'consistency_', 'recent_form_')
        )]
        logger.info(f"\nFeatures Created: {len(feature_cols)}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_chronological_features(
    df: pd.DataFrame,
    state_trackers: Optional[Dict[str, Any]] = None,
    elo_config: Optional[EloConfig] = None,
    history_config: Optional[HistoryConfig] = None,
    feature_config: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to create enhanced chronological features.
    
    Args:
        df: Input dataframe
        state_trackers: Optional existing state (for incremental updates)
        elo_config: Optional ELO configuration
        history_config: Optional history tracking configuration
        feature_config: Optional feature engineering configuration
        
    Returns:
        Tuple of (feature-engineered dataframe, state trackers)
    """
    engineer = ChronologicalFeatureEngineer(
        elo_config=elo_config,
        history_config=history_config,
        feature_config=feature_config,
        existing_state=state_trackers
    )
    
    return engineer.engineer_features(df)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_temporal_integrity(df: pd.DataFrame) -> None:
    """Validate that there is no temporal leakage in the dataset."""
    logger.info("\nValidating temporal integrity...")
    
    if not df['race_datetime'].is_monotonic_increasing:
        violations = (df['race_datetime'].diff() < pd.Timedelta(0)).sum()
        raise ValueError(
            f"Data not in chronological order! Found {violations} violations. "
            "Data must be sorted by race_datetime before feature engineering."
        )
    
    logger.info("✅ Temporal integrity validated")


def detect_leakage_features(df: pd.DataFrame) -> List[str]:
    """Detect potentially leaky features in the dataframe."""
    leakage_keywords = [
        'winning_distance', 'sp_win_return', 'ew_return', 'betfair_win_return',
        'place_return', 'betfair_lay_return', 'ip_min', 'ip_max', 'final',
        'result', 'actual', 'outcome', 'betfair_sp', 'industry_sp'
    ]
    
    suspicious = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in leakage_keywords):
            suspicious.append(col)
    
    if suspicious:
        logger.warning(f"Found {len(suspicious)} potentially leaky columns: {suspicious}")
    
    return suspicious