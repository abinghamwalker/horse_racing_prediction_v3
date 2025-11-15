# src/get_predictions.py
"""
Production script to generate calibrated predictions for new horse racing data.

This script loads a pre-trained champion model pipeline (base model + calibrator),
processes new raw data through the full feature engineering pipeline, and outputs
a final DataFrame with calibrated probabilities.

Usage:
    python src/get_predictions.py \
        --champion_dir_path champion_folder/FLAT_champ_catboost_20251113_200545 \
        --new_data_path /data/get_bets/new_raw_data.csv \
        --output_path /predictions/predictions.csv
"""
import pandas as pd
import numpy as np
import pickle
import cloudpickle
import sys
from pathlib import Path
from collections import defaultdict, deque
import argparse # For command-line arguments

# --- Import from our own source files ---
# This assumes the script is run from the project root or 'src' is in the path
from data_processing import standardize_column_names, process_foundational_data
from feature_engineering import (
    create_chronological_features, create_derived_features,
    create_advanced_market_features, validate_data
)
from feature_selection import FeatureSelector

def run_prediction_pipeline(champion_dir: Path, new_data_file: Path) -> pd.DataFrame:
    """
    Executes the full end-to-end prediction pipeline.

    Args:
        champion_dir (Path): Path to the ring-fenced champion artifact directory.
        new_data_file (Path): Path to the new raw data CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the new data with a 'model_prob' column.
    """
    print("\n" + "="*80)
    print("🚀 STARTING PRODUCTION PREDICTION PIPELINE")
    print("="*80)

    # =========================================================================
    # 1. LOAD ARTIFACTS
    # =========================================================================
    print(f"\n--- [1/4] Loading artifacts from: {champion_dir} ---")
    base_model_path = champion_dir / "base_model.pkl"
    calibrator_path = champion_dir / "calibrator.pkl"
    state_trackers_path = champion_dir / "state_trackers.pkl"
    feature_selector_path = champion_dir / "feature_selector.pkl"

    with open(base_model_path, 'rb') as f: base_model = cloudpickle.load(f)
    with open(calibrator_path, 'rb') as f: calibrator = cloudpickle.load(f)
    with open(state_trackers_path, 'rb') as f: state_trackers = pickle.load(f)
    with open(feature_selector_path, 'rb') as f: feature_selector = pickle.load(f)

    # ------------- FIX IS HERE: UNABRIDGED RECONSTITUTION LOGIC -------------
    print("  -> Reconstituting state trackers...")
    # Define maxlen from your original script constants if available, otherwise use a default
    HORSE_HISTORY_LEN = 20
    JOCKEY_HISTORY_LEN = 50
    TRAINER_HISTORY_LEN = 100
    
    # Recreate defaultdicts with the correct factory function
    state_trackers['horse_elo'] = defaultdict(lambda: state_trackers['config']['ELO_DEFAULT_RATING'], state_trackers.get('horse_elo', {}))
    
    # Recreate deques with the correct maxlen for each history tracker
    horse_history_reconstituted = {k: deque(v, maxlen=HORSE_HISTORY_LEN) for k, v in state_trackers.get('horse_history', {}).items()}
    state_trackers['horse_history'] = defaultdict(lambda: deque(maxlen=HORSE_HISTORY_LEN), horse_history_reconstituted)

    jockey_history_reconstituted = {k: deque(v, maxlen=JOCKEY_HISTORY_LEN) for k, v in state_trackers.get('jockey_history', {}).items()}
    state_trackers['jockey_history'] = defaultdict(lambda: deque(maxlen=JOCKEY_HISTORY_LEN), jockey_history_reconstituted)

    trainer_history_reconstituted = {k: deque(v, maxlen=TRAINER_HISTORY_LEN) for k, v in state_trackers.get('trainer_history', {}).items()}
    state_trackers['trainer_history'] = defaultdict(lambda: deque(maxlen=TRAINER_HISTORY_LEN), trainer_history_reconstituted)
    # ----------------------------------------------------------------------
    print("✅ All artifacts loaded and reconstituted successfully.")

    # =========================================================================
    # 2. LOAD & PROCESS NEW DATA
    # =========================================================================
    print(f"\n--- [2/4] Loading and processing new raw data from: {new_data_file} ---")
    new_df_raw = pd.read_csv(new_data_file)
    
    # Define the cleaning function locally
    def run_cleaning(df_raw):
        df = df_raw.copy()
        if 'weight' in df.columns:
            def weight_to_lbs(w):
                if pd.isna(w): return np.nan
                try:
                    if isinstance(w, (int, float)): return float(w)
                    if '-' in str(w): s, p = str(w).split('-'); return int(s) * 14 + int(p)
                    return float(w)
                except: return np.nan
            df['weight_lbs'] = df['weight'].apply(weight_to_lbs)
            df.drop(columns=['weight'], inplace=True)
        return df

    df_standardized = standardize_column_names(new_df_raw)
    df_cleaned = run_cleaning(df_standardized)
    df_processed = process_foundational_data(df_cleaned)
    if 'distance_m' not in df_processed.columns: df_processed['distance_m'] = 2000
    if 'course' not in df_processed.columns and 'track' in df_processed.columns: df_processed['course'] = df_processed['track']
    print("✅ Initial data processing complete.")

    # =========================================================================
    # 3. EXECUTE FEATURE ENGINEERING & SELECTION
    # =========================================================================
    print("\n--- [3/4] Executing feature engineering and selection pipeline ---")
    df_chrono, _ = create_chronological_features(df_processed, state_trackers=state_trackers)
    df_derived = create_derived_features(df_chrono)
    df_engineered = create_advanced_market_features(df_derived)
    validate_data(df_engineered)
    df_final_features = feature_selector.transform(df_engineered)
    print("✅ Feature pipeline complete.")

    # =========================================================================
    # 4. GENERATE CALIBRATED PREDICTIONS
    # =========================================================================
    print("\n--- [4/4] Generating final calibrated predictions ---")
    feature_cols = feature_selector.get_feature_names_out()
    X_predict = df_final_features[feature_cols].copy()

    # Correct data types for CatBoost
    cat_feature_indices = base_model.get_cat_feature_indices()
    cat_feature_names = [X_predict.columns[i] for i in cat_feature_indices]
    for col in cat_feature_names:
        X_predict[col] = X_predict[col].fillna('missing').astype(str)
    
    numeric_feature_names = [col for col in X_predict.columns if col not in cat_feature_names]
    for col in numeric_feature_names:
        X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce')

    # Generate predictions
    raw_probs = base_model.predict_proba(X_predict)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)
    
    predictions_df = df_final_features.copy()
    predictions_df['model_prob'] = calibrated_probs
    print("✅ Prediction generation complete.")

    print("\n" + "="*80)
    print("✅ PRODUCTION PREDICTION PIPELINE FINISHED SUCCESSFULLY")
    print("="*80)

    return predictions_df

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(description="Generate predictions using a saved horse racing model pipeline.")
    parser.add_argument("--champion_dir_path", type=Path, required=True, help="Path to the champion artifact directory.")
    parser.add_argument("--new_data_path", type=Path, required=True, help="Path to the new raw data CSV file.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the final predictions CSV file.")
    args = parser.parse_args()

    try:
        # Before running the pipeline, ensure the 'src' directory is in the path
        # This is important when calling the script from any location.
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.append(str(script_dir))

        final_predictions = run_prediction_pipeline(args.champion_dir_path, args.new_data_path)
        
        # Ensure output directory exists
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the output
        final_predictions.to_csv(args.output_path, index=False)
        print(f"✅ Final predictions saved successfully to: {args.output_path}")

    except Exception as e:
        import traceback
        print(f"❌ PIPELINE FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()