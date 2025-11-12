# src/data_processing.py

import logging
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use __file__ to dynamically find the project root, making the script portable
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Fallback for interactive environments like Jupyter or IPython
    PROJECT_ROOT = Path(".").resolve().parents[0]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_CSV_FILES = [
    RAW_DATA_DIR / "horse_race_010120_310525.csv",
    RAW_DATA_DIR / "horse_race_010625_310825.csv",
]
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_race_data.parquet"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_and_combine_raw_data(file_list: list[Path]) -> pd.DataFrame:
    """Loads and combines multiple raw CSV files into a single DataFrame."""
    dataframes = []
    for filepath in file_list:
        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue
        logger.info(f"Loading data from: {filepath.name}...")
        df = pd.read_csv(filepath, low_memory=False)
        dataframes.append(df)
    
    if not dataframes:
        raise FileNotFoundError("No valid data files found to process.")
        
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Successfully loaded and combined {len(dataframes)} file(s). Total raw records: {len(combined_df):,}")
    return combined_df


# =============================================================================
# COLUMN STANDARDIZATION FUNCTIONS
# =============================================================================

def log_available_columns(df: pd.DataFrame, stage: str = "current") -> None:
    """Logs all available columns at a given stage for debugging."""
    logger.info(f"\nAvailable columns at {stage} stage ({len(df.columns)} total):")
    
    # Group columns by common prefixes for easier reading
    col_groups = {}
    for col in sorted(df.columns):
        prefix = col.split('_')[0] if '_' in col else col[:3]
        if prefix not in col_groups:
            col_groups[prefix] = []
        col_groups[prefix].append(col)
    
    for prefix, cols in sorted(col_groups.items()):
        if len(cols) <= 5:
            logger.info(f"  {prefix}: {', '.join(cols)}")
        else:
            logger.info(f"  {prefix}: {', '.join(cols[:5])} ... ({len(cols)} total)")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all column names to a consistent snake_case format."""
    logger.info("Standardizing column names to snake_case...")
    
    def to_snake_case(name):
        name = re.sub(r'[^a-zA-Z0-9\s%]', '', name).strip()
        name = name.replace('%', 'pct').replace(' ', '_')
        return name.lower()

    df.columns = [to_snake_case(col) for col in df.columns]
    logger.info("Column names standardized.")
    return df


# =============================================================================
# DUPLICATE HANDLING FUNCTIONS
# =============================================================================

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicate records that may exist across CSV files.
    """
    initial_count = len(df)
    
    # Define what constitutes a duplicate (adjust based on your data)
    # Example: Same horse, same race datetime
    if 'horse' in df.columns and 'date_of_race' in df.columns and 'time' in df.columns:
        duplicate_mask = df.duplicated(subset=['date_of_race', 'time', 'track', 'horse'], keep='first')
        df = df[~duplicate_mask].copy()
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate records ({removed/initial_count*100:.2f}%)")
        else:
            logger.info("No duplicates found.")
    else:
        logger.warning("Cannot check for duplicates - required columns not found")
    
    return df


# =============================================================================
# RACE CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_race_type(race_type: str) -> str:
    """
    Classifies a raw race type string into 'Jumps', 'Flat', or 'Unknown'.
    
    Returns:
        str: One of 'Jumps', 'Flat', or 'Unknown'
    """
    # Handle missing/null values explicitly
    if pd.isna(race_type) or not isinstance(race_type, str):
        return "Unknown"
    
    t = race_type.strip().lower()
    
    if not t:  # Empty string after stripping
        return "Unknown"
    
    # Jumps: Order matters - check more specific patterns first
    jumps_keywords = ['hurdle', 'chase', 'nh flat', 'hunter', 'bumper']
    if any(k in t for k in jumps_keywords):
        return "Jumps"
    
    # Flat: Comprehensive list
    flat_keywords = [
        'stakes', 'handicap', 'maiden', 'group', 'listed', 'nursery',
        'conditions', 'classified', 'claiming', 'selling', 'auction',
        'fillies', 'colts', 'restricted'
    ]
    if any(k in t for k in flat_keywords):
        return "Flat"
    
    # Log unknown types for investigation
    return "Unknown"


def analyze_race_types(df: pd.DataFrame) -> None:
    """Analyzes and logs race type classification results."""
    if 'race_category' not in df.columns:
        return
    
    logger.info("\nRace Category Distribution:")
    distribution = df['race_category'].value_counts(normalize=True) * 100
    for category, pct in distribution.items():
        logger.info(f"  {category}: {pct:.2f}%")
    
    # Sample unknown types for investigation
    unknown_mask = df['race_category'] == 'Unknown'
    if unknown_mask.sum() > 0 and 'type' in df.columns:
        logger.info(f"\nSample of 'Unknown' race types for review:")
        unknown_samples = df.loc[unknown_mask, 'type'].value_counts().head(10)
        for race_type, count in unknown_samples.items():
            logger.info(f"  '{race_type}': {count} occurrences")


# =============================================================================
# RACE ID CREATION FUNCTIONS
# =============================================================================

def create_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unique race identifier using only pre-race information.
    
    WARNING: Assumes 'time' column contains SCHEDULED race time known
    before the race starts. Verify this in your data documentation.
    """
    # Add validation that required columns exist
    required_cols = ['date_of_race', 'track', 'time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for race_id generation: {missing_cols}")
    
    logger.info("Creating race_id from pre-race scheduled information...")
    
    df['race_id'] = (
        df['date_of_race'].dt.strftime('%Y-%m-%d') + '_' +
        df['track'].astype(str).str.strip().str.upper() + '_' +  # Normalize track names
        df['time'].astype(str).str.strip()
    )
    
    # Check for duplicates (multiple races with identical identifiers)
    duplicate_count = df['race_id'].duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate race_ids. "
                      f"Consider adding race number or sequence to race_id.")
    
    return df


# =============================================================================
# MAIN DATA PROCESSING FUNCTION
# =============================================================================

def process_foundational_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs foundational processing: creates datetime, race_id, race_category,
    creates target variable 'won', cleans odds data, and sorts the data.
    """
    logger.info("Performing foundational data processing...")
    
    # --- 1. Create primary datetime column ---
    logger.info("Creating race_datetime column...")
    df['date_of_race'] = pd.to_datetime(df['date_of_race'], errors='coerce')
    
    # Drop rows with invalid dates or missing time
    initial_count = len(df)
    df.dropna(subset=['date_of_race', 'time'], inplace=True)
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing date_of_race or time")
    
    df['race_datetime'] = pd.to_datetime(
        df['date_of_race'].dt.date.astype(str) + ' ' + df['time'].astype(str),
        errors='coerce'
    )
    
    # Drop rows where race_datetime couldn't be created
    initial_count = len(df)
    df.dropna(subset=['race_datetime'], inplace=True)
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid race_datetime")
    
    # --- 2. Create unique race identifier ---
    df = create_race_id(df)
    
    # --- 3. Engineer a clean race_category ---
    if 'type' in df.columns:
        logger.info("Engineering 'race_category' from raw 'type' column...")
        df['race_category'] = df['type'].apply(classify_race_type)
        analyze_race_types(df)
    else:
        logger.warning("Raw 'type' column not found. Skipping 'race_category' creation.")

    # --- 4. Create 'pos' and 'won' target variables ---
    # We need to determine finishing position and whether the horse won
    logger.info("Creating 'pos' and 'won' target variables...")
    
    # Check what columns are available for determining winners
    possible_pos_cols = ['pos', 'position', 'finish_position', 'final_position', 'place']  # Added 'place'
    possible_win_cols = ['won', 'win', 'winner', 'is_winner']
    
    # Also check for conditional winner columns that might help
    conditional_winner_cols = ['course_winner', 'distance_winner', 'going_winner']
    
    pos_col = None
    for col in possible_pos_cols:
        if col in df.columns:
            pos_col = col
            logger.info(f"  Found position column: '{col}'")
            break
    
    # If we don't have a pos column, try to create it from other indicators
    if pos_col is None:
        # Check if we have a direct winner indicator
        win_col = None
        for col in possible_win_cols:
            if col in df.columns:
                win_col = col
                logger.info(f"  Found winner indicator column: '{col}'")
                break
        
        if win_col:
            # Create pos from winner indicator
            logger.info(f"  Creating 'pos' from '{win_col}' column...")
            df['pos'] = df[win_col].apply(lambda x: 1 if x in [1, '1', True, 'True', 'true', 'Y', 'yes', 'Yes'] else 99)
            df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
            pos_col = 'pos'
        else:
            # Check if we have conditional winner columns as last resort
            available_winner_cols = [col for col in conditional_winner_cols if col in df.columns]
            
            if available_winner_cols:
                # Use the most general winner indicator (prefer in order: course, distance, going)
                # Since these indicate "did this horse win at this course/distance/going before"
                # Not "did this horse win THIS race", we can't use them directly
                logger.warning(f"⚠️  Found conditional winner columns: {available_winner_cols}")
                logger.warning("  These indicate past performance, not current race results.")
                logger.warning("  Cannot use these to create 'won' target variable.")
            
            # Last resort: look for columns that might indicate results
            result_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['result', 'finish', 'outcome'])]
            if result_cols:
                logger.warning(f"⚠️  No standard position column found. Found these result-related columns: {result_cols}")
                logger.warning("  Please examine your data and update the code to extract position from the appropriate column.")
            
            raise ValueError(
                "Cannot create 'pos' and 'won' target variables: No position or winner indicator column found in data.\n"
                f"Available columns: {list(df.columns[:20])}...\n"
                "Expected one of: {pos_cols} or {win_cols}".format(
                    pos_cols=possible_pos_cols,
                    win_cols=possible_win_cols
                )
            )
    
    # Ensure we have a 'pos' column with the standardized name
    if pos_col != 'pos':
        df['pos'] = df[pos_col]
    
    # Convert pos to numeric, treating non-numeric values as NaN
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    
    # Create binary 'won' column: 1 if position is 1, else 0
    df['won'] = (df['pos'] == 1).astype(int)
    
    # Log distribution
    total_races = len(df)
    total_winners = df['won'].sum()
    total_valid_pos = df['pos'].notna().sum()
    win_rate = (total_winners / total_races * 100) if total_races > 0 else 0
    
    logger.info(f"  Created 'pos' and 'won' target variables:")
    logger.info(f"    Total entries: {total_races:,}")
    logger.info(f"    Valid positions: {total_valid_pos:,} ({total_valid_pos/total_races*100:.1f}%)")
    logger.info(f"    Winners (won=1): {total_winners:,} ({win_rate:.2f}%)")
    logger.info(f"    Non-winners (won=0): {total_races - total_winners:,} ({100-win_rate:.2f}%)")
    
    # Validate that we have winners
    if total_winners == 0:
        logger.warning("⚠️  WARNING: No winners found in data! Check 'pos' column values.")
    
    # Sanity check: win rate should be roughly 1/avg_field_size
    if 'runners' in df.columns:
        avg_field_size = df['runners'].mean()
        expected_win_rate = (1 / avg_field_size * 100) if avg_field_size > 0 else 0
        logger.info(f"    Expected win rate (1/avg_field_size): {expected_win_rate:.2f}%")
        
        if abs(win_rate - expected_win_rate) > 2:  # More than 2% difference
            logger.warning(f"⚠️  Win rate differs from expected by {abs(win_rate - expected_win_rate):.2f}%")
        else:
            logger.info(f"    ✓ Win rate is consistent with field sizes")

    # --- 5. Clean and convert 'forecasted_odds' ---
    if 'forecasted_odds' in df.columns:
        logger.info("Cleaning 'forecasted_odds' column...")
        
        # Replace non-numeric strings with pd.NA
        df['forecasted_odds'] = df['forecasted_odds'].replace(
            ['', 'SCRATCHED', 'N/A', '-', 'nan', 'NaN'], 
            pd.NA
        )

        # Coerce remaining non-numeric values to NaN and convert to nullable float
        df['forecasted_odds'] = pd.to_numeric(
            df['forecasted_odds'],
            errors='coerce'
        ).astype('Float64')
        
        # Log how many were cleaned
        null_count = df['forecasted_odds'].isnull().sum()
        if null_count > 0:
            logger.info(f"  Cleaned 'forecasted_odds': {null_count:,} invalid values set to NaN")
    else:
        logger.warning("'forecasted_odds' column not found, skipping odds cleaning.")

    # --- 6. Sort by time to ensure correct chronological order ---
    logger.info("Sorting data chronologically...")
    df = df.sort_values(['date_of_race', 'race_datetime', 'race_id']).reset_index(drop=True)
    
    logger.info("Foundational processing complete. Data is sorted chronologically.")
    return df


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_processed_data(df: pd.DataFrame) -> None:
    """
    Validates critical data quality requirements before saving.
    Raises ValueError if validation fails.
    """
    logger.info("Running data quality validation...")
    
    # 1. Check for null critical columns
    critical_cols = ['race_datetime', 'race_id', 'track', 'won']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(f"Critical column '{col}' has {null_count} null values")
    
    # 2. Check date range is reasonable
    if 'race_datetime' in df.columns:
        min_date = df['race_datetime'].min()
        max_date = df['race_datetime'].max()
        
        logger.info(f"Date range: {min_date} to {max_date}")
        
        if min_date < pd.Timestamp('2000-01-01'):
            logger.warning(f"Suspiciously early date found: {min_date}")
        if max_date > pd.Timestamp.now() + pd.Timedelta(days=365):
            logger.warning(f"Future date beyond 1 year found: {max_date}")
    
    # 3. Check for data monotonicity (should be sorted)
    if 'race_datetime' in df.columns:
        if not df['race_datetime'].is_monotonic_increasing:
            raise ValueError("Data is not properly sorted by race_datetime")
    
    # 4. Validate race_id uniqueness per horse per race
    if 'race_id' in df.columns and 'horse' in df.columns:
        duplicates = df.groupby(['race_id', 'horse']).size()
        if (duplicates > 1).any():
            dup_count = (duplicates > 1).sum()
            logger.warning(f"Found {dup_count} race_id+horse duplicates - same horse in same race multiple times")
    
    # 5. Validate 'won' target variable
    if 'won' in df.columns:
        unique_values = df['won'].unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"'won' column contains invalid values: {unique_values}")
        
        if df['won'].sum() == 0:
            raise ValueError("'won' column has no winners (all zeros)")
    
    logger.info("✅ Data quality validation passed")


def generate_data_profile(df: pd.DataFrame) -> dict:
    """Generates a comprehensive data profile for logging and validation."""
    profile = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'date_range': (df['race_datetime'].min(), df['race_datetime'].max()) if 'race_datetime' in df.columns else None,
        'unique_tracks': df['track'].nunique() if 'track' in df.columns else None,
        'unique_races': df['race_id'].nunique() if 'race_id' in df.columns else None,
        'unique_horses': df['horse'].nunique() if 'horse' in df.columns else None,
        'total_winners': df['won'].sum() if 'won' in df.columns else None,
        'win_rate_pct': (df['won'].mean() * 100) if 'won' in df.columns else None,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'race_category_distribution': df['race_category'].value_counts().to_dict() if 'race_category' in df.columns else None,
    }
    
    logger.info("\n" + "="*80)
    logger.info("DATA PROFILE SUMMARY")
    logger.info("="*80)
    for key, value in profile.items():
        if key == 'date_range' and value:
            logger.info(f"{key}: {value[0]} to {value[1]}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("="*80 + "\n")
    
    return profile


def document_schema(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generates and saves schema documentation."""
    logger.info("Generating schema documentation...")
    
    schema_doc = {
        'column': df.columns.tolist(),
        'dtype': df.dtypes.astype(str).tolist(),
        'null_count': df.isnull().sum().tolist(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2).tolist(),
        'unique_count': [df[col].nunique() for col in df.columns],
    }
    
    schema_df = pd.DataFrame(schema_doc)
    schema_path = output_path.parent / "schema_documentation.csv"
    schema_df.to_csv(schema_path, index=False)
    logger.info(f"Schema documentation saved: {schema_path}")
    
    return schema_df


# =============================================================================
# FILE SAVING FUNCTIONS
# =============================================================================

def save_with_versioning(df: pd.DataFrame, base_path: Path) -> Path:
    """Saves parquet with timestamp backup for version control."""
    logger.info(f"Preparing to save output to {base_path.parent}...")
    
    # Ensure output directory exists
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main file
    df.to_parquet(base_path, index=False)
    logger.info(f"Saved primary output: {base_path}")
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
    df.to_parquet(backup_path, index=False)
    logger.info(f"Created backup: {backup_path.name}")
    
    return base_path


# =============================================================================
# MAIN PIPELINE EXECUTION FUNCTION
# =============================================================================

def run_data_pipeline():
    """Executes the full data processing pipeline with robust error handling."""
    try:
        # Step 1: Load
        logger.info("\n" + "="*80)
        logger.info("STEP 1/5: Loading raw data...")
        logger.info("="*80)
        raw_df = load_and_combine_raw_data(RAW_CSV_FILES)
        logger.info(f"Loaded {len(raw_df):,} raw records")
        
        # Step 2: Standardize
        logger.info("\n" + "="*80)
        logger.info("STEP 2/5: Standardizing columns...")
        logger.info("="*80)
        standardized_df = standardize_column_names(raw_df)
        
        # Log available columns for debugging
        log_available_columns(standardized_df, "after standardization")
        
        # Step 3: Handle Duplicates
        logger.info("\n" + "="*80)
        logger.info("STEP 3/5: Handling duplicates...")
        logger.info("="*80)
        deduplicated_df = handle_duplicates(standardized_df)
        
        # Step 4: Process
        logger.info("\n" + "="*80)
        logger.info("STEP 4/5: Processing foundational features...")
        logger.info("="*80)
        processed_df = process_foundational_data(deduplicated_df)
        
        # Step 5: Validate and Save
        logger.info("\n" + "="*80)
        logger.info("STEP 5/5: Validating and saving...")
        logger.info("="*80)
        validate_processed_data(processed_df)
        
        # Generate data profile
        profile = generate_data_profile(processed_df)
        
        # Save with versioning
        output_path = save_with_versioning(processed_df, PROCESSED_PARQUET_PATH)
        
        # Document schema
        schema = document_schema(processed_df, PROCESSED_PARQUET_PATH)
        
        logger.info("\n" + "="*80)
        logger.info("✅ DATA PROCESSING PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Output: {output_path}")
        logger.info(f"Final shape: {processed_df.shape}")
        if 'race_datetime' in processed_df.columns:
            logger.info(f"Date range: {processed_df['race_datetime'].min()} to {processed_df['race_datetime'].max()}")
        logger.info("="*80 + "\n")
        
        return processed_df, schema
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    except KeyError as e:
        logger.error(f"❌ Missing expected column: {e}")
        if 'raw_df' in locals():
            logger.error("Available columns: " + ", ".join(raw_df.columns.tolist()[:20]) + "...")
        raise
    except ValueError as e:
        logger.error(f"❌ Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in pipeline: {type(e).__name__}: {e}")
        raise


# =============================================================================
# SCRIPT EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    """
    This block allows the script to be run directly from the command line,
    e.g., `python src/data_processing.py`.
    """
    logger.info("="*80)
    logger.info("RUNNING DATA PROCESSING PIPELINE")
    logger.info("="*80)
    
    try:
        processed_df, schema = run_data_pipeline()
        
        logger.info("\n" + "="*80)
        logger.info("✅ SCRIPT FINISHED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Processed {len(processed_df):,} records")
        logger.info(f"Created {len(processed_df.columns)} columns")
        
        # Display sample
        logger.info("\nSample of processed data (first 5 rows):")
        print(processed_df.head())
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("❌ SCRIPT EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        raise