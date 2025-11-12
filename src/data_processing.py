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

# --- All your data processing and validation functions go here ---
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


def process_foundational_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs foundational processing: creates datetime, race_id, race_category,
    cleans odds data, and sorts the data.
    """
    logger.info("Performing foundational data processing...")
    
    # --- 1. Create primary datetime column ---
    # Convert date and combine with time to create a full datetime object.
    df['date_of_race'] = pd.to_datetime(df['date_of_race'], errors='coerce')
    df.dropna(subset=['date_of_race', 'time'], inplace=True)
    
    df['race_datetime'] = pd.to_datetime(
        df['date_of_race'].dt.date.astype(str) + ' ' + df['time'].astype(str),
        errors='coerce'
    )
    df.dropna(subset=['race_datetime'], inplace=True)
    
    # --- 2. Create unique race identifier ---
    df = create_race_id(df) # Assumes this function is defined
    
    # --- 3. Engineer a clean race_category ---
    if 'type' in df.columns:
        df['race_category'] = df['type'].apply(classify_race_type) # Assumes this function is defined
        analyze_race_types(df) # Assumes this function is defined
    else:
        logger.warning("Raw 'type' column not found.")

    # --- 4. Clean and convert 'forecasted_odds' ---
    # Replace non-numeric strings with pd.NA.
    df['forecasted_odds'] = df['forecasted_odds'].replace(
        ['', 'SCRATCHED', 'N/A', '-'], 
        pd.NA
    )

    # Coerce remaining non-numeric values to NaN and convert to nullable float.
    df['forecasted_odds'] = pd.to_numeric(
        df['forecasted_odds'],
        errors='coerce'
    ).astype('Float64') 

    # --- 5. Sort by time ---
    df = df.sort_values('race_datetime').reset_index(drop=True)
    
    logger.info("Foundational processing complete. Data is sorted.")
    return df
# =============================================================================
# DATA VALIDATION & DOCUMENTATION FUNCTIONS (INCLUDED NOW)
# =============================================================================

def validate_processed_data(df: pd.DataFrame) -> None:
    """Validates critical data quality requirements before saving."""
    logger.info("Running data quality validation...")
    critical_cols = ['race_datetime', 'race_id', 'track']
    for col in critical_cols:
        if col in df.columns and df[col].isnull().any():
            raise ValueError(f"Critical column '{col}' has null values")
    if 'race_datetime' in df.columns:
        if not df['race_datetime'].is_monotonic_increasing:
            raise ValueError("Data is not properly sorted by race_datetime")
    logger.info("✅ Data quality validation passed")

def generate_data_profile(df: pd.DataFrame) -> dict:
    """Generates a comprehensive data profile for logging."""
    profile = {
        'total_records': len(df), 'total_columns': len(df.columns),
        'date_range': (df['race_datetime'].min(), df['race_datetime'].max()) if 'race_datetime' in df.columns else None,
        'unique_races': df['race_id'].nunique() if 'race_id' in df.columns else None,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    logger.info("\n" + "="*80 + "\nDATA PROFILE SUMMARY\n" + "="*80)
    for key, value in profile.items():
        if key == 'date_range' and value: logger.info(f"{key}: {value[0]} to {value[1]}")
        else: logger.info(f"{key}: {value}")
    logger.info("="*80 + "\n")
    return profile

def document_schema(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generates and saves schema documentation."""
    logger.info("Generating schema documentation...")
    schema_info = {
        'column': df.columns, 'dtype': df.dtypes.astype(str),
        'null_count': df.isnull().sum(), 'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
    }
    schema_df = pd.DataFrame(schema_info).reset_index(drop=True)
    schema_path = output_path.parent / "schema_documentation.csv"
    schema_df.to_csv(schema_path, index=False)
    logger.info(f"Schema documentation saved: {schema_path}")
    return schema_df

# =============================================================================
# FILE SAVING FUNCTIONS
# =============================================================================
def save_with_versioning(df: pd.DataFrame, base_path: Path) -> Path:
    """Saves a Parquet file with a timestamped backup for version control."""
    logger.info(f"Preparing to save output to {base_path.parent}...")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the main file
    df.to_parquet(base_path, index=False)
    logger.info(f"Saved primary output: {base_path}")
    
    # Create a timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
    df.to_parquet(backup_path, index=False)
    logger.info(f"Created versioned backup: {backup_path.name}")
    
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
        logger.info("✅ NOTEBOOK 01 COMPLETE")
        logger.info("="*80)
        logger.info(f"Output: {output_path}")
        logger.info(f"Final shape: {processed_df.shape}")
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
    logger.info("Running data processing script as a standalone process...")
    try:
        run_data_pipeline()
        logger.info("Script finished successfully.")
    except Exception:
        logger.error("Script execution failed.")