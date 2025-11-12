# ===========================================================================

# NOTEBOOK 02: DATA CLEANING AND PREPROCESSING

# ===========================================================================

# 

# Purpose: Clean and preprocess the data from Notebook 01

# 

# Input:  ../data/processed/01_extracted_cleaned.parquet

# Output: ../data/processed/02_cleaned_imputed.parquet

#

# Steps:

# 1. Load cleaned data from notebook 01

# 2. Standardize column names to snake_case

# 3. Create race_id and helper features

# 4. Clean data types and handle special values

# 5. Impute missing values intelligently

# 6. Validate data quality

# 7. Save processed data

# ===========================================================================



import pandas as pd

import numpy as np

from pathlib import Path

import warnings

warnings.filterwarnings('ignore')



# Set display options

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)



print("="*80)

print("NOTEBOOK 02: DATA CLEANING AND PREPROCESSING")

print("="*80)



# Define paths

DATA_DIR = Path('../data/processed')

INPUT_FILE = DATA_DIR / "01_extracted_cleaned.parquet"



# Verify file exists

if not INPUT_FILE.exists():

    raise FileNotFoundError(

        f"Input file not found: {INPUT_FILE}\n"

        f"Please run Notebook 01 first to generate the cleaned data."

    )



# Load the data

df = pd.read_parquet(INPUT_FILE)



print(f"\n✓ Loaded data successfully")

print(f"  File: {INPUT_FILE.name}")

print(f"  Shape: {df.shape}")

print(f"  Rows: {len(df):,}")

print(f"  Columns: {len(df.columns)}")



# Show column names

print(f"\nColumns loaded:")

for i, col in enumerate(df.columns, 1):

    marker = "← TARGET" if col == "Won" else ""

    print(f"  {i:2d}. {col} {marker}")



# Create a working copy

df_clean = df.copy()

print(f"\n✓ Created working copy: df_clean")



# Basic info

print("\n" + "="*80)

print("DATA OVERVIEW")

print("="*80)

print(df_clean.info())
# ===========================================================================

# CELL 2: STANDARDIZE COLUMN NAMES TO SNAKE_CASE

# ===========================================================================



print("\n" + "="*80)

print("STANDARDIZING COLUMN NAMES")

print("="*80)



def to_snake_case(name):

    """

    Convert column name to snake_case.

    

    Examples:

        'Date of Race' -> 'date_of_race'

        'SP Fav' -> 'sp_fav'

        'Avg % SP Drop Last 5 races' -> 'avg_pct_sp_drop_last_5_races'

    """

    # Replace special characters

    name = name.replace('/', '_')

    name = name.replace('%', 'pct')

    name = name.replace(' ', '_')

    

    # Convert to lowercase

    name = name.lower()

    

    # Remove consecutive underscores

    while '__' in name:

        name = name.replace('__', '_')

    

    # Remove leading/trailing underscores

    name = name.strip('_')

    

    return name





# Show before/after mapping

print("\nColumn name mapping:")

print("-" * 80)



column_mapping = {}

for col in df_clean.columns:

    new_name = to_snake_case(col)

    column_mapping[col] = new_name

    print(f"  {col:40s} -> {new_name}")



# Apply the renaming

df_clean = df_clean.rename(columns=column_mapping)



print("\n" + "="*80)

print(f"✓ Renamed {len(column_mapping)} columns to snake_case")

print("="*80)



# Verify

print("\nNew column names:")

for i, col in enumerate(df_clean.columns, 1):

    marker = "← TARGET" if col == "won" else ""

    print(f"  {i:2d}. {col} {marker}")



# Show sample

print("\n" + "="*80)

print("SAMPLE OF RENAMED DATA")

print("="*80)

print(df_clean.head(3))
# ===========================================================================

# CELL 3: CREATE RACE_ID AND HELPER FEATURES

# ===========================================================================



print("\n" + "="*80)

print("CREATING RACE ID AND HELPER FEATURES")

print("="*80)



# Ensure date_of_race is datetime

df_clean['date_of_race'] = pd.to_datetime(df_clean['date_of_race'])



# Create race_id (unique identifier for each race)

# Format: YYYY-MM-DD_TRACK_HH:MM:SS

df_clean['race_id'] = (

    df_clean['date_of_race'].dt.strftime('%Y-%m-%d') + '_' +

    df_clean['track'].astype(str) + '_' +

    df_clean['time'].astype(str)

)



print(f"\n✓ Created race_id")

print(f"  Sample race_id: {df_clean['race_id'].iloc[0]}")

print(f"  Unique races: {df_clean['race_id'].nunique():,}")

print(f"  Total entries: {len(df_clean):,}")

print(f"  Avg runners per race: {len(df_clean) / df_clean['race_id'].nunique():.1f}")



# Create is_first_time_runner flag

# A horse is a first-time runner if it has no previous race history

df_clean['is_first_time_runner'] = (

    df_clean['total_prev_races'].isna() | 

    (df_clean['total_prev_races'] == 0)

).astype(int)



print(f"\n✓ Created is_first_time_runner flag")

print(f"  First-time runners: {df_clean['is_first_time_runner'].sum():,} "

      f"({df_clean['is_first_time_runner'].mean()*100:.1f}%)")



# Create year, month, day of week features (useful for modeling)

df_clean['year'] = df_clean['date_of_race'].dt.year

df_clean['month'] = df_clean['date_of_race'].dt.month

df_clean['day_of_week'] = df_clean['date_of_race'].dt.dayofweek  # 0=Monday, 6=Sunday



print(f"\n✓ Created temporal features: year, month, day_of_week")





# Show sample

print("\n" + "="*80)

print("SAMPLE WITH NEW FEATURES")

print("="*80)

print(df_clean[['race_id', 'horse', 'is_first_time_runner',

                'total_prev_races', 'year', 'month', 'day_of_week']].head())
# ===========================================================================

# CELL 4: EXPLORE DATA TYPES AND IDENTIFY CLEANING NEEDS

# ===========================================================================



print("\n" + "="*80)

print("DATA TYPE ANALYSIS")

print("="*80)



# Separate columns by dtype

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns.tolist()



print(f"\nNumeric columns ({len(numeric_cols)}):")

for col in numeric_cols:

    print(f"  - {col}")



print(f"\nObject columns ({len(object_cols)}):")

for col in object_cols:

    print(f"  - {col}")



print(f"\nDatetime columns ({len(datetime_cols)}):")

for col in datetime_cols:

    print(f"  - {col}")



# Check for object columns that should be numeric

print("\n" + "="*80)

print("CHECKING OBJECT COLUMNS FOR NUMERIC CONVERSION")

print("="*80)



potentially_numeric = []

for col in object_cols:

    if col in ['won', 'race_id']:  # Skip known categorical columns

        continue

    

    # Sample unique values

    unique_vals = df_clean[col].dropna().unique()[:10]

    

    print(f"\n{col}:")

    print(f"  Unique values (sample): {unique_vals}")

    print(f"  Total unique: {df_clean[col].nunique()}")

    print(f"  Missing: {df_clean[col].isna().sum()} ({df_clean[col].isna().mean()*100:.1f}%)")

    

    # Try to convert to numeric to see if it makes sense

    test_convert = pd.to_numeric(df_clean[col], errors='coerce')

    pct_convertible = test_convert.notna().sum() / len(df_clean)

    

    if pct_convertible > 0.5:  # If >50% can be converted

        potentially_numeric.append(col)

        print(f"  → Can convert {pct_convertible*100:.1f}% to numeric")



if potentially_numeric:

    print(f"\n" + "="*80)

    print(f"COLUMNS THAT SHOULD BE NUMERIC: {potentially_numeric}")

    print("="*80)
# ===========================================================================

# CELL 5: CLEAN OBJECT COLUMNS THAT SHOULD BE NUMERIC

# ===========================================================================



print("\n" + "="*80)

print("CLEANING OBJECT COLUMNS TO NUMERIC")

print("="*80)



# Columns that are stored as objects but should be numeric

numeric_conversion_cols = [

    'forecasted_odds',

    'last_time_out_position',

    'course_winner',

    'distance_winner', 

    'up_in_trip'

]



# Filter to only columns that exist

numeric_conversion_cols = [col for col in numeric_conversion_cols if col in df_clean.columns]



for col in numeric_conversion_cols:

    print(f"\nCleaning '{col}':")

    print(f"  Original dtype: {df_clean[col].dtype}")

    

    # Show problematic values

    non_numeric = df_clean[col].dropna()

    test_numeric = pd.to_numeric(non_numeric, errors='coerce')

    problematic = non_numeric[test_numeric.isna()]

    

    if len(problematic) > 0:

        print(f"  Non-numeric values found: {problematic.unique()[:10]}")

    

    # Convert to numeric, non-numeric values become NaN

    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    

    print(f"  New dtype: {df_clean[col].dtype}")

    print(f"  Missing after conversion: {df_clean[col].isna().sum()} "

          f"({df_clean[col].isna().mean()*100:.1f}%)")



print("\n" + "="*80)

print("✓ Numeric conversion complete")

print("="*80)
# ===========================================================================

# CELL 6: HANDLE CATEGORICAL MISSING VALUES

# ===========================================================================



print("\n" + "="*80)

print("HANDLING CATEGORICAL MISSING VALUES")

print("="*80)



# Headgear: NaN means no headgear

print("\n1. Cleaning 'headgear':")

print(f"   Missing before: {df_clean['headgear'].isna().sum()}")

df_clean['headgear'] = df_clean['headgear'].fillna('None')

print(f"   Missing after: {df_clean['headgear'].isna().sum()}")

print(f"   Value counts:\n{df_clean['headgear'].value_counts().head()}")



# Placed in Betfair Market: NaN means not in market

print("\n2. Cleaning 'placed_in_betfair_market':")

print(f"   Missing before: {df_clean['placed_in_betfair_market'].isna().sum()}")

df_clean['placed_in_betfair_market'] = df_clean['placed_in_betfair_market'].fillna('NO_MARKET')

print(f"   Missing after: {df_clean['placed_in_betfair_market'].isna().sum()}")

print(f"   Value counts:\n{df_clean['placed_in_betfair_market'].value_counts()}")



print("\n" + "="*80)

print("✓ Categorical cleaning complete")

print("="*80)
# ===========================================================================

# CELL 7: IMPUTE MISSING NUMERIC VALUES

# ===========================================================================



print("\n" + "="*80)

print("IMPUTING MISSING NUMERIC VALUES")

print("="*80)



# Create a copy to track changes

df_imputed = df_clean.copy()



# Strategy 1: Count-based features -> 0 (no history means 0 counts)

count_cols = [

    'total_prev_races',

    'runs_last_18_months',

    'wins_last_5_races',

    'course_wins',

    'distance_wins',

    'class_wins',

    'going_wins'

]

count_cols = [col for col in count_cols if col in df_imputed.columns]



print("\n1. Count-based features (filling with 0):")

for col in count_cols:

    missing_before = df_imputed[col].isna().sum()

    df_imputed[col] = df_imputed[col].fillna(0)

    print(f"   {col:30s}: {missing_before:6,} missing → 0")



# Strategy 2: Days since last time out -> 9999 for first-timers

print("\n2. Days since last time out (filling with 9999 for first-timers):")

missing_before = df_imputed['days_since_last_time_out'].isna().sum()

df_imputed['days_since_last_time_out'] = df_imputed['days_since_last_time_out'].fillna(9999)

print(f"   days_since_last_time_out: {missing_before:6,} missing → 9999")



# Strategy 3: Stall -> 0 (non-stall races)

print("\n3. Stall (filling with 0 for non-stall races):")

missing_before = df_imputed['stall'].isna().sum()

df_imputed['stall'] = df_imputed['stall'].fillna(0)

print(f"   stall: {missing_before:6,} missing → 0")



# Strategy 4: Class -> 0 (unclassified races)

print("\n4. Class (filling with 0 for unclassified races):")

missing_before = df_imputed['class'].isna().sum()

df_imputed['class'] = df_imputed['class'].fillna(0)

print(f"   class: {missing_before:6,} missing → 0")



# Strategy 5: Rating/Rank features -> 0

rating_cols = [

    'rbd_rating',

    'rbd_rank',

    'avg_pct_sp_drop_last_18_mths',

    'avg_pct_sp_drop_last_5_races',

    'betfair_place_sp',

    'betfair_rank'

]

rating_cols = [col for col in rating_cols if col in df_imputed.columns]



print("\n5. Rating/Rank features (filling with 0):")

for col in rating_cols:

    missing_before = df_imputed[col].isna().sum()

    df_imputed[col] = df_imputed[col].fillna(0)

    print(f"   {col:30s}: {missing_before:6,} missing → 0")



# Strategy 6: Historical position/winner features -> 0

historical_cols = [

    'last_time_out_position',

    'course_winner',

    'distance_winner',

    'up_in_trip'

]

historical_cols = [col for col in historical_cols if col in df_imputed.columns]



print("\n6. Historical position/winner features (filling with 0):")

for col in historical_cols:

    missing_before = df_imputed[col].isna().sum()

    df_imputed[col] = df_imputed[col].fillna(0)

    print(f"   {col:30s}: {missing_before:6,} missing → 0")



# Strategy 7: Pace -> median (few missing, central tendency)

print("\n7. Pace (filling with median):")

if 'pace' in df_imputed.columns:

    pace_median = df_imputed['pace'].median()

    missing_before = df_imputed['pace'].isna().sum()

    df_imputed['pace'] = df_imputed['pace'].fillna(pace_median)

    print(f"   pace: {missing_before:6,} missing → {pace_median:.1f} (median)")



print("\n" + "="*80)

print("✓ Numeric imputation complete")

print("="*80)
# ===========================================================================

# CELL 8: VALIDATE IMPUTATION

# ===========================================================================



print("\n" + "="*80)

print("VALIDATING IMPUTATION")

print("="*80)



# Check for remaining missing values

remaining_missing = df_imputed.isnull().sum()

remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)



if remaining_missing.empty:

    print("\n✅ SUCCESS! No missing values remain in the dataset.")

else:

    print(f"\n⚠️  WARNING: {len(remaining_missing)} columns still have missing values:")

    print("\n" + "-"*80)

    for col, count in remaining_missing.items():

        pct = (count / len(df_imputed)) * 100

        print(f"  {col:30s}: {count:6,} missing ({pct:5.2f}%)")

    print("-"*80)



# Verify first-time runner logic

print("\n" + "="*80)

print("FIRST-TIME RUNNER VALIDATION")

print("="*80)



first_timers = df_imputed[df_imputed['is_first_time_runner'] == 1]

print(f"\nFirst-time runners: {len(first_timers):,}")

print(f"\nVerifying imputation for first-timers:")

print(f"  total_prev_races = 0: {(first_timers['total_prev_races'] == 0).sum():,}")

print(f"  runs_last_18_months = 0: {(first_timers['runs_last_18_months'] == 0).sum():,}")

print(f"  days_since_last_time_out = 9999: {(first_timers['days_since_last_time_out'] == 9999).sum():,}")



# Show summary statistics

print("\n" + "="*80)

print("IMPUTED DATA SUMMARY")

print("="*80)

print(df_imputed.describe())
# ===========================================================================

# CELL 9: VALIDATE RACE-LEVEL CONSISTENCY

# ===========================================================================



print("\n" + "="*80)

print("VALIDATING RACE-LEVEL CONSISTENCY")

print("="*80)



# Race-level columns should be the same for all horses in the same race

race_level_cols = ['runners', 'track', 'going', 'type', 'country', 'date_of_race', 'class']

race_level_cols = [col for col in race_level_cols if col in df_imputed.columns]



print("\nChecking if these columns are consistent within each race:")

for col in race_level_cols:

    print(f"  - {col}")



# Check for inconsistencies

print("\n" + "-"*80)

print("Checking for inconsistencies...")

print("-"*80)



inconsistent_races = {}

for col in race_level_cols:

    # Count unique values per race_id

    unique_per_race = df_imputed.groupby('race_id')[col].nunique()

    

    # Find races with more than one unique value

    inconsistent = unique_per_race[unique_per_race > 1]

    

    if len(inconsistent) > 0:

        inconsistent_races[col] = len(inconsistent)

        print(f"  ⚠️  {col:20s}: {len(inconsistent):,} races with inconsistent values")

    else:

        print(f"  ✓  {col:20s}: All races consistent")



# Fix inconsistencies if found

if inconsistent_races:

    print("\n" + "="*80)

    print("FIXING INCONSISTENCIES")

    print("="*80)

    

    # For numeric columns, use max value

    numeric_race_cols = ['runners', 'class']

    numeric_race_cols = [col for col in numeric_race_cols if col in df_imputed.columns]

    

    for col in numeric_race_cols:

        df_imputed[col] = df_imputed.groupby('race_id')[col].transform('max')

        print(f"  ✓ Fixed {col} (using max value per race)")

    

    # For categorical columns, use first value (they should all be the same)

    categorical_race_cols = ['track', 'going', 'type', 'country', 'date_of_race']

    categorical_race_cols = [col for col in categorical_race_cols if col in df_imputed.columns]

    

    for col in categorical_race_cols:

        df_imputed[col] = df_imputed.groupby('race_id')[col].transform('first')

        print(f"  ✓ Fixed {col} (using first value per race)")

    

    # Verify fix

    print("\n" + "-"*80)

    print("Verifying fixes...")

    print("-"*80)

    

    for col in race_level_cols:

        unique_per_race = df_imputed.groupby('race_id')[col].nunique()

        inconsistent = unique_per_race[unique_per_race > 1]

        

        if len(inconsistent) > 0:

            print(f"  ❌ {col:20s}: Still {len(inconsistent):,} inconsistent races!")

        else:

            print(f"  ✅ {col:20s}: All races now consistent")

else:

    print("\n✅ No inconsistencies found. All race-level data is consistent.")



print("\n" + "="*80)

print("✓ Race consistency validation complete")

print("="*80)
# ===========================================================================

# CELL 10: IMPUTE REMAINING MISSING ODDS VALUES

# ===========================================================================



print("\n" + "="*80)

print("IMPUTING REMAINING MISSING ODDS")

print("="*80)



# These are betting odds columns with very few missing values

odds_cols = ['forecasted_odds', 'pre_min', 'pre_max', 'betfair_sp', 'industry_sp']



print("\nMissing values before imputation:")

for col in odds_cols:

    missing = df_imputed[col].isna().sum()

    pct = (missing / len(df_imputed)) * 100

    print(f"  {col:20s}: {missing:6,} ({pct:5.2f}%)")



# Strategy: Use SP Fav ranking to estimate missing odds

# Odds typically follow a pattern based on favorite ranking

print("\n" + "-"*80)

print("Imputation strategy:")

print("  1. Group by race_id and sp_fav")

print("  2. Fill with median odds for that favorite ranking")

print("  3. If still missing, use overall median")

print("-"*80)



for col in odds_cols:

    # Calculate median odds by favorite ranking within each race

    race_fav_median = df_imputed.groupby(['race_id', 'sp_fav'])[col].transform('median')

    

    # Fill missing with race/favorite median first

    df_imputed[col] = df_imputed[col].fillna(race_fav_median)

    

    # If still missing, use overall median as last resort

    overall_median = df_imputed[col].median()

    df_imputed[col] = df_imputed[col].fillna(overall_median)

    

    print(f"\n  ✓ {col}: filled with race/favorite median (backup: {overall_median:.2f})")



# Verify

print("\n" + "="*80)

print("VERIFICATION: Missing values after imputation:")

print("="*80)

for col in odds_cols:

    missing = df_imputed[col].isna().sum()

    print(f"  {col:20s}: {missing:6,}")



if df_imputed[odds_cols].isna().sum().sum() == 0:

    print("\n✅ All odds columns now complete!")

else:

    print("\n⚠️  Some values still missing (acceptable for some models)")



# Final check: NO missing values in entire dataset

total_missing = df_imputed.isnull().sum().sum()

print("\n" + "="*80)

print(f"TOTAL MISSING VALUES IN DATASET: {total_missing:,}")

print("="*80)



if total_missing == 0:

    print("\n🎉 SUCCESS! Dataset is 100% complete!")

else:

    print(f"\n⚠️  {total_missing:,} missing values remain")

    remaining = df_imputed.isnull().sum()

    remaining = remaining[remaining > 0]

    print("\nColumns with missing values:")

    for col, count in remaining.items():

        print(f"  {col:30s}: {count:6,}")

# ===========================================================================

# CELL 11: FINAL DATA QUALITY REPORT

# ===========================================================================



print("\n" + "="*80)

print("FINAL DATA QUALITY REPORT")

print("="*80)



# 1. Shape and size

print("\n1. DATASET SHAPE:")

print(f"   Rows: {len(df_imputed):,}")

print(f"   Columns: {len(df_imputed.columns)}")



# 2. Missing values

print("\n2. MISSING VALUES:")

missing_summary = df_imputed.isnull().sum()

missing_summary = missing_summary[missing_summary > 0]

if missing_summary.empty:

    print("   ✅ No missing values")

else:

    print(f"   ⚠️  {len(missing_summary)} columns with missing values")



# 3. Data types

print("\n3. DATA TYPES:")

dtype_counts = df_imputed.dtypes.value_counts()

for dtype, count in dtype_counts.items():

    print(f"   {str(dtype):20s}: {count:3d} columns")



# 4. Target variable

print("\n4. TARGET VARIABLE:")

target_col = 'won'

if target_col in df_imputed.columns:

    print(f"   Total races: {len(df_imputed):,}")

    print(f"   Winners: {df_imputed[target_col].sum():,}")

    print(f"   Losers: {(df_imputed[target_col] == 0).sum():,}")

    print(f"   Win rate: {df_imputed[target_col].mean()*100:.2f}%")

else:

    print("   ⚠️  Target column 'won' not found")



# 5. Date range

print("\n5. DATE RANGE:")

print(f"   Start: {df_imputed['date_of_race'].min()}")

print(f"   End: {df_imputed['date_of_race'].max()}")

print(f"   Span: {(df_imputed['date_of_race'].max() - df_imputed['date_of_race'].min()).days} days")



# 6. Unique values

print("\n6. KEY STATISTICS:")

print(f"   Unique races: {df_imputed['race_id'].nunique():,}")

print(f"   Unique horses: {df_imputed['horse'].nunique():,}")

print(f"   Unique jockeys: {df_imputed['jockey'].nunique():,}")

print(f"   Unique trainers: {df_imputed['trainer'].nunique():,}")

print(f"   Unique tracks: {df_imputed['track'].nunique()}")

print(f"   Avg runners per race: {df_imputed.groupby('race_id').size().mean():.1f}")



# 7. First-time runners

print("\n7. FIRST-TIME RUNNERS:")

print(f"   Count: {df_imputed['is_first_time_runner'].sum():,}")

print(f"   Percentage: {df_imputed['is_first_time_runner'].mean()*100:.1f}%")



# 8. Memory usage

print("\n8. MEMORY USAGE:")

memory_mb = df_imputed.memory_usage(deep=True).sum() / 1024 / 1024

print(f"   Total: {memory_mb:.1f} MB")



print("\n" + "="*80)

print("✅ DATA QUALITY REPORT COMPLETE")

print("="*80)
# ===========================================================================

# CELL 12: SAVE PROCESSED DATA

# ===========================================================================



print("\n" + "="*80)

print("SAVING PROCESSED DATA")

print("="*80)



# Define output path

OUTPUT_DIR = Path('../data/processed')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "02_cleaned_imputed.parquet"



# Save the fully processed dataframe

df_imputed.to_parquet(OUTPUT_FILE, index=False)



print(f"\n✅ Successfully saved processed data!")

print(f"\nOutput file: {OUTPUT_FILE}")

print(f"Shape: {df_imputed.shape}")

print(f"Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")



print("\n" + "="*80)

print("NOTEBOOK 02 COMPLETE!")

print("="*80)

print("\nProcessed data is ready for:")

print("  → Notebook 03: Feature Engineering")

print("  → Notebook 04: Feature Selection")

print("  → Notebook 05: Model Training")
