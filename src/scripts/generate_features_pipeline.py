import gc
import polars as pl
from pathlib import Path
from typing import List
from simplified_feature_engineering import create_simplified_features
from utils.load_env import PATH_DATA

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_data(months: List[str], data_path: str = PATH_DATA) -> pl.DataFrame:
    """
    Load data for the specified months from parquet files
    
    Args:
        months: List of month identifiers (e.g., ['2019-Oct', '2019-Nov'])
        data_path: Base path to the data directory
    
    Returns:
        Combined polars DataFrame
    """
    dataframes = []
    
    for month in months:
        try:
            file_path = f"{data_path}/silver/{month}_filtered.parquet"
            print(f"Loading {month}: ", end='')
            
            # Load without forcing schema - let polars infer from the saved parquet
            df_month = pl.read_parquet(file_path)
            
            # Convert string columns to categorical for better memory usage
            df_month = df_month.with_columns([
                pl.col("user_id").cast(pl.Categorical),
                pl.col("product_id").cast(pl.Categorical),
                pl.col("category_id").cast(pl.Categorical),
                pl.col("event_type").cast(pl.Categorical),
                pl.col("category_code").cast(pl.Categorical),
                pl.col("brand").cast(pl.Categorical),
                pl.col("user_session").cast(pl.Categorical)
            ])
            
            # Parse datetime and add month column
            df_month = df_month.with_columns([
                pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %Z", strict=False).alias("event_time"),
                pl.lit(month).alias("month")
            ])
            
            print(f"{df_month.shape[0]:,} rows")
            dataframes.append(df_month)
            
        except Exception as e:
            print(f"File not found or error - {e}")
    
    if dataframes:
        combined_df = pl.concat(dataframes)
        print(f"\nTotal combined data: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
        return combined_df
    else:
        print("No data files found or loaded successfully")
        return pl.DataFrame()

def process_time_windows(months_windows: List[List[str]], data_path: str = PATH_DATA) -> None:
    """
    Process multiple time windows and generate features for each
    
    Args:
        months_windows: List of time windows, each containing a list of months
        data_path: Base path to the data directory
    """
    print("="*80)
    print("SIMPLIFIED FEATURE GENERATION PIPELINE")
    print("="*80)
    
    # Ensure output directory exists
    output_dir = Path(data_path) / "gold"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_windows = len(months_windows)
    
    for i, months in enumerate(months_windows, 1):
        print(f"\n[{i}/{total_windows}] Processing months window: {months}", flush=True)
        print("-" * 60)
        
        try:
            # Load data for this time window
            df = load_data(months, data_path)
            
            if df.height == 0:
                print(f"⚠️  No data loaded for window {months}, skipping...")
                continue
            
            # Generate simplified features
            print("\nGenerating simplified features...")
            df_features = create_simplified_features(df)
            
            # Clean up original dataframe
            del df
            
            # Save features
            window_name = f"{months[0]}_{months[-1]}"
            output_path = output_dir / f"{window_name}_simplified_features.parquet"
            
            print(f"Saving features to: {output_path}")
            df_features.write_parquet(str(output_path), compression="gzip")
            
            print(f"✅ Successfully saved {df_features.shape[0]:,} users with {df_features.shape[1]-1} features")
            
            # Clean up features dataframe
            del df_features
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing window {months}: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)

def main():
    """Main execution function"""
    
    # Define time windows for feature generation
    months_windows = [
        ['2019-Oct', '2019-Nov', '2019-Dec', '2020-Jan'],
        ['2019-Nov', '2019-Dec', '2020-Jan', '2020-Feb'],
        ['2019-Dec', '2020-Jan', '2020-Feb', '2020-Mar'],
        ['2020-Jan', '2020-Feb', '2020-Mar', '2020-Apr']
    ]
    
    # Process all time windows
    process_time_windows(months_windows, PATH_DATA)

if __name__ == "__main__":
    main()
