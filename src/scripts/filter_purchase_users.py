"""
Optimized script to filter dataset by users with at least one purchase
and save the combined data as parquet format.

This script processes multiple months of e-commerce data, identifies users
who made at least one purchase, filters all data to include only those users,
and saves the result as a compressed parquet file.

Usage:
    python filter_purchase_users.py --save-mode combined  # Save as single combined file
    python filter_purchase_users.py --save-mode individual  # Save each month separately
"""


from utils.load_env import PATH_DATA
import polars as pl
import gc
import argparse
from typing import Set, Tuple, List
from tqdm import tqdm

# Enable string cache to handle categorical columns from different sources
pl.enable_string_cache()


def identify_purchase_users(months: List[str], path_data: str) -> Tuple[Set[str], Set[str]]:
    """
    Identify users with at least one purchase across all months.
    
    Args:
        months: List of month strings (e.g., ['2019-Oct', '2019-Nov'])
        path_data: Path to the data directory
        
    Returns:
        Tuple of (purchase_user_ids, all_user_ids)
    """
    purchase_user_ids = set()
    all_user_ids = set()

    print("Step 1: Identifying users with at least one purchase...")
    
    for month in tqdm(months, desc="Processing months"):
        try:
            file_path = f"{path_data}/source/{month}.csv.gz"
            
            # Load only necessary columns for efficiency
            df_month = pl.read_csv(
                file_path,
                schema_overrides={
                    'user_id': pl.Categorical,
                    'event_type': pl.Categorical
                },
                try_parse_dates=False,
                infer_schema_length=1000,
                columns=["user_id", "event_type"]
            )
            
            # Collect all user IDs
            all_user_ids.update(df_month["user_id"].unique().to_list())
            
            # Filter for purchases and collect user IDs
            purchase_users = df_month.filter(
                pl.col("event_type") == "purchase"
            )["user_id"].unique().to_list()
            
            purchase_user_ids.update(purchase_users)
            
        except Exception as e:
            # Skip files that don't exist or have errors
            tqdm.write(f"Warning: Could not process {month}: {e}")
            continue
        
        # Clean up memory
        del df_month
        if 'purchase_users' in locals():
            del purchase_users
        gc.collect()

    print(f"Found {len(purchase_user_ids):,} unique users with purchases out of {len(all_user_ids):,} total users")
    return purchase_user_ids, all_user_ids


def process_and_save_filtered_data(months: List[str], purchase_user_ids: Set[str], path_data: str, save_mode: str = "combined") -> None:
    """
    Load, filter, and save the dataset.
    
    Args:
        months: List of month strings
        purchase_user_ids: Set of user IDs who made purchases
        path_data: Path to the data directory
        save_mode: Either "combined" or "individual"
    """
    print(f"\nStep 2: Processing and saving data in '{save_mode}' mode...")
    
    if save_mode == "individual":
        # Save each month separately
        for month in tqdm(months, desc="Processing months"):
            try:
                file_path = f"{path_data}/source/{month}.csv.gz"
                
                # Load with optimized schema
                df_month = pl.read_csv(
                    file_path,
                    schema_overrides={
                        'user_id': pl.Categorical,
                        'product_id': pl.Categorical,
                        'category_id': pl.Categorical,
                        'price': pl.Float32,
                        'event_type': pl.Categorical,
                        'category_code': pl.Categorical,
                        'brand': pl.Categorical,
                        'user_session': pl.Categorical,
                        'event_time': pl.Utf8
                    },
                    try_parse_dates=False,
                    infer_schema_length=1000
                )
                
                # Filter by purchase users and positive prices
                df_month = df_month.filter(
                    (pl.col("user_id").is_in(purchase_user_ids)) &
                    (pl.col("price") > 0)
                )
                
                if df_month.shape[0] > 0:
                    # Convert categorical columns to string to avoid parquet issues
                    df_month = df_month.with_columns([
                        pl.col("user_id").cast(pl.Utf8),
                        pl.col("product_id").cast(pl.Utf8),
                        pl.col("category_id").cast(pl.Utf8),
                        pl.col("event_type").cast(pl.Utf8),
                        pl.col("category_code").cast(pl.Utf8),
                        pl.col("brand").cast(pl.Utf8),
                        pl.col("user_session").cast(pl.Utf8)
                    ])
                    
                    # Save individual month file with better compression
                    output_path = f"{path_data}/silver/{month}_filtered.parquet"
                    df_month.write_parquet(output_path, compression="snappy")
                    
                    # Log file size for monitoring
                    import os
                    file_size = os.path.getsize(output_path) / (1024**3)  # Size in GB
                    tqdm.write(f"Saved {month}: {df_month.shape[0]:,} rows, {file_size:.2f} GB")
                else:
                    tqdm.write(f"Skipped {month}: No data after filtering")
                
            except Exception as e:
                tqdm.write(f"Warning: Could not process {month}: {e}")
                continue
            
            # Clean up memory after each iteration
            del df_month
            gc.collect()
    
    else:  # combined mode
        dataframes = []
        
        for month in tqdm(months, desc="Loading months"):
            try:
                file_path = f"{path_data}/source/{month}.csv.gz"
                
                # Load with optimized schema
                df_month = pl.read_csv(
                    file_path,
                    schema_overrides={
                        'user_id': pl.Categorical,
                        'product_id': pl.Categorical,
                        'category_id': pl.Categorical,
                        'price': pl.Float32,
                        'event_type': pl.Categorical,
                        'category_code': pl.Categorical,
                        'brand': pl.Categorical,
                        'user_session': pl.Categorical,
                        'event_time': pl.Utf8
                    },
                    try_parse_dates=False,
                    infer_schema_length=1000
                )
                
                # Filter by purchase users and positive prices in one operation
                df_month = df_month.filter(
                    (pl.col("user_id").is_in(purchase_user_ids)) &
                    (pl.col("price") > 0)
                )
                
                if df_month.shape[0] > 0:  # Only add non-empty dataframes
                    dataframes.append(df_month)
                    tqdm.write(f"Loaded {month}: {df_month.shape[0]:,} rows")
                else:
                    tqdm.write(f"Skipped {month}: No data after filtering")
                
            except Exception as e:
                tqdm.write(f"Warning: Could not process {month}: {e}")
                continue
            
            # Clean up memory after each iteration
            del df_month
            gc.collect()
        
        # Combine all dataframes if any exist
        if dataframes:
            print("Combining all dataframes...")
            combined_df = pl.concat(dataframes)
            
            # Convert categorical columns to string to avoid parquet issues
            combined_df = combined_df.with_columns([
                pl.col("user_id").cast(pl.Utf8),
                pl.col("product_id").cast(pl.Utf8),
                pl.col("category_id").cast(pl.Utf8),
                pl.col("event_type").cast(pl.Utf8),
                pl.col("category_code").cast(pl.Utf8),
                pl.col("brand").cast(pl.Utf8),
                pl.col("user_session").cast(pl.Utf8)
            ])
            
            # Save as compressed parquet
            output_path = f"{path_data}/silver/combined_data.parquet"
            combined_df.write_parquet(output_path, compression="snappy")
            
            # Log file size for monitoring
            import os
            file_size = os.path.getsize(output_path) / (1024**3)  # Size in GB
            print(f"Saved combined data: {combined_df.shape[0]:,} rows, {file_size:.2f} GB to {output_path}")
            
            # Clean up final dataframe
            del combined_df
        else:
            print("Warning: No data to save after filtering")
            
        # Clean up list of dataframes
        del dataframes
        gc.collect()


def main():
    """Main execution function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Filter e-commerce data by users with at least one purchase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_purchase_users.py --save-mode combined     # Save as single file
  python filter_purchase_users.py --save-mode individual   # Save each month separately
        """
    )
    
    parser.add_argument(
        '--save-mode',
        choices=['combined', 'individual'],
        default='combined',
        help='How to save the filtered data: "combined" for single file, "individual" for separate monthly files (default: combined)'
    )
    
    args = parser.parse_args()
    
    # Define months to process
    months = [
        '2019-Oct', '2019-Nov', '2019-Dec', 
        '2020-Jan', '2020-Feb', '2020-Mar', '2020-Apr'
    ]
    
    print(f"Starting data filtering process...")
    print(f"Save mode: {args.save_mode}")
    print(f"Months to process: {len(months)}")
    
    # Step 1: Identify users with purchases
    purchase_user_ids, all_user_ids = identify_purchase_users(months, PATH_DATA)
    
    # Step 2: Process and save filtered data
    process_and_save_filtered_data(months, purchase_user_ids, PATH_DATA, args.save_mode)
    
    # Final cleanup
    del purchase_user_ids, all_user_ids
    gc.collect()
    
    print("\nData filtering process completed successfully!")


if __name__ == "__main__":
    main()
