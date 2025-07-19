import math
import polars as pl
from typing import Dict, List, Optional, Tuple

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_event_type_dataframes(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Split dataframe by event type for reuse across functions"""
    return {
        'purchase': df.filter(pl.col("event_type") == "purchase"),
        'view': df.filter(pl.col("event_type") == "view"),
        'cart': df.filter(pl.col("event_type") == "cart")
    }

def calculate_price_percentile_features(purchase_df: pl.DataFrame) -> Tuple[float, float, pl.DataFrame]:
    """Calculate price percentiles and product normal prices"""
    if purchase_df.height == 0:
        return 0, 0, pl.DataFrame()
    
    price_90th = purchase_df.select(pl.col("price").quantile(0.9)).item()
    price_25th = purchase_df.select(pl.col("price").quantile(0.25)).item()
    
    product_normal_prices = purchase_df.group_by("product_id").agg([
        pl.col("price").median().alias("normal_price")
    ])
    
    return price_90th, price_25th, product_normal_prices

def join_user_features(all_users: pl.DataFrame, feature_dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """Join multiple feature dataframes to all users"""
    result = all_users
    for feature_df in feature_dfs:
        result = result.join(feature_df, on="user_id", how="left")
    return result.fill_null(0)

# ============================================================================
# SIMPLIFIED FEATURE ENGINEERING CLASS
# ============================================================================

class SimplifiedFeatureEngineering:
    """Simplified feature engineering class focusing on specific feature categories"""
    
    def __init__(self, df: pl.DataFrame, analysis_date: Optional[str] = None):
        self.df = df
        self.analysis_date = analysis_date or df.select(pl.col("event_time").max()).item()
        self.event_dfs = get_event_type_dataframes(df)
        self.all_users = df.select("user_id").unique()
        
        # Calculate shared components once
        self._session_data = self._calculate_session_data()
        self._price_percentiles = calculate_price_percentile_features(self.event_dfs['purchase'])
    
    def _calculate_session_data(self) -> pl.DataFrame:
        """Calculate session-level data used across multiple functions"""
        return self.df.group_by(["user_id", "user_session"]).agg([
            pl.col("event_time").min().alias("session_start"),
            pl.col("event_time").max().alias("session_end"),
            pl.len().alias("session_events"),
            pl.col("event_type").filter(pl.col("event_type") == "purchase").len().alias("purchases_in_session"),
            pl.col("event_type").filter(pl.col("event_type") == "view").len().alias("views_in_session")
        ])
    
    def calculate_transactional_features(self) -> pl.DataFrame:
        """Calculate transactional features"""
        print("Calculating transactional features...")
        
        purchase_df = self.event_dfs['purchase']
        if purchase_df.height == 0:
            # Return empty features for users with no purchases
            return self.all_users.with_columns([
                pl.lit(0).alias("total_purchases"),
                pl.lit(0.0).alias("total_spend"),
                pl.lit(0.0).alias("median_order_value"),
                pl.lit(0.0).alias("price_sensitivity_score"),
                pl.lit(0.0).alias("purchase_frequency_per_day"),
                pl.lit(0.0).alias("price_range")
            ])
        
        transactional_features = purchase_df.group_by("user_id").agg([
            pl.len().alias("total_purchases"),
            pl.col("price").sum().alias("total_spend"),
            pl.col("price").median().alias("median_order_value"),
            pl.col("price").std().alias("price_sensitivity_score"),
            (pl.len() / pl.col("event_time").dt.date().n_unique()).alias("purchase_frequency_per_day"),
            (pl.col("price").max() - pl.col("price").min()).alias("price_range")
        ])
        
        return join_user_features(self.all_users, [transactional_features])
    
    def calculate_price_affinity_features(self) -> pl.DataFrame:
        """Calculate price affinity features"""
        print("Calculating price affinity features...")
        
        purchase_df = self.event_dfs['purchase']
        if purchase_df.height == 0:
            # Return empty features for users with no purchases
            return self.all_users.with_columns([
                pl.lit(0.0).alias("premium_product_affinity"),
                pl.lit(0.0).alias("budget_product_affinity"),
                pl.lit(0.0).alias("discount_affinity"),
                pl.lit(0.0).alias("avg_discount_pct")
            ])
        
        price_90th, price_25th, product_normal_prices = self._price_percentiles
        
        purchase_with_normal = purchase_df.join(product_normal_prices, on="product_id", how="left")
        
        price_affinity_features = purchase_with_normal.group_by("user_id").agg([
            (pl.col("price").filter(pl.col("price") >= price_90th).len() / pl.len()).alias("premium_product_affinity"),
            (pl.col("price").filter(pl.col("price") <= price_25th).len() / pl.len()).alias("budget_product_affinity"),
            (pl.col("price").filter(pl.col("price") < pl.col("normal_price")).len() / pl.len()).alias("discount_affinity"),
            ((100 * (pl.col("normal_price") - pl.col("price")) / pl.col("normal_price")).filter(pl.col("price") < pl.col("normal_price")).mean()).alias("avg_discount_pct")
        ])
        
        return join_user_features(self.all_users, [price_affinity_features])
    
    def calculate_inter_session_return_features(self) -> pl.DataFrame:
        """Calculate inter-session gap and return pattern features"""
        print("Calculating inter-session return features...")
        
        session_gaps = self._session_data.sort(["user_id", "session_start"]).with_columns([
            pl.col("session_start").shift(-1).over("user_id").alias("next_session_start")
        ]).with_columns([
            (pl.col("next_session_start") - pl.col("session_end")).dt.total_hours().alias("gap_to_next_hours")
        ]).filter(
            pl.col("gap_to_next_hours").is_not_null() & (pl.col("gap_to_next_hours") > 0)
        )
        
        if session_gaps.height == 0:
            # Return empty features for users with no session transitions
            return self.all_users.with_columns([
                pl.lit(0.0).alias("median_inter_session_gap_hours"),
                pl.lit(0.0).alias("inter_session_gap_variance"),
                pl.lit(0).alias("session_transitions"),
                pl.lit(0.0).alias("same_day_return_rate"),
                pl.lit(0.0).alias("weekly_return_rate")
            ])
        
        inter_session_features = session_gaps.group_by("user_id").agg([
            pl.col("gap_to_next_hours").median().alias("median_inter_session_gap_hours"),
            pl.col("gap_to_next_hours").std().alias("inter_session_gap_variance"),
            pl.len().alias("session_transitions"),
            (pl.col("gap_to_next_hours") < 24).mean().alias("same_day_return_rate"),
            (pl.col("gap_to_next_hours") > 168).mean().alias("weekly_return_rate")
        ])
        
        return join_user_features(self.all_users, [inter_session_features])
    
    def calculate_all_features(self) -> pl.DataFrame:
        """Calculate all simplified features"""
        print("CALCULATING SIMPLIFIED FEATURES")
        print("="*60)
        
        feature_functions = [
            self.calculate_transactional_features,
            self.calculate_price_affinity_features,
            self.calculate_inter_session_return_features
        ]
        
        feature_dfs = []
        for func in feature_functions:
            try:
                feature_df = func()
                feature_dfs.append(feature_df)
                print(f"✅ {func.__name__}: {feature_df.shape[1]-1} features")
            except Exception as e:
                print(f"❌ {func.__name__}: Error - {e}")
        
        # Combine all features
        print("\nCombining all feature sets...")
        master_features = self.all_users
        for feature_df in feature_dfs:
            master_features = master_features.join(feature_df, on="user_id", how="left")
        
        master_features = master_features.fill_null(0)
        
        print(f"FINAL FEATURE SET: {master_features.shape[1]-1} features for {master_features.shape[0]:,} users")
        return master_features

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_simplified_features(df: pl.DataFrame, analysis_date: Optional[str] = None) -> pl.DataFrame:
    """Convenience function to create simplified features"""
    feature_eng = SimplifiedFeatureEngineering(df, analysis_date)
    return feature_eng.calculate_all_features()

def get_feature_definitions() -> Dict[str, List[str]]:
    """Return the definitions of feature categories"""
    return {
        "transactional_features": [
            "total_purchases",          # Number of purchase events
            "total_spend",              # Sum of all purchase amounts
            "median_order_value",       # Median purchase amount
            "price_sensitivity_score",  # Standard deviation of purchase prices
            "purchase_frequency_per_day", # Purchases per active day
            "price_range",              # Range of purchase prices
        ],
        
        "price_affinity_features": [
            "premium_product_affinity", # % of purchases in top 10% price range
            "budget_product_affinity",  # % of purchases in bottom 25% price range
            "discount_affinity",        # % of purchases below normal product price
            "avg_discount_pct"          # Average discount received on discounted purchases
        ],
        
        "inter_session_return_features": [
            "median_inter_session_gap_hours", # Median hours between sessions
            "inter_session_gap_variance",     # Standard deviation of session gaps
            "session_transitions",            # Number of session-to-session transitions
            "same_day_return_rate",           # % of returns within 24 hours
            "weekly_return_rate"              # % of returns after 1 week
        ]
    }

if __name__ == "__main__":
    # Example usage
    print("Simplified Feature Engineering Script")
    print("Feature categories available:")
    
    feature_defs = get_feature_definitions()
    for category, features in feature_defs.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  - {feature}")
