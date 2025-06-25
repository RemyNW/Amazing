import math
import polars as pl
from typing import Dict, List, Optional, Tuple

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def get_event_type_dataframes(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Split dataframe by event type for reuse across functions"""
    return {
        'purchase': df.filter(pl.col("event_type") == "purchase"),
        'view': df.filter(pl.col("event_type") == "view"),
        'cart': df.filter(pl.col("event_type") == "cart")
    }

def calculate_diversity_ratios(df_subset: pl.DataFrame, columns: List[str]) -> List[pl.Expr]:
    """Calculate normalized diversity ratios for given columns"""
    return [
        (pl.col(col).n_unique() / pl.len()).alias(f"{col}_diversity_ratio")
        for col in columns
    ]

def calculate_concentration_ratios(df_subset: pl.DataFrame, columns: List[str]) -> List[pl.Expr]:
    """Calculate concentration ratios (top item percentage) for given columns"""
    return [
        (pl.col(col).value_counts().struct.field("count").max() / pl.len()).alias(f"{col}_concentration")
        for col in columns
    ]

def calculate_event_type_ratios(event_types: List[str]) -> List[pl.Expr]:
    """Calculate ratios for different event types"""
    return [
        (pl.col("product_id").filter(pl.col("event_type") == event_type).n_unique() / pl.col("product_id").n_unique()).alias(f"{event_type}_ratio")
        for event_type in event_types
    ]

def calculate_conversion_rates(conversions: List[Tuple[str, str]]) -> List[pl.Expr]:
    """Calculate conversion rates between event types"""
    return [
        (pl.col("product_id").filter(pl.col("event_type") == target).n_unique() /
         pl.col("product_id").filter(pl.col("event_type") == source).n_unique().clip(1)).clip(0, 1).alias(f"{source}_to_{target}_rate")
        for source, target in conversions
    ]

def calculate_recency_features(df_subset: pl.DataFrame, analysis_date, event_name: str) -> List[pl.Expr]:
    """Calculate recency features for a specific event type"""
    return [
        (pl.lit(analysis_date) - pl.col("event_time").max()).dt.total_days().alias(f"{event_name}_recency_days").fill_null(-1)
    ]

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

def calculate_entropy(values_expr: pl.Expr) -> pl.Expr:
    """Calculate entropy of a distribution"""
    return values_expr.map_elements(
        lambda x: -sum([(p/sum(x)) * math.log(p/sum(x)) for p in x if p > 0]) if sum(x) > 0 else 0, 
        return_dtype=pl.Float64
    )

def join_user_features(all_users: pl.DataFrame, feature_dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """Join multiple feature dataframes to all users"""
    result = all_users
    for feature_df in feature_dfs:
        result = result.join(feature_df, on="user_id", how="left")
    return result.fill_null(0)

# ============================================================================
# CORE FEATURE FUNCTIONS
# ============================================================================

class FeatureEngineering:
    """Modular feature engineering class with reusable components"""
    
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
    
    def calculate_engagement_features(self) -> pl.DataFrame:
        """Calculate engagement and activity features"""
        print("Calculating engagement features...")
        
        # Basic metrics
        basic_features = self.df.group_by("user_id").agg([
            pl.len().alias("total_events"),
            pl.col("user_session").n_unique().alias("unique_sessions"),
            pl.col("event_time").dt.date().n_unique().alias("days_active"),
            
            # Event type ratios
            *calculate_event_type_ratios(["view", "cart", "purchase"]),
            
            # Conversion rates
            *calculate_conversion_rates([("cart", "purchase"), ("view", "purchase")]),
            
            # Temporal patterns
            pl.col("event_time").dt.hour().mode().first().alias("peak_activity_hour"),
            pl.col("event_time").dt.weekday().mode().first().alias("peak_activity_weekday"),
            
            # Session characteristics
            (pl.len() / pl.col("user_session").n_unique()).alias("avg_events_per_session"),
            
            # Activity regularity
            pl.col("event_time").dt.date().value_counts().struct.field("count").std().alias("activity_regularity")
        ])
        
        # Session duration features
        session_duration_features = self._session_data.with_columns([
            (pl.col("session_end") - pl.col("session_start")).dt.total_minutes().alias("session_duration_minutes")
        ]).group_by("user_id").agg([
            pl.col("session_duration_minutes").mean().alias("avg_session_duration_minutes"),
            pl.col("session_duration_minutes").std().alias("session_duration_variance")
        ])
        
        return join_user_features(self.all_users, [basic_features, session_duration_features])
    
    def calculate_purchase_features(self) -> pl.DataFrame:
        """Calculate purchase behavior features"""
        print("Calculating purchase features...")
        
        purchase_df = self.event_dfs['purchase']
        if purchase_df.height == 0:
            return self.all_users.with_columns(pl.lit(0).alias("no_purchases"))
        
        # Basic purchase metrics
        basic_purchase_features = purchase_df.group_by("user_id").agg([
            pl.len().alias("total_purchases"),
            pl.col("price").sum().alias("total_spend"),
            pl.col("price").mean().alias("avg_order_value"),
            pl.col("price").median().alias("median_order_value"),
            pl.col("price").std().alias("price_sensitivity_score"),
            pl.col("price").min().alias("min_purchase_price"),
            pl.col("price").max().alias("max_purchase_price"),
            
            # Frequency normalized by active days
            (pl.len() / pl.col("event_time").dt.date().n_unique()).alias("purchase_frequency_per_day"),
            
            # Diversity and concentration
            *calculate_diversity_ratios(purchase_df, ["product_id", "category_id", "brand"]),
            (pl.len() / pl.col("product_id").n_unique()).alias("avg_purchases_per_product"),
            
            # Loyalty indicators
            (pl.len() > 1).cast(pl.Int32).alias("is_repeat_purchaser"),
            (pl.col("product_id").value_counts().struct.field("count").max() > 1).cast(pl.Int32).alias("has_product_loyalty"),
            (pl.col("event_time").max() - pl.col("event_time").min()).dt.total_days().alias("purchase_span_days")
        ])
        
        # Price affinity features
        price_affinity_features = self._calculate_price_affinity_features()
        
        return join_user_features(self.all_users, [basic_purchase_features, price_affinity_features])
    
    def _calculate_price_affinity_features(self) -> pl.DataFrame:
        """Calculate price affinity features using percentiles"""
        price_90th, price_25th, product_normal_prices = self._price_percentiles
        
        if self.event_dfs['purchase'].height == 0:
            return pl.DataFrame({"user_id": [], "premium_product_affinity": []})
        
        purchase_with_normal = self.event_dfs['purchase'].join(product_normal_prices, on="product_id", how="left")
        
        return purchase_with_normal.group_by("user_id").agg([
            (pl.col("price").filter(pl.col("price") >= price_90th).len() / pl.len()).alias("premium_product_affinity"),
            (pl.col("price").filter(pl.col("price") <= price_25th).len() / pl.len()).alias("budget_product_affinity"),
            (pl.col("price").filter(pl.col("price") < pl.col("normal_price")).len() / pl.len()).alias("discount_affinity"),
            ((100 * pl.col("price") / pl.col("normal_price")).filter(pl.col("price") < pl.col("normal_price")).mean()).alias("avg_discount_pct")
        ])
    
    def calculate_product_preference_features(self) -> pl.DataFrame:
        """Calculate product and category preference features"""
        print("Calculating product preference features...")
        
        return self.df.group_by("user_id").agg([
            # Diversity metrics
            *calculate_diversity_ratios(self.df, ["category_id", "brand", "product_id"]),
            
            # Concentration metrics
            *calculate_concentration_ratios(self.df, ["category_id", "brand"]),
            
            # Price preferences
            pl.col("price").filter(pl.col("event_type") == "purchase").mean().alias("avg_purchase_price_overall"),
            
            # Conversion efficiency
            (pl.col("product_id").filter(pl.col("event_type") == "purchase").n_unique() /
             pl.col("category_id").filter(pl.col("event_type") == "view").n_unique().clip(1)).alias("category_purchase_efficiency")
        ])
    
    def calculate_temporal_features(self) -> pl.DataFrame:
        """Calculate temporal behavior features"""
        print("Calculating temporal features...")
        
        temporal_features = self.df.group_by("user_id").agg([
            # Peak patterns
            pl.col("event_time").dt.hour().mode().first().alias("peak_activity_hour"),
            pl.col("event_time").dt.weekday().mode().first().alias("peak_activity_weekday"),
            
            # Circular hour statistics - calculated directly here to avoid column duplication
            (pl.col("event_time").dt.hour() * 2 * math.pi / 24).sin().mean().alias("hour_sin_mean"),
            (pl.col("event_time").dt.hour() * 2 * math.pi / 24).cos().mean().alias("hour_cos_mean"),
            
            # Activity spread
            (pl.col("event_time").dt.hour().max() - pl.col("event_time").dt.hour().min()).alias("activity_hour_range"),
            pl.col("event_time").dt.hour().n_unique().alias("unique_active_hours"),
            
            # Weekday vs weekend
            (pl.col("event_time").dt.weekday().filter(pl.col("event_time").dt.weekday() < 5).len() / pl.len()).alias("weekday_activity_ratio"),
            (pl.col("event_time").dt.weekday().filter(pl.col("event_time").dt.weekday() >= 5).len() / pl.len()).alias("weekend_activity_ratio"),
            
            # Activity consistency
            (pl.col("event_time").dt.date().value_counts().struct.field("count").std() / 
             pl.col("event_time").dt.date().value_counts().struct.field("count").mean()).alias("daily_activity_cv"),
            
            # Time span
            pl.col("event_time").dt.date().n_unique().alias("total_active_days"),
            (pl.col("event_time").max() - pl.col("event_time").min()).dt.total_days().alias("activity_span_days"),
            
            # Distribution entropy and concentration
            calculate_entropy(pl.col("event_time").dt.hour().value_counts().struct.field("count")).alias("hour_distribution_entropy"),
            calculate_entropy(pl.col("event_time").dt.weekday().value_counts().struct.field("count")).alias("day_distribution_entropy"),
            (pl.col("event_time").dt.hour().value_counts().struct.field("count").max() / pl.len()).alias("peak_hour_concentration"),
            (pl.col("event_time").dt.weekday().value_counts().struct.field("count").max() / pl.len()).alias("peak_day_concentration")
        ])
        
        # Calculate circular mean hour
        return temporal_features.with_columns([
            pl.arctan2(pl.col("hour_sin_mean"), pl.col("hour_cos_mean")).alias("circular_mean_hour_radians")
        ]).with_columns([
            ((pl.col("circular_mean_hour_radians") * 24 / (2 * math.pi)) % 24).alias("circular_mean_hour")
        ]).drop(["hour_sin_mean", "hour_cos_mean", "circular_mean_hour_radians"])
    
    def calculate_rfm_features(self) -> pl.DataFrame:
        """Calculate RFM features with improved modularity"""
        print("Calculating RFM features...")
        
        # Activity span for normalization (shared calculation)
        activity_spans = self.df.group_by("user_id").agg([
            (pl.col("event_time").max() - pl.col("event_time").min()).dt.total_days().clip(1).alias("activity_span_days")
        ])
        
        # RFM for each event type
        rfm_features = []
        for event_type, event_df in self.event_dfs.items():
            if event_df.height > 0:
                event_rfm = event_df.group_by("user_id").agg([
                    *calculate_recency_features(event_df, self.analysis_date, event_type),
                    pl.len().alias(f"{event_type}_frequency_raw")
                ])
                
                # Add monetary value for purchase events
                if event_type == "purchase":
                    purchase_monetary = event_df.group_by("user_id").agg([
                        pl.col("price").sum().alias("monetary_value")
                    ])
                    event_rfm = event_rfm.join(purchase_monetary, on="user_id", how="left")
                
                rfm_features.append(event_rfm)
        
        # Combine and normalize frequencies
        rfm_combined = join_user_features(self.all_users, rfm_features + [activity_spans])
        
        # Normalize frequencies by activity span
        frequency_columns = [f"{event_type}_frequency_raw" for event_type in self.event_dfs.keys()]
        normalized_columns = [
            (pl.col(col) / pl.col("activity_span_days")).alias(col.replace("_raw", "_per_day"))
            for col in frequency_columns if col in rfm_combined.columns
        ]
        
        if normalized_columns:
            rfm_combined = rfm_combined.with_columns(normalized_columns)
            rfm_combined = rfm_combined.drop([col for col in frequency_columns if col in rfm_combined.columns])
        
        return rfm_combined.fill_null(999)  # High recency for inactive users
    
    def calculate_behavioral_features(self) -> pl.DataFrame:
        """Calculate advanced behavioral features including inter-session patterns"""
        print("Calculating behavioral features...")
        
        # Calculate abandoned carts
        abandoned_carts_features = self.df.group_by(["user_id", "product_id"]).agg([
            pl.col("event_type").filter(pl.col("event_type") == "cart").len().alias("cart_events_per_product"),
            pl.col("event_type").filter(pl.col("event_type") == "purchase").len().alias("purchase_events_per_product")
        ]).with_columns([
            (pl.col("cart_events_per_product") - pl.min_horizontal("cart_events_per_product", "purchase_events_per_product")).alias("abandoned_carts_per_product")
        ]).group_by("user_id").agg([
            pl.col("abandoned_carts_per_product").sum().alias("abandoned_carts"),
            pl.col("cart_events_per_product").sum().alias("total_cart_events")
        ])
        
        # Core behavioral patterns
        behavioral_features = self.df.sort(["user_id", "event_time"]).group_by("user_id").agg([
            # Research intensity
            (pl.col("product_id").filter(pl.col("event_type") == "purchase").n_unique() /
             pl.col("product_id").filter(pl.col("event_type") == "view").n_unique().clip(1)).alias("product_research_intensity"),
            (pl.col("event_type").filter(pl.col("event_type") == "purchase").len() /
             pl.col("event_type").filter(pl.col("event_type") == "view").len().clip(1)).alias("purchase_to_browse_ratio"),
            
            # Session patterns
            pl.col("user_session").n_unique().alias("total_sessions"),
            (pl.col("user_session").filter(pl.col("event_type") == "purchase").n_unique() /
             pl.col("user_session").n_unique()).alias("session_conversion_rate"),
            (pl.col("product_id").n_unique() / pl.len()).alias("product_exploration_rate"),
        ])
        
        # Combine abandoned carts calculation with other behavioral features
        behavioral_combined = join_user_features(self.all_users, [abandoned_carts_features, behavioral_features]).with_columns([
            (pl.col("abandoned_carts") / pl.col("total_cart_events").clip(1)).alias("cart_abandonment_rate")
        ])
        
        # Add browse-to-buy, multi-session, and inter-session features
        browse_to_buy_features = self._calculate_browse_to_buy_features()
        multi_session_features = self._calculate_multi_session_features()
        inter_session_features = self._calculate_inter_session_features()
        
        return join_user_features(self.all_users, [behavioral_combined, browse_to_buy_features, multi_session_features, inter_session_features])
    
    def _calculate_browse_to_buy_features(self) -> pl.DataFrame:
        """Calculate browse-to-buy time features"""
        purchase_df = self.event_dfs['purchase']
        view_df = self.event_dfs['view']
        
        if purchase_df.height == 0 or view_df.height == 0:
            return pl.DataFrame({"user_id": []})
        
        # Find first view time for each user-product pair
        first_views = view_df.group_by(["user_id", "product_id"]).agg([
            pl.col("event_time").min().alias("first_view_time")
        ])
        
        # Join with purchases and calculate times
        purchase_with_views = purchase_df.join(first_views, on=["user_id", "product_id"], how="left")
        
        return purchase_with_views.with_columns([
            (pl.col("event_time") - pl.col("first_view_time")).dt.total_minutes().alias("browse_to_buy_minutes")
        ]).filter(
            pl.col("browse_to_buy_minutes").is_not_null() & (pl.col("browse_to_buy_minutes") >= 0)
        ).group_by("user_id").agg([
            pl.col("browse_to_buy_minutes").mean().alias("avg_browse_to_buy_minutes"),
            pl.col("browse_to_buy_minutes").median().alias("median_browse_to_buy_minutes"),
            (pl.col("browse_to_buy_minutes") < 60).mean().alias("quick_decision_rate"),
            pl.len().alias("researched_purchases")
        ])
    
    def _calculate_multi_session_features(self) -> pl.DataFrame:
        """Calculate multi-session purchase journey features"""
        return self._session_data.with_columns([
            (pl.col("purchases_in_session") > 0).cast(pl.Int32).alias("is_purchase_session"),
            ((pl.col("purchases_in_session") > 0) & (pl.col("views_in_session") > 0)).cast(pl.Int32).alias("is_research_purchase_session")
        ]).group_by("user_id").agg([
            pl.col("is_purchase_session").sum().alias("total_purchase_sessions"),
            pl.col("is_research_purchase_session").sum().alias("research_purchase_sessions"),
            (pl.col("is_research_purchase_session").sum() / pl.col("is_purchase_session").sum().clip(1)).alias("multi_session_journey_rate")
        ])
    
    def _calculate_inter_session_features(self) -> pl.DataFrame:
        """Calculate inter-session gap and return pattern features"""
        session_gaps = self._session_data.sort(["user_id", "session_start"]).with_columns([
            pl.col("session_start").shift(-1).over("user_id").alias("next_session_start")
        ]).with_columns([
            (pl.col("next_session_start") - pl.col("session_end")).dt.total_hours().alias("gap_to_next_hours")
        ]).filter(
            pl.col("gap_to_next_hours").is_not_null() & (pl.col("gap_to_next_hours") > 0)
        )
        
        return session_gaps.group_by("user_id").agg([
            pl.col("gap_to_next_hours").mean().alias("avg_inter_session_gap_hours"),
            pl.col("gap_to_next_hours").median().alias("median_inter_session_gap_hours"),
            pl.col("gap_to_next_hours").std().alias("inter_session_gap_variance"),
            (pl.col("gap_to_next_hours") < 24).mean().alias("same_day_return_rate"),
            (pl.col("gap_to_next_hours") > 168).mean().alias("weekly_return_rate"),
            pl.len().alias("session_transitions")
        ])
    
    def calculate_all_features(self) -> pl.DataFrame:
        """Calculate all features using the modular approach"""
        print("CALCULATING ALL FEATURES")
        print("="*60)
        
        feature_functions = [
            self.calculate_engagement_features,
            self.calculate_purchase_features,
            self.calculate_product_preference_features,
            self.calculate_temporal_features,
            self.calculate_rfm_features,
            self.calculate_behavioral_features
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