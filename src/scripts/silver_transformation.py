import os
import duckdb

parquet_dir = './src/data/bronze'
parquet_files = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir)])

# création d'un nouveau fichier parquet silver à partir des fichiers bronze
output_silver_parquet = './src/data/silver/2019-events.parquet'

con = duckdb.connect()

# On filtre les lignes uniques où le user_id à au moins 4 events
# et où le prix est supérieur à 0
con.execute(f"""
    COPY (
        SELECT *
        FROM parquet_scan('{parquet_dir}/2019*.parquet')
        WHERE user_id IN (
            SELECT user_id
            FROM parquet_scan('{parquet_dir}/2019*.parquet')
            GROUP BY user_id
            HAVING COUNT(*) >= 4
        )
        AND price > 0
    )
    TO '{output_silver_parquet}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
""")

con.close()
