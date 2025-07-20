import os
import duckdb

parquet_dir = './src/data/bronze'
parquet_files = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir)])

# création d'un nouveau fichier parquet silver à partir des fichiers bronze
year = 2019
month = '*' # * pour tous les mois
output_silver_parquet = f'./src/data/silver/{year}-{month if month != "*" else "all"}-events.parquet'

nb_max_events = 4

con = duckdb.connect()

# On filtre les lignes uniques où le user_id à au moins 4 events
# et où le prix est supérieur à 0
con.execute(f"""
    COPY (
        SELECT *
        FROM parquet_scan('{parquet_dir}/{year}-*.parquet')
        WHERE user_id IN (
            SELECT user_id
            FROM parquet_scan('{parquet_dir}/{year}-*.parquet')
            GROUP BY user_id
            HAVING COUNT(*) >= {nb_max_events}
        )
        AND price > 0
    )
    TO '{output_silver_parquet}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
""")

con.close()
