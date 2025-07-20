import os
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import time
from tqdm import tqdm
import gc

input_folder = "/data/raw"
output_folder = "/data/bronze"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

start_time = time.time()

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024**2)


def process_file_streaming(file):
    """Ultra-conservative streaming approach for very large files"""

    try:
        path = os.path.join(input_folder, file)
        file_name = file.replace('.csv.gz', '')
        output_path = os.path.join(output_folder, f"{file_name}.parquet")

        if not os.path.exists(path):
            return f"Erreur: Fichier non trouvé {path}"

        if os.path.exists(output_path):
            return f"Fichier déjà existant, ignoré: {output_path}"

        file_size = get_file_size_mb(path)
        print(f"  Traitement en streaming de {file} ({file_size:.1f} MB)")

        # Very conservative settings for large files
        read_options = pa_csv.ReadOptions(
            use_threads=False,  # Disable threading to save memory
            block_size=2<<20,  # 2MB blocks (very small)
        )

        # Create streaming reader
        csv_reader = pa_csv.open_csv(
            path,
            read_options=read_options,
            parse_options=pa_csv.ParseOptions(delimiter=','),
            convert_options=pa_csv.ConvertOptions(check_utf8=False),
        )

        # Create Parquet writer
        writer = None
        total_rows = 0

        # Process in very small batches
        batch_num = 0
        for batch in csv_reader:
            batch_num += 1
            batch_rows = len(batch)
            total_rows += batch_rows

            if batch_num % 10 == 0:  # Show progress every 10 batches
                print(f"    Batch {batch_num} ({batch_rows} lignes, total: {total_rows:,})")

            if writer is None:
                # Initialize writer with first batch schema
                try:
                    writer = pq.ParquetWriter(
                        output_path,
                        batch.schema,
                        compression='zstd',
                        use_dictionary=True,
                        write_statistics=False,  # Skip statistics to save memory
                    )
                except Exception as e:
                    return f"✗ Erreur création writer: {str(e)}"

            # Write batch
            try:
                writer.write_batch(batch)
            except Exception as e:
                if writer:
                    writer.close()
                return f"✗ Erreur écriture batch {batch_num}: {str(e)}"

            # Aggressive cleanup
            del batch
            gc.collect()

        if writer:
            writer.close()

        output_size = get_file_size_mb(output_path)
        compression_ratio = file_size / output_size if output_size > 0 else 0

        return f"✓ {file_name}.parquet ({total_rows:,} lignes, {output_size:.1f} MB, {compression_ratio:.1f}x)"

    except Exception as e:
        gc.collect()
        return f"✗ Erreur streaming de {file}: {str(e)}"

# Check if input folder exists
if not os.path.exists(input_folder):
    print(f"Erreur: Le dossier d'entrée '{input_folder}' n'existe pas")
    exit(1)

# Only process .csv.gz files
csv_gz_files = [f for f in os.listdir(input_folder) if f.endswith('.csv.gz')]

if not csv_gz_files:
    print(f"Aucun fichier .csv.gz trouvé dans {input_folder}")
    exit(1)

print(f"Traitement de {len(csv_gz_files)} fichiers avec PyArrow...")

# Sort files by size (smallest first)
csv_gz_files.sort(key=lambda f: os.path.getsize(os.path.join(input_folder, f)))

# Show file sizes
print("\nTailles des fichiers:")
total_size = 0
for file in csv_gz_files:
    size_mb = get_file_size_mb(os.path.join(input_folder, file))
    total_size += size_mb
    print(f"  {file}: {size_mb:.1f} MB")

print(f"\nTaille totale: {total_size:.1f} MB")
print(f"Début du traitement...")

# Process files one by one
results = []
for i, file in enumerate(csv_gz_files, 1):
    print(f"\n[{i}/{len(csv_gz_files)}] Traitement de {file}...")

    file_size = get_file_size_mb(os.path.join(input_folder, file))

    # Use streaming for all files since they're all > 100MB
    result = process_file_streaming(file)

    results.append(result)
    print(f"  Résultat: {result}")

print(f"\n{'='*50}")
print("RÉSUMÉ:")
for result in results:
    print(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTemps d'exécution total : {execution_time:.2f} secondes")
