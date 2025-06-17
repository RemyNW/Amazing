# Amazing – Segmentation clients (projet data)

- **But** : extraire des segments clients homogènes à partir des logs (visites, paniers, achats) pour guider le marketing.  
- **Données** : événements pseudonymisés (~56 Go) stockés sur S3.  
- **Pipeline** : Spark/Polars ➜ nettoyage (Parquet) ➜ features ➜ clustering (scikit-learn).
