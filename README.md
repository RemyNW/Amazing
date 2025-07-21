# Amazing – Segmentation clients (projet data)

- **But** : extraire des segments clients homogènes à partir des logs (visites, paniers, achats) pour guider le marketing.
- **Données** : événements pseudonymisés (~56 Go) stockés sur S3.
- **Pipeline** : Spark/Polars ➜ nettoyage (Parquet) ➜ features ➜ clustering (scikit-learn).

## Procédure d'installation

### Installation de Python (Ubuntu)

```shell
sudo apt update
sudo apt install python3.10
```

### Récupération du repository

Cloner ce repository :
```shell
git clone git@github.com:RemyNW/Amazing.git
```

Se mettre dans le dossier `Amazing` :
```shell
cd Amazing
```

### Mise en place de l'environnement

Initialiser l'environnement virtuel et se déplacer dans le dossier `Amazing` :
```shell
pip install pyenv && pyenv virtualenv 3.12.10 amazing && echo "amazing" > Amazing/.python-version && cd Amazing
```

#### Variables d'environnement et librairies python

Copier le template d'environnement et le remplir avec les paramètres souhaités :
```shell
cp .env.template .env && sed -i "s|PATH_DATA=.*|PATH_DATA=$(pwd)/src/data|" .env
```

Installer les librairies python :
```shell
pip install -r requirements.txt
```

## Procédure d'exécution

TODO
