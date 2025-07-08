Veille Modèle

# **Introduction :**

Dans ce projet, notre but est de mieux comprendre les différents types de clients qui utilisent la plateforme Amazing Basics. On ne cherche pas à prédire une valeur (comme le prix d’un achat) ou à classer les clients dans des catégories déjà connues. Au contraire, on veut découvrir automatiquement des groupes de clients qui se comportent de manière similaire, sans avoir besoin d’étiquettes au départ.

# **Présentation des données :**

- Nous possédons des données permettant d’identifier le client (user_id).
- Des données relatives aux produits qui ont suscité son intérêt (product_id, category_id, category_code, brand, price).
- Des données relatives à l’action effectuée avec ce produit (event_type).

**Type :** Les données auxquelles nous avons accès sont des fichiers au format CSV.

Il y a **7 fichiers CSV compressés**.

**Couverture temporelle :** Les données couvrent les mois d’octobre 2019 à avril 2020. Un fichier CSV correspond à **un mois** de données de cet ensemble.

| **Estimation du volume de données** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **Fichier** | **Taille Compressé (Gb)** | **Estimation du Nombre de Lignes** | **Estimation Taille Décompressé (Gb)** |
| 2019-Oct | 1.62 | 42448765 | 5.52 |
| 2019-Nov | 2.69 | 70485912 | 9.16 |
| 2019-Dev | 2.74 | 71796059 | 9.33 |
| 2020-Jan | 2.23 | 58432559 | 7.60 |
| 2020-Feb | 2.19 | 57384442 | 7.46 |
| 2020-Mar | 2.25 | 58956618 | 7.66 |
| 2020-Apr | 2.73 | 71534030 | 9.3 |
| **Total** | 16.45 | 431038385 | 56.03 |

# **Typologie des modèles IA :**

## **La Régression :**

### Définition simple

La **régression** est une technique de machine learning **supervisée** qui permet de **prédire une valeur numérique continue** à partir d’autres variables (appelées _features_ ou _caractéristiques_).

On l’utilise quand la **sortie attendue n’est pas une catégorie**, mais une **valeur chiffrée**.

### Modèles classiques de régression

Régression linéaire, Régression polynomiale, Ridge / Lasso, SVR, Arbres de décision régressifs, Random Forest Regressor et XGBoost,...

### Metriques

- **MAE – Mean Absolute Error (Erreur absolue moyenne) :** Moyenne des valeurs absolues des écarts entre les vraies valeurs et les valeurs prédites. La MAE mesure l’erreur moyenne que fait le modèle, en unités réelles (ex : euros, minutes).  

- **MSE – Mean Squared Error (Racine de l’erreur quadratique moyenne) :** Moyenne des carrés des erreurs entre les prédictions et les vraies valeurs. Les grandes erreurs sont plus fortement pénalisées, ce qui peut rendre ce score plus sensible aux anomalies.  

- **R² – Coefficient de détermination :** Permet de voir si le modèle explique bien ce qu’il essaie de prédire.

### Compatibilité avec le projet Amazing

La régression ne répond pas à l’objectif de segmentation ou modélisation de profils clients car le but de la régression est de prédire une valeur numérique continue, tandis que Amazing cherche à mieux identifier des groupes de clients afin de fournir à l’équipe parketing des clés de lecture sur les comportements clients.

## **La classification:**

### Définition simple

La classification consiste à prédire une **catégorie ou un label discret**. Par exemple, dire si un client achètera un produit (oui/non) ou à quel segment il appartient (fidèle, occasionnel, à risque).

### Modèles classiques de la classification

Naive Bayes, Linear SVC, KNeighbors, SGD Classifier…

### Métriques

- **Accuracy (Exactitude) :** Elle permet de montrer la performance globale du modèle : Proportion de bonnes prédictions parmi toutes les prédictions.  

- **Précision :** Elle mesure la fiabilité des prédictions positives : Parmi les éléments prédits positifs, combien sont corrects (vrai positifs).  

- **Recall (Sensibilité) :** Mesure la capacité à retrouver tous les cas positifs : Parmi les éléments réellement positifs, combien sont correctement trouvés.  

- **F1-score :** Moyenne entre précision et rappel : Permet de trouver un équilibre entre faux positifs et faux négatifs.  

- **Matrice de confusion :** Permet de visualiser les vrais positifs, faux positifs, vrais négatifs, faux négatifs.  

- **ROC AUC :** Courbe qui mesure la capacité du modèle à distinguer entre classes (valeurs proches de 1 = excellent).

### Compatibilité avec le projet Amazing

Pour utiliser la classification, il faut **avoir des étiquettes** (labels) précises sur chaque client, ce qui n’est pas le cas ici (pas de catégories clients définies). Sans les données étiquetées, la classification ne peut pas être utilisée

## **La Réduction de dimensionnalité :**

### Définition simple

La réduction de dimensionnalité est une méthode visant à simplifier un jeu de données en réduisant le nombre de variables tout en conservant l’essentiel de l’information. Cela facilite l’analyse, la visualisation et le traitement des données complexes.

### Modèles classiques de la réduction de dimensionnalité

Analyse en composantes principales (ACP), IsoMap, Spectral Embbeding, …

### Metriques

- **Explained Variance (notamment pour l’ACP)** : % de la variance conservée dans les composantes principales. Elle permet de vérifier si la réduction ne supprime pas trop d’informations utiles.  

- **KL Divergence :** Mesure la perte d’information due à une simplification.  

- **Reconstruction Error :** Mesure ce que la réduction fait perdre.

### Compatibilité avec le projet Amazing

La réduction de dimensionnalité n’est pas une méthode de segmentation ou de modélisation de clients types. Elle ne permet ni de prédire une catégorie, ni de former des groupes à elle seule. Son rôle est de simplifier les données, en réduisant le nombre de variables, pour aider d'autres modèles comme le clustering ou la régression à mieux fonctionner.

## **Le Clustering :**

### Définition simple

Le clustering est une technique d’apprentissage **non supervisé** qui consiste à regrouper des objets similaires en groupes ou clusters, sans avoir besoin de labels préalables. L'objectif principal du regroupement est de :

- Simplifier les grands ensembles de données en sous-groupes significatifs.
- Identifier les regroupements naturels dans les données.

### Modèles classiques du clustering

KMeans, MiniBatch KMeans, Spectral Clustering, DBSCAN, …

### Metriques

- **Silhouette Score :** Mesure à quel point un point est proche de son propre groupe et loin des autres groupes.  

- **Davies-Bouldin Index :** Moyenne de la ressemblance entre chaque groupe et celui qui lui est le plus proche.  

- **Calinski-Harabasz Index :** Mesure combien les groupes sont concentrés en interne et séparés entre eux.  

- **Visualisation 2D/3D :** avec ACP ou t-SNE pour vérifier visuellement la qualité des clusters.  

- **Elbow Method :** Permet de trouver le **nombre optimal de groupes**.

### Compatibilité avec le projet Amazing

Le clustering est parfaitement adapté pour modéliser des clients types dans le cadre de Amazing. En l’absence d’étiquettes, il permet d’identifier des groupes de clients aux comportements similaires, ce qui va permettre une meilleure compréhension des profiles clients sur la plateforme.

# **Le choix du modèle de Clustering :**

Dans notre projet, on cherche à développer un modèle capable de catégoriser les clients selon leurs habitudes d’achats, de navigation et d’interaction avec les produits. Nous possédons des données permettant d’identifier le client, ainsi que les produits qui ont suscité son intérêt. On va donc définir un nombre de segment (cluster) de clients à partir des données disponibles. Notre but sera d’identifier des clusters de clients similaires.

Il existe deux catégories de modèles de clustering :

1. **Les modèles à nombre de clusters inconnu :**  
    Ces modèles déterminent automatiquement le nombre de clusters en fonction de la structure des données.  
    **Exemple :** DBSCAN, Mean Shift.  

2. **Les modèles à nombre de clusters défini :**  
    Ces modèles nécessitent de spécifier à l’avance le nombre de segments (ou clusters) que l’on souhaite identifier dans les données.  
    **Exemple :** K-Means, MiniBatch K-Means, Spectral Clustering, GMM..

## **Modèle à nombre de clusters inconnu**

**Qu'est-ce que DBSCAN ?**

DBSCAN est un algorithme de clustering qui regroupe des points très proches les uns des autres dans l'espace de données. Avec DBSCAN on n’est pas obliger de spécifier le nombre de clusters au préalable, ce qui le rend particulièrement utile pour l'analyse exploratoire des données. L'algorithme fonctionne en définissant les clusters (groupes) comme des régions denses séparées par des régions de moindre densité. Cette approche permet à DBSCAN de découvrir des clusters de forme arbitraire et d'identifier les valeurs aberrantes comme du bruit.

**Les paramètres de DBSCAN :**

DBSCAN utilise deux paramètres principaux :

- **ε (epsilon) :** La distance maximale entre deux points pour qu'ils soient considérés comme voisins.
- **MinPts :** Le nombre minimum de points requis pour former une région dense.

DBSCAN s'articule autour de trois concepts clés :

**Points essentiels :** Il s'agit de points qui ont au moins un nombre minimum d'autres points (MinPts) à une distance spécifiée (epsilon).

**Points frontières :** Il s'agit de points qui se trouvent à une distance ε d'un point central mais qui n'ont pas de MinPts voisins eux-mêmes.

**Points de bruit :** Il s'agit de points qui ne sont ni des points centraux ni des points frontières. Ils ne sont pas assez proches d'un groupe pour être inclus.

En ajustant ces paramètres, on peut contrôler la façon dont l'algorithme définit les clusters, ce qui lui permet de s'adapter à différents types d'ensembles de données et d'exigences en matière de clusters.

**Comment fonctionne DBSCAN ?**

1. **Sélection des paramètres :**

ε (epsilon) : On définit le paramètre ε qui représente la distance maximale entre deux points pour qu'ils soient considérés comme voisins.

MinPts : On définit le nombre minimum de points requis pour former une région dense.

1. **Sélection d’un point de départ :**

L'algorithme commence par un point arbitraire dans l'ensemble de données.

1. **Recherche autour du point :**

Il récupère tous les points situés à une distance ε du point de départ.

Si le nombre de points voisins est inférieur à MinPts, le point est étiqueté comme bruit (pour l'instant).

S'il y a au moins MinPts points à moins de ε de distance, le point est marqué comme un point central et un nouveau groupe est formé.

1. **Développer le cluster :**

Tous les voisins du point central sont ajoutés au cluster.

Pour chacun de ces voisins :

S'il s'agit d'un point central, ses voisins sont ajoutés au cluster de manière récursive.

S'il ne s'agit pas d'un point central, il est marqué comme point frontalier et l'expansion s'arrête.

1. **Répéter le processus :**

L'algorithme passe au prochain point non visité de l'ensemble de données.

Les étapes 3 et 4 sont répétées jusqu'à ce que tous les points aient été visités.

1. **Finaliser les regroupements**

Une fois que tous les points ont été traités, l'algorithme identifie tous les groupes.

Les points initialement étiquetés comme étant du bruit peuvent maintenant être des points frontières s'ils se trouvent à moins de ε de distance d'un point central.

1. **Bruit de manipulation**

Les points n'appartenant à aucun cluster restent classés comme du bruit.

## **Modèle à nombre de clusters connu**

**Qu'est-ce que MiniBatch K-means ?**

L'algorithme MiniBatch K-means est une variante de l'algorithme de clustering K-means traditionnel, conçue pour gérer de grands ensembles de données. Avec l'algorithme K-means traditionnel, l'ensemble des données est traité à chaque itération, ce qui peut s'avérer coûteux en calculs pour les grands ensembles de données. Le MiniBatch K-means résout ce problème en ne traitant qu'un petit sous-ensemble de données, appelé mini-lot, à chaque itération. Ce mini-lot est échantillonné aléatoirement à partir de l'ensemble de données, et l'algorithme met à jour les centroïdes des clusters en fonction des données qu'il contient. Cela permet à l'algorithme de converger plus rapidement et d'utiliser moins de mémoire que le K-means traditionnel.

**Pourquoi MiniBatch K-means est le modèle le plus adapté à notre projet ?**

Dans notre cas, nous allons définir un nombre précis de clusters à l’avance (acheteurs réguliers, occasionnels, visiteurs passifs, etc.). Cette connaissance préalable du nombre de groupes cible nous pousse à choisir un modèle à nombre de clusters défini (K-means, MiniBatch K-means, GMM, etc…)

Entre ces options, nous avons choisi MiniBatch K-means qui est particulièrement adapté lorsque le nombre de clusters est relativement faible. MiniBatch K-means utilise des mini-lots de données pour effectuer les mises à jour des centroïdes (centres de clusters), ce qui le rend plus rapide et moins coûteux que K-means classique, tout en conservant une bonne qualité de segmentation.

**Comment fonctionne l’algorithme MiniBatch K-means ?**

Le processus de l'algorithme K-means en mini-lots peut être résumé comme suit :

1\. Choisir le nombre de clusters (groupes).  
2\. Initialisez les centroïdes des clusters de manière aléatoire.  
Répétez les étapes suivantes jusqu'à atteindre la convergence ou un nombre maximal d'itérations :  
3\. Sélectionner un mini-lot de données dans l'ensemble de données.  
4\. Affecter chaque point de données du mini-lot au centroïde du cluster le plus proche.  
5\. Mettre à jour les centroïdes des clusters en fonction des points de données qui leur sont affectés. Cela revient à recalculer la moyenne des points dans chaque groupe.  
6\. Renvoyer les centroïdes finaux des clusters et les affectations de points de données aux clusters.

# **Techniques d’évaluation :**

## **Évaluer la sensibilité à l’initialisation par la variation de l’inertie :**

L’initialisation des centroïdes est une étape clé dans l’algorithme MiniBatch K-means, car elle influence fortement la qualité de la convergence. Plusieurs stratégies d’initialisation existent, dont les plus courantes sont :

- **init="random" :** les centroïdes sont choisis aléatoirement parmi les points de données.
- **init="k-means++" :** une méthode intelligente qui sélectionne les centroïdes initiaux de manière à être bien espacés, ce qui accélère la convergence et améliore la qualité finale du clustering.

Pour évaluer l’impact de ces méthodes, on lance l’algorithme plusieurs fois avec différentes initialisations (paramètre n_init) et on observe la variabilité de la qualité du regroupement, mesurée par l’inertie (la somme des distances au carré entre les points et leur centroïde).

- Si la variation de l’inertie est faible, cela signifie que l’algorithme converge toujours vers des regroupements similaires, donc la méthode est robuste.
- Si la variation est grande, alors certains résultats sont mauvais, et l’algorithme est instable.