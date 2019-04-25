## Pipeline pour Logistic Regression

Après avoir trouvé le meilleur modèle avec ses paramètres calibrés pour avoir la meilleure prédiction possible des avis, nous lui avons créer un pipeline. Celui ci nous permettra d'enchaîner les étapes de traitement précedente.

### Résultats

Afin d'évaluer notre modèle, nous avons utilisé le dataset du challenge et un dataset tierce contenant des reviews de IMDB. 

Nous obtenons des résultats correctes malgré le manque de traitemnt de l'ironie. Ceci est montré par la valeur de l'accuracy de 89% et par le nombre de documents classifiés correctement pour les deux classes comme nous pouvons le voir dans la matrice de confusion. De plus, en utilisant d'autres mesures d'évaluation [pointé vers le tableau sur l'écran], notre modèle obtient des scores qui affirment sa fiabilité et sa performance.

## Pipeline pour Gaussian Naive Bayes

Nous avons décidé de créer un pipeline supplémentaire afin de visualiser l'effet d'utiliser un modèle probabiliste dans le cadre de l'analyse des sentiments. Ce modèle est intéressant car il n'a pas besoin de calibrage de ses hyperparamètres. En effet, celui s'adapte dynamiquement aux données lors de l'apprentissage.

Nous retrouvons le même procédé de création du pipeline que précédemment à défaut que ce modèle travaille sur des matrices denses pour son apprentissage. Pour ce faire, nous avons utiliser un transformateur de matrice creuses fourni par la vectorisation de nos documents afin de l'adapter aux prochaines étapes. 

### Résultats

En raison de son extraction de features indépendantes les unes des autres, les performances de ce modèle est moindre et nous pouvons le constater par les résultats obtenus sur les deux datasets 