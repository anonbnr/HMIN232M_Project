## Pretraitement

### Préparation à la tokenisation

Afin de préparer nos documents à la tokenisation, nous avons commencé par remplacer les contractions et supprimer les balises HTML et nottament les URL. Nous avions remarqué qu'à la suppresion des ponctuations, nous nous retrouvons avec une concaténation de mots qui au final n'avait aucun sens. Cela était du à un

### Tokenisation et normalisation

Par la suite, nous avons tokeniser nos documents afin d'effectuer des traitements de nettoyage sur les termes. 

### Visualisation de la donnée

Pour visualiser les données que nous traitons, nous avons utilisé le module WordCloud qui permet d'afficher les mots les plus fréquents parmi nos deux classes. Nous avons remarqué qu'il y avait des mots positifs très fréquents dans les avis négatifs. On peut en tirer donc la conclusion d'une forte utilisation d'ironie ou bien de longs avis avec plusieurs phrases à polaritées différentes. 

### Extraction des features

Une fois nos données normalisées, nous avons besoin de les transformer en données numériques afin d'être traité par la suite. Pour cela, nous avons utilisé TF-IDF pour la vectorisation qui assigne à chaque feature la fréquence de ce terme multiplié par l'inverse de sa fréquence dans le corpus.

A celà, on a ajouté l'utilisation de n-gram qui nous permet d'extraire deux mots consécutives en tant que feature. Cette technique nous apporte une solution au traitement de la négation des termes.

Enfin, pour améliorer la précision d'extraction de features pertinentes, nous avons filtrer tous les termes dont leur occurence dans le corpus est supérieur ou égale à 12.