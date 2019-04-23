# Classification de documents d'opinions

---
## 2. Apprentissage du modèle
---

### 2.1 Vectorisation et sélection des features 

Une fois nos documents nettoyés, on a besoin de transformer les données textuelles en features ayant une probabilité en tant que valeur afin qu'elles soient interprétées dans nos les modèles par la suite.

#### Documents to matrix of TF-IDF features 

TF-IDF (Term Frequency-Inverse Document Frequency) est une méthode de pondération souvent utilisée dans la fouille de texte et nous permet d'évaluer l'importance d'un terme contenu dans un document, relativement à un corpus. Cette mesure statistique est calculée par la multiplcation du nombre d'occurences du mot dans le document par une variation de la fréquence du mot dans le corpus; et se code de la manière suivante :

```
vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df = 12)
```

#### Gestion de la négation

Le premier paramètre de cette fonction permet de prendre en compte non seulement le mot mais 2-mot, c'est-à-dire le mot et le mot qui suit en tant que feature. Cette technique permet de prendre en compte la négation dans une suite de mots du type de motif suivant `not mot`, où mot est un mot quelconque

#### Sélection des features

A la construction du vocabulaire des features, nous allons sélectionnés uniquement ceux dont leur fréquence dans le corpus est supérieur ou égale à la valeur affectée au second paramètre `min_df`. En conséquent, nous pouvons éliminer les features non pertinents afin d'affiner ceux que nous avons sélectionnés afin d'obtenir une meilleure précision par nos modèles.

---
## 3. Optimisation 
---

### 3.1 WordCloud

Afin d'analyser visuellement notre donnée, nous avons utiliser **WordCloud** qui nous permet d'afficher les mots les plus fréquents dans les avis positifs et négatifs [Voir figure 1 & 2]. Cette visualisation nous permettra de mieux configurer les fonctions de prétraitement, notamment la liste des stopwords qui pourra être améliorer. 

*Positif*  
[Figure 1]

*Négatif*  
[Figure 2]

De ces visualisations, nous avons remarqué des mots à polariter négative dans les avis positifs et inversement. Nous sommes tombés sur deux conclusions qui sont les suivantes :

- Il y a beaucoup d'irones dans les avis.
- Les avis sont de grandes tailles et il faudrait faire une analyse de phrase subjective parmi ces multiples phrases que se compose ces avis.

Nous avons pas eut le temps d'implémenter le traitement de ces deux cas qui nous auraient augmenté la précision de nos modèles.

### 3.2 Une possibilité de traitement de l'ironie

- https://github.com/MirunaPislar/Sarcasm-Detection

---
## Conclusion
---