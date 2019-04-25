## Perspectives
---

### NER

Nous pouvons pousser plus loin notre système de TAG afin d'étiquetter les personnages publiques et les lieux pour réduire leur poids dans notre ensemble de features afin d'améliorer la perception du sentiment de la phrase.

### Traitement des amplificateurs

Le traitement des ponctuations et des majuscules peuvent servir à amplifier la certitude de la polarité d'une phrase.

### Extraction des phrases subjectives

Dans la classifications des sentiments, il est plus important de se focaliser sur les phrases subjectives et filtrer ceux qui sont objectives car elles expriment plus le sentiment de l'auteur.

### SentiWordNet

Ce module est un moyen simplifié de discerner le sarcasme dans une phrase. En effet, ce module utilise un lexicon permettant d'attribuer un sentiment (positif, négatif, neutre) à chaque mot de nos documents. Cela nous permettrait de trouver un motif récurrent dans l'ironie où deux mots de polarités extrêmes sont contenus dans la phrase, comme dans l'exemple suivant