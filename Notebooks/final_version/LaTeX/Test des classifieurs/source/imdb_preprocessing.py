#PREPROCESSING OF IMDB DATASET FOR VISUALIZATION
imdb_transformed = imdb_df.copy()
imdb_transformed['Avis'] = preprocess_dataset(imdb_transformed['Avis'])
display(imdb_transformed['Avis'].head())
