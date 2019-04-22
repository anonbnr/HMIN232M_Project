#VECTORIZING
#splitting the dataset prior to vectorization, to prevent memory-related errors during processing
df_first_partition = df_transformed.iloc[:5000]
df_second_partition = df_transformed.iloc[5000:]

#vectorization of the opinions column
vectorizer = TfidfVectorizer(min_df=12, ngram_range=(1,2))
vectors = vectorizer.fit_transform(df_transformed['Avis'])
