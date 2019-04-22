#PREPROCESSING DATASET
df_transformed = df.copy() #creating a new copy of the dataset that will be preprocessed
df_transformed['Avis'] = preprocess_dataset(df_transformed['Avis']) #preprocessing of opinions column
display(df_transformed['Avis'].head())
